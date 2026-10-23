"""Fallback delivery for GatewayStreamConsumer: continuation sends after edits
stop working, chunking, cursor cleanup, commentary and silence retraction."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable, Optional

from gateway.platforms.base import BasePlatformAdapter as _BasePlatformAdapter, _delivery_failure_error
from gateway.delivery_handoff import normalize_delivery_result
from gateway.stream_consumer_fences import ensure_closed_code_fences

logger = logging.getLogger("gateway.stream_consumer")


class StreamFallbackMixin:
    """Non-streaming delivery paths used once progressive edits fail or the turn ends oddly."""

    async def _send_new_chunk(self, text: str, reply_to_id: Optional[str], *,
                              final: bool = False, later_chunks: tuple[str, ...] = ()) -> Optional[str]:
        """Send a new chunk threaded to ``reply_to_id``; returns the new message_id."""
        if self._delivery_handoff is not None and self._delivery_handoff.blocked:
            return None
        text = self._clean_for_display(text)
        if not text.strip():
            return reply_to_id
        try:
            result = await self.adapter.send(
                chat_id=self.chat_id, content=text, reply_to=reply_to_id,
                metadata=self._metadata_for_send(final=final, expect_edits=not final))
            result = normalize_delivery_result(result)
            if not result.success:
                from gateway.relay.egress import declined_send
                if declined_send(result):
                    self._retain_egress_decline(
                        result, attempted_content=text, later_chunks=later_chunks)
                else:
                    self._retain_delivery_remainder(
                        self._clean_for_display(self._stream_ledger or text), result,
                        later_chunks=later_chunks, attempted_content=text)
                    return reply_to_id
                self._edit_supported = False
                return reply_to_id
            if not (result.success and result.message_id):
                self._edit_supported = False
                return reply_to_id
            self._retain_delivery_remainder(
                self._clean_for_display(self._stream_ledger or text), result,
                later_chunks=later_chunks, attempted_content=text)
            self._message_id = str(result.message_id)
            self._track_preview_ids_from_result(result)
            self._already_sent = True
            self._last_sent_text = text
            self._last_sent_cursor_appended = False
            self._notify_new_message()
            return str(result.message_id)
        except Exception as e:
            from gateway.platforms.base import SendResult
            self._retain_delivery_remainder(
                self._clean_for_display(self._stream_ledger or text),
                SendResult(success=False, error=str(e), raw_response={
                    "delivery_ambiguous": True, "remaining_content": text,
                    "model_remaining_content": text, "attachment_remaining_content": ""}),
                later_chunks=later_chunks, attempted_content=text)
            logger.error("Stream send chunk error: %s", e)
            return reply_to_id

    def _visible_prefix(self) -> str:
        """Return the visible text already shown in the streamed message."""
        prefix = self._last_sent_text or ""
        if (self.cfg.cursor and getattr(self, "_last_sent_cursor_appended", False)
                and prefix.endswith(self.cfg.cursor)):
            prefix = prefix[:-len(self.cfg.cursor)]
        return self._clean_for_display(prefix)

    def _continuation_text(self, final_text: str) -> str:
        """Return only the part of final_text the user has not already seen."""
        prefix = self._fallback_prefix or self._visible_prefix()
        if prefix and final_text.startswith(prefix):
            cut = len(prefix)
            # ``prefix`` is whatever the last successful edit put on screen. Edits
            # fire on a throttle tick, not at a word boundary, so that prefix can
            # end inside a word.  Back the cut up to the last space or newline so
            # the continuation re-sends the broken word's tail and reads as an
            # ordinary continuation.  A prefix with no boundary (one very long
            # token) keeps the original cut rather than re-sending the whole reply.
            # A prefix that already ends on a whole word needs no back-up: doing it
            # re-sent that word at the seam.
            if cut < len(final_text) and not final_text[cut].isspace() and not final_text[cut - 1].isspace():
                boundary = max(
                    final_text.rfind(" ", 0, cut),
                    final_text.rfind("\n", 0, cut),
                )
                if boundary >= 0:
                    cut = boundary + 1
            return final_text[cut:].lstrip()
        return final_text

    @staticmethod
    def _split_text_chunks(text: str, limit: int, len_fn: "Callable[[str], int]" = len,
                           ) -> list[str]:
        """Split text for fallback sends: newline-preferred, fence-balanced across chunks."""
        from gateway.platforms.helpers import split_text_fence_aware
        return split_text_fence_aware(text, limit, len_fn, prefer_paragraphs=False,
                                      balance_fences=True)

    def _truncate_for_stream(self, text: str, limit: int, len_fn: "Callable[[str], int]",
                             ) -> list[str]:
        """Split via the adapter's canonical truncate_message (platform-specific rules);
        non-base test doubles / legacy adapters keep the two-argument call shape."""
        truncate = getattr(self.adapter, "truncate_message", None)
        if not callable(truncate):
            return self._split_text_chunks(text, limit, len_fn)
        if isinstance(self.adapter, _BasePlatformAdapter):
            chunks = truncate(text, limit, len_fn=len_fn)
        else:
            chunks = truncate(text, limit)
        if not isinstance(chunks, (list, tuple)) or not all(isinstance(c, str) for c in chunks):
            return self._split_text_chunks(text, limit, len_fn)
        return list(chunks)

    async def _send_fallback_final(self, text: str) -> None:
        """Send the final continuation after streaming edits stop working (one flood retry
        per chunk)."""
        if self._delivery_handoff is not None and self._delivery_handoff.blocked:
            return
        if getattr(self, "_egress_declined", False):
            # The connector refused this destination earlier in the run. The
            # whole point of this path is to deliver the unseen tail as a NEW
            # message, which is exactly the re-addressing the egress guard
            # exists to stop.
            logger.warning(
                "suppressing the fallback continuation: the connector already "
                "declined this destination for this run"
            )
            return
        # Balance fences BEFORE computing the continuation so the closing fence
        # reaches the user even when only the tail is delivered.
        final_text = ensure_closed_code_fences(self._clean_for_display(text))
        continuation = self._continuation_text(final_text)
        self._fallback_final_send = False
        if not continuation.strip():
            continuation = await self._fallback_when_nothing_unseen(final_text)
            if continuation is None:
                return

        _len_fn, raw_limit = self._fallback_len_budget()
        chunks = self._split_text_chunks(continuation, max(500, raw_limit - 100), len_fn=_len_fn)

        stale_message_id = self._message_id  # partial message to clean up
        last_message_id: Optional[str] = None
        sent_any_chunk = False
        # Thread only a FULL resend (it replaces the preview); a tail continuation
        # keeps its existing unthreaded delivery on every platform.
        anchor = self._initial_reply_to_id if continuation == final_text else None
        for index, chunk in enumerate(chunks):
            try:
                result = await self._send_with_flood_retry(
                    content=chunk, reply_to=None if sent_any_chunk else anchor,
                    retry_log="Flood control on fallback send, retrying in %.1fs")
                result = normalize_delivery_result(result)
            except Exception as exc:
                from gateway.platforms.base import SendResult
                result = SendResult(success=False, error=str(exc), raw_response={
                    "delivery_ambiguous": True, "remaining_content": chunk,
                    "model_remaining_content": chunk, "attachment_remaining_content": ""})
            if not result or not result.success:
                from gateway.relay.egress import declined_send
                if result and declined_send(result):
                    self._retain_egress_decline(
                        result, source_content=final_text, attempted_content=chunk,
                        later_chunks=tuple(chunks[index + 1:]))
                    return
                if result is None:
                    from gateway.platforms.base import SendResult
                    result = SendResult(success=False, error="send returned no receipt", raw_response={
                        "delivery_ambiguous": True, "remaining_content": chunk,
                        "model_remaining_content": chunk, "attachment_remaining_content": ""})
                self._retain_delivery_remainder(
                    final_text, result, later_chunks=tuple(chunks[index + 1:]),
                    attempted_content=chunk)
                return
            sent_any_chunk = True
            last_message_id = result.message_id or last_message_id
            self._retain_delivery_remainder(
                final_text, result, later_chunks=tuple(chunks[index + 1:]),
                attempted_content=chunk)
            self._notify_new_message()

        # Best-effort delete of the frozen partial — ONLY when the FULL final was
        # re-sent.  If only the missing tail went out, the partial IS the head of
        # the answer ("sent only the second half" symptom).
        if (stale_message_id and stale_message_id != last_message_id
                and not self._fallback_preserve_partial_messages and continuation == final_text):
            deleted = await self._delete_previews([stale_message_id], label="Fallback partial",
                                                  skip_sentinel=False)
            if self._delivery_handoff is not None:
                self._delivery_handoff = self._delivery_handoff.without_deleted(deleted)

        self._message_id = last_message_id
        self._already_sent = True
        # Recorder substitutes the unsplit ledger on a split turn.
        self._mark_final_delivered(record=final_text)
        self._last_sent_text = chunks[-1]
        self._last_sent_cursor_appended = False
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False

    async def _fallback_when_nothing_unseen(self, final_text: str) -> Optional[str]:
        """Fallback entered but the visible prefix already covers ``final_text``: returns the
        continuation to send (the whole final when the prefix is from a *previous* segment)
        or None when the turn is settled here."""
        visible = self._visible_prefix()
        # Telegram clients can lose (part of) a streamed preview after a failed
        # final edit, so opt-in adapters commit a fresh final send.
        if (final_text.strip() and final_text == visible
                and getattr(self.adapter, "RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK", False) is True):
            delivery = await self._send_empty_fallback_final(final_text)
            if self._delivery_handoff is not None and self._delivery_handoff.blocked:
                return None
            if delivery == "delivered":
                return None
            # Only an already-acknowledged preview can stand in for a
            # definitively flood-rejected cosmetic commit.
            if delivery == "preview":
                self._already_sent = True
                self._fallback_prefix = ""
                self._fallback_preserve_partial_messages = False
                self._final_content_delivered = True
                self._record_turn_final_payload(final_text)
            return None
        # The prefix may be from a *previous* segment (before a tool boundary),
        # wrongly reading as "already shown" — send final_text as-is.
        if final_text.strip() and final_text != visible:
            return final_text
        # Best-effort strip of a cursor left stuck by the edit failure.
        if (self._message_id and self._last_sent_text and self.cfg.cursor
                and getattr(self, "_last_sent_cursor_appended", False)
                and self._last_sent_text.endswith(self.cfg.cursor)):
            clean_text = self._last_sent_text[:-len(self.cfg.cursor)]
            with contextlib.suppress(Exception):
                result = await self._edit_message(message_id=self._message_id, content=clean_text)
                if result.success:
                    self._retain_delivery_remainder(
                        self._clean_for_display(self._stream_ledger or clean_text), result,
                        edited_message_id=str(self._message_id), attempted_content=clean_text)
                    self._last_sent_text = clean_text
                    self._last_sent_cursor_appended = False
        self._already_sent = True
        # Recorder substitutes the full ledger on a split turn.
        self._mark_final_delivered(record=final_text)
        return None

    def _fallback_len_budget(self) -> "tuple[Callable[[str], int], int]":
        """(len_fn, raw_limit) for fallback chunking — per-chat cap/unit on base adapters."""
        raw_limit = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        _len_fn: "Callable[[str], int]" = len
        if isinstance(self.adapter, _BasePlatformAdapter):
            _len_fn = self.adapter.message_len_fn
            try:  # per-chat cap/unit (relay adapter fronting N platforms)
                raw_limit = self.adapter.max_message_length_for_chat(self.chat_id)
                _len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("per-chat limit resolution failed: %s", e)
        return _len_fn, raw_limit

    async def _send_with_flood_retry(self, *, content: str, retry_log: str, reply_to=None):
        """adapter.send(final metadata) with ONE bounded flood retry; returns the last
        SendResult.  Exceptions propagate (callers decide whether a raise is "ambiguous")."""
        kwargs = dict(chat_id=self.chat_id, content=content,
                      metadata=self._metadata_for_send(final=True))
        if reply_to is not None:
            kwargs["reply_to"] = reply_to
        result = None
        for attempt in range(2):
            result = await self.adapter.send(**kwargs)
            result = normalize_delivery_result(result)
            if getattr(result, "success", False):
                break
            from gateway.relay.egress import declined_send
            if declined_send(result):
                break  # no retry may re-address a destination the connector refused
            raw = getattr(result, "raw_response", None)
            if isinstance(raw, dict) and raw.get("delivery_ambiguous"):
                break
            remainder = raw.get("remaining_content") if isinstance(raw, dict) else None
            if isinstance(remainder, str) and remainder:
                kwargs["content"] = remainder
            elif kwargs["content"] != content:
                result.raw_response = {
                    **(raw if isinstance(raw, dict) else {}),
                    "remaining_content": kwargs["content"],
                }
            retry_delay = self._fallback_flood_retry_delay(result)
            if attempt or retry_delay is None:
                break  # non-flood error, long flood wait, or second failure
            if isinstance(raw, dict) and raw.get("partial_overflow"):
                break  # split head already on screen: re-sending the whole content duplicates it
            logger.debug(retry_log, retry_delay)
            await asyncio.sleep(retry_delay)
        return result

    async def _send_empty_fallback_final(self, final_text: str) -> str:
        """Commit a completed answer after Telegram finalization fails: return "delivered",
        "failed" (including uncertain acknowledgements), or "preview" when a definite
        flood rejection leaves the already-complete preview authoritative."""
        # Segment-scoped only: never delete an earlier finalized preamble.
        stale_ids = self._stale_preview_ids(segment_only=True)
        try:
            result = await self._send_with_flood_retry(
                content=final_text, reply_to=self._initial_reply_to_id,
                retry_log="Flood control on empty fallback final send; retrying in %.1fs")
        except Exception as exc:
            from gateway.platforms.base import SendResult
            failed = normalize_delivery_result(
                SendResult(success=False, error=str(exc) or type(exc).__name__), exception=exc)
            self._retain_delivery_remainder(
                final_text, SendResult(success=False, error=failed.error, raw_response={
                    "delivery_ambiguous": (failed.raw_response or {}).get("delivery_ambiguous", False),
                    "remaining_content": final_text, "model_remaining_content": final_text,
                    "attachment_remaining_content": ""}))
            logger.debug("Empty fallback final send failed: %s", exc)
            return "failed"
        if not getattr(result, "success", False):
            raw = result.raw_response if isinstance(getattr(result, "raw_response", None), dict) else {}
            if (self._is_flood_error(result) and not raw.get("remaining_content")
                    and self._delivery_handoff is not None and self._delivery_handoff.complete
                    and self._visible_prefix() == final_text):
                return "preview"
            self._retain_delivery_remainder(final_text, result)
            return "failed"

        new_message_id = getattr(result, "message_id", None)
        # Telegram reports delete failure by returning False; the flood window that
        # broke the finalize can reject this too — one bounded retry.
        deleted = await self._delete_previews(stale_ids, skip=new_message_id, label="Empty fallback",
                                               retry_on_false=True)
        if self._delivery_handoff is not None:
            self._delivery_handoff = self._delivery_handoff.without_deleted(deleted)
        self._retain_delivery_remainder(final_text, result, attempted_content=final_text)
        self._segment_preview_message_ids = set()
        self._message_id = new_message_id or "__no_edit__"
        self._already_sent = True
        self._mark_final_delivered()
        # Record VERBATIM, not via _record_turn_final_payload: the sealed previews
        # were just deleted, so the ledger (still holding sealed heads) would claim
        # delivery for text this path removed.
        self._delivered_final_text = self._display_payload(final_text)
        self._last_sent_text = final_text
        self._last_sent_cursor_appended = False
        self._fallback_prefix = ""
        self._fallback_preserve_partial_messages = False
        self._notify_new_message()
        return "delivered"


    def _fallback_flood_retry_delay(self, result: Any) -> float | None:
        """Return a bounded retry delay for a fallback send, if safe to retry."""
        if not self._is_flood_error(result):
            return None
        try:
            delay = float(getattr(result, "retry_after", None) or 3.0)
        except (TypeError, ValueError):
            delay = 3.0
        if delay > self._max_fallback_flood_retry_seconds:
            logger.debug("Flood control requests %.1fs; leaving final delivery to the gateway",
                         delay)
            return None
        return max(0.0, delay)

    def _is_flood_error(self, result) -> bool:
        """Check the underlying transport failure, not aggregate display text."""
        err_lower = _delivery_failure_error(result).lower()
        return "flood" in err_lower or "retry after" in err_lower or "rate" in err_lower

    async def _flush_segment_tail_on_edit_failure(self) -> None:
        """Before a segment reset, send the unseen tail as a new message (and best-effort
        strip the stuck cursor from the partial)."""
        if self._delivery_handoff is not None and self._delivery_handoff.blocked:
            return
        if getattr(self, "_egress_declined", False):
            return  # a new message is exactly the re-addressing the egress guard refused
        if not self._fallback_final_send:
            await self._try_strip_cursor()
            if self._delivery_handoff is not None and self._delivery_handoff.blocked:
                return
        visible = self._fallback_prefix or self._visible_prefix()
        tail = self._accumulated
        if visible and tail.startswith(visible):
            tail = tail[len(visible):]
        tail = self._clean_for_display(tail)
        if not tail.strip():
            return
        try:
            # Interim: must never seal a native stream (see _send_commentary).
            _md = dict(self.metadata) if self.metadata else {}
            _md["_interim_send"] = True
            result = await self.adapter.send(chat_id=self.chat_id, content=tail, metadata=_md)
            result = normalize_delivery_result(result)
            if result.success:
                self._already_sent = True
            self._retain_delivery_remainder(
                self._clean_for_display(self._stream_ledger), result, attempted_content=tail)
        except Exception as e:
            from gateway.platforms.base import SendResult
            self._retain_delivery_remainder(
                self._clean_for_display(self._stream_ledger),
                SendResult(success=False, error=str(e), raw_response={
                    "delivery_ambiguous": True, "remaining_content": tail,
                    "model_remaining_content": tail, "attachment_remaining_content": ""}),
                attempted_content=tail)
            logger.error("Segment-break tail flush error: %s", e)

    async def _try_strip_cursor(self) -> None:
        """Best-effort edit removing a stuck cursor when entering fallback mode."""
        prefix = self._visible_prefix()
        if (not self._has_real_preview() or not prefix.strip()
                or not self._last_sent_cursor_appended):
            return
        with contextlib.suppress(Exception):  # never block the fallback path
            result = await self._edit_message(message_id=self._message_id, content=prefix)
            if getattr(result, "success", False):
                self._retain_delivery_remainder(
                    self._clean_for_display(self._stream_ledger or prefix), result,
                    edited_message_id=str(self._message_id), attempted_content=prefix)
                self._last_sent_text = prefix
                self._last_sent_cursor_appended = False
            else:
                from gateway.relay.egress import declined_send
                if declined_send(result):
                    self._retain_egress_decline(
                        result, attempted_content=prefix, edited_message_id=str(self._message_id))

    async def _send_commentary(self, text: str) -> bool:
        """Send a completed interim assistant commentary message."""
        if self._delivery_handoff is not None and self._delivery_handoff.blocked:
            return False
        text = self._clean_for_display(text)
        if not text.strip():
            return False
        try:
            # Interim: a stream-is-the-message adapter's seal-interception must not
            # turn this into draft(final=true), which would seal the live stream
            # with interim text and orphan the true final.
            _md = self._metadata_for_send(final=False) or {}
            _md["_interim_send"] = True
            # reply_to only for reply-anchored threading; Discord/Telegram use
            # thread_id metadata and reply_to on every commentary is spam.
            _plat = getattr(getattr(self.adapter, "platform", None), "value", None)
            _platform_name = str(_plat or getattr(self.adapter, "name", "")).lower()
            _needs_reply_anchor = _platform_name in ("buzz", "slack", "mattermost", "feishu")
            result = await self.adapter.send(
                chat_id=self.chat_id, content=text,
                reply_to=self._initial_reply_to_id if _needs_reply_anchor else None, metadata=_md)
            result = normalize_delivery_result(result)
            # Do NOT set _already_sent: commentary is interim, and the flag would
            # suppress the real final after multiple tool calls.
            self._retain_delivery_remainder(text, result, attempted_content=text)
            if result.success:
                self._notify_new_message()
                # Lets run.py confirm whether an interim send carried the final.
                # Record the exact delivered text so run.py can confirm whether an interim "preview"
                # actually carried the final response, vs. unrelated commentary delivered during a session
                # split (#14238).
                self._delivered_commentary_texts.append(text)
            return result.success
        except Exception as e:
            from gateway.platforms.base import SendResult
            self._retain_delivery_remainder(
                text, SendResult(success=False, error=str(e), raw_response={
                    "delivery_ambiguous": True, "remaining_content": text,
                    "model_remaining_content": text, "attachment_remaining_content": ""}))
            logger.error("Commentary send error: %s", e)
            return False

    def _raw_message_limit(self) -> int:
        """Per-chat length budget (``message_len_fn`` units) before overflow splits; rich
        adapters may raise it via ``streaming_overflow_limit`` so a reply that fits one rich
        message isn't fragmented at the edit limit."""
        base = getattr(self.adapter, "MAX_MESSAGE_LENGTH", 4096)
        # isinstance gate keeps MagicMock adapters (mock attrs, not ints) on base.
        if isinstance(self.adapter, _BasePlatformAdapter):
            try:
                base = self.adapter.max_message_length_for_chat(self.chat_id)
            except Exception as e:
                logger.debug("max_message_length_for_chat failed: %s", e)
            try:
                cap = self.adapter.streaming_overflow_limit()
            except Exception as e:
                logger.debug("streaming_overflow_limit check failed: %s", e)
                cap = None
            if isinstance(cap, int) and cap > base:
                return cap
        return base

    # Fresh send carried exactly ``text`` — record it so the gateway can reconcile the flag against the
    # completed response (#71643/#95382 content-vs-flag contract).
    async def _suppress_silence_marker(self) -> None:
        """Retract any streamed preview when the final reply is a bare silence marker.  Flags
        stay False so the gateway's whole-response filter owns what goes out next: "" for a
        machinery turn, the visible fallback for a human one."""
        # A native-stream bubble isn't a deletable message — close an open one
        # (e.g. from an eager re-seed) with an empty finalize so it doesn't hang.
        if self._native_stream_opened:
            await self._close_empty_native_bubble("Silence-marker native stream close failed: %s")

        deleted = await self._delete_previews(self._stale_preview_ids(), label="Silence-marker")
        if self._delivery_handoff is not None:
            self._delivery_handoff = self._delivery_handoff.without_deleted(deleted)
        self._preview_message_ids = set()
        self._message_id = None
        self._accumulated = self._stream_ledger = self._last_sent_text = ""
        self._last_sent_cursor_appended = False
        self._already_sent = False
        self._clear_turn_final_flags()
        logger.info("Suppressed streamed intentional-silence marker (chat=%s)", self.chat_id)
