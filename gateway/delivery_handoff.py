"""Turn-scoped delivery receipts: acknowledged carriages, owed payload, and uncertainty."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gateway.relay.egress import EGRESS_DECLINE_CODE, declined_send

if TYPE_CHECKING:
    from gateway.platforms.base import SendResult


@dataclass(frozen=True)
class StreamDeliveryHandoff:
    """The receipt of one turn, including fully acknowledged turns."""
    source_content: str
    result: SendResult
    settled: bool = False
    address: dict | None = None

    def with_address(self, address: dict | None) -> StreamDeliveryHandoff:
        if address is None:
            return replace(self, address=None)
        if not isinstance(address, dict):
            raise ValueError("delivery address must be an object")
        keys = ("session_id", "row_id", "turn_id", "disposition_hint")
        if any(key not in address for key in keys):
            raise ValueError("incomplete delivery address")
        if (any(not isinstance(address[key], str) or not address[key]
                for key in keys if key != "row_id")
                or type(address["row_id"]) is not int or address["row_id"] <= 0):
            raise ValueError("invalid delivery address")
        return replace(self, address={key: address[key] for key in keys})

    def to_record(self) -> dict:
        raw = self.result.raw_response if isinstance(self.result.raw_response, dict) else {}
        return {
            "version": 1, "source_content": self.source_content,
            "success": bool(self.result.success),
            "message_id": str(self.result.message_id) if self.result.message_id is not None else None,
            "acknowledged_carriages": [
                {"message_id": item["message_id"], "text": item["text"]}
                for item in raw.get("acknowledged_carriages", ())],
            "model_remaining_content": raw.get("model_remaining_content", ""),
            "attachment_remaining_content": raw.get("attachment_remaining_content", ""),
            "delivery_ambiguous": self.ambiguous,
            "error": self.result.error, "error_kind": self.result.error_kind,
            "egress_declined": declined_send(self.result),
            "retryable": bool(self.result.retryable), "retry_after": self.result.retry_after,
            "stream_cursor": raw.get("stream_cursor", ""),
            **({"cursor_carriage": dict(raw["cursor_carriage"])}
               if isinstance(raw.get("cursor_carriage"), dict) else {}),
            "pending_chunks": bool(raw.get("pending_chunks")),
            "empty_projection": self.result.delivered_text == "",
            **({"transport_chunks": list(raw["transport_chunks"])}
               if isinstance(raw.get("transport_chunks"), (list, tuple)) else {}),
            "settled": self.settled,
            "address": dict(self.address) if self.address is not None else None,
        }

    @classmethod
    def from_record(cls, record: dict) -> StreamDeliveryHandoff:
        from gateway.platforms.base import SendResult
        if not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != 1:
            raise ValueError("unsupported delivery record")
        strings = ("source_content", "model_remaining_content", "attachment_remaining_content", "stream_cursor")
        booleans = ("success", "delivery_ambiguous", "settled", "retryable")
        nullable = ("message_id", "error", "error_kind")
        if any(key not in record for key in (*nullable, "retry_after")):
            raise ValueError("incomplete delivery record")
        if (any(not isinstance(record.get(key), str) for key in strings)
                or any(type(record.get(key)) is not bool for key in booleans)
                or any(record.get(key) is not None and not isinstance(record[key], str) for key in nullable)
                or type(record.get("pending_chunks")) is not bool
                or "address" not in record
                or record.get("retry_after") is not None
                and (type(record["retry_after"]) not in (float, int))):
            raise ValueError("invalid delivery record")
        if type(record.get("egress_declined", False)) is not bool:
            raise ValueError("invalid delivery decline")
        if (type(record.get("empty_projection", False)) is not bool
                or ("transport_chunks" in record and (
                    not isinstance(record["transport_chunks"], list)
                    or any(not isinstance(chunk, str) for chunk in record["transport_chunks"])))):
            raise ValueError("invalid delivery projection")
        cursor_carriage = record.get("cursor_carriage")
        if cursor_carriage is not None and (
                not isinstance(cursor_carriage, dict)
                or set(cursor_carriage) != {"message_id", "text"}
                or cursor_carriage["message_id"] is not None
                and not isinstance(cursor_carriage["message_id"], str)
                or not isinstance(cursor_carriage["text"], str)):
            raise ValueError("invalid cursor carriage")
        carriages = record.get("acknowledged_carriages")
        if not isinstance(carriages, list) or any(
            not isinstance(item, dict) or set(item) != {"message_id", "text"}
            or (item["message_id"] is not None and not isinstance(item["message_id"], str))
            or not isinstance(item["text"], str) for item in carriages
        ):
            raise ValueError("invalid acknowledged carriages")
        model = record["model_remaining_content"]
        attachment = record["attachment_remaining_content"]
        owed = model + ("\n" if model and attachment and not attachment.startswith(("\n", " ")) else "") + attachment
        texts = [item["text"] for item in carriages if item["text"]]
        ids = [item["message_id"] for item in carriages if item["message_id"]]
        result = SendResult(
            success=record["success"], message_id=record.get("message_id"),
            error=record.get("error"), error_kind=record.get("error_kind"),
            retryable=record["retryable"], retry_after=record.get("retry_after"),
            delivered_text="\n".join(texts) if texts else (
                "" if record.get("empty_projection") else None),
            raw_response={
                "acknowledged_carriages": [dict(item) for item in carriages],
                "message_ids": ids, "acknowledged_texts": texts,
                "model_remaining_content": model, "attachment_remaining_content": attachment,
                "remaining_content": owed, "delivery_ambiguous": record["delivery_ambiguous"],
                "stream_cursor": record["stream_cursor"], "pending_chunks": record["pending_chunks"],
                **({"code": EGRESS_DECLINE_CODE} if record.get("egress_declined") else {}),
                **({"cursor_carriage": dict(cursor_carriage)}
                   if cursor_carriage is not None else {}),
                **({"transport_chunks": tuple(record["transport_chunks"])}
                   if "transport_chunks" in record else {}),
                "stream_handoff": True,
                "partial_overflow": bool((owed or not record["success"]) and carriages),
            },
        )
        return cls(record["source_content"], result, record["settled"]).with_address(record.get("address"))

    @property
    def owed(self) -> str:
        return self.result.raw_response.get("remaining_content", "")

    def formatted_transport_chunks(self, owed: str) -> tuple[str, ...] | None:
        """Only exact owed bytes may bypass Telegram's source formatter."""
        chunks = self.result.raw_response.get("transport_chunks")
        if (isinstance(chunks, (tuple, list)) and chunks
                and all(isinstance(chunk, str) for chunk in chunks)
                and "\n".join(chunks) == owed):
            return tuple(chunks)
        return None

    @property
    def ambiguous(self) -> bool:
        return bool(self.result.raw_response.get("delivery_ambiguous"))

    @property
    def complete(self) -> bool:
        return not self.owed and not self.ambiguous and self.result.success

    @property
    def blocked(self) -> bool:
        """Only a proven unattempted split tail may continue inline."""
        return self.ambiguous or (not self.complete and not self.result.raw_response.get("pending_chunks"))

    def without_deleted(self, message_ids: Iterable[str]) -> StreamDeliveryHandoff:
        """Remove only carriages whose platform deletion was confirmed."""
        deleted = set(message_ids)
        if not deleted:
            return self
        raw = self.result.raw_response
        carriages = [dict(item) for item in raw.get("acknowledged_carriages", ())
                     if item["message_id"] not in deleted]
        texts = [item["text"] for item in carriages if item["text"]]
        ids = [item["message_id"] for item in carriages if item["message_id"]]
        updated_raw = {**raw, "acknowledged_carriages": carriages,
                       "acknowledged_texts": texts, "message_ids": ids}
        if updated_raw.get("cursor_carriage") not in carriages:
            updated_raw.pop("cursor_carriage", None)
        return replace(self, result=replace(
            self.result, delivered_text="\n".join(texts) if texts else None,
            raw_response=updated_raw))

    def reserve_model(self, remainder: str, source_content: str) -> StreamDeliveryHandoff:
        """Record a sealed head's not-yet-attempted tail without inventing an ACK."""
        raw = self.result.raw_response
        return replace(self, source_content=source_content,
                       result=replace(self.result, success=False, raw_response={
            **raw, "model_remaining_content": remainder,
            "attachment_remaining_content": raw.get("attachment_remaining_content", ""),
            "remaining_content": remainder,
            "pending_chunks": True,
            "partial_overflow": bool(raw.get("message_ids") or self.result.delivered_text),
        }))


def normalize_delivery_result(receipt: SendResult, *,
                              exception: BaseException | None = None) -> SendResult:
    """Retain uncertainty from timeouts that may have reached the platform."""
    from gateway.platforms.base import _delivery_failure_error
    if receipt.success:
        return receipt
    # Explicit pre-send proof wins over textual timeout classification; retryable
    # by itself is not such proof.
    error = _delivery_failure_error(receipt).lower()
    kind = (receipt.error_kind or "").lower()
    raw = receipt.raw_response if isinstance(receipt.raw_response, dict) else {}
    if raw.get("delivery_ambiguous") is False:
        return receipt
    if not isinstance(exception, TimeoutError) and not (
            exception is not None and not str(exception)) and not any(
            term in error or term in kind for term in ("timeout", "timed out", "deadline exceeded")):
        return receipt
    return replace(receipt, raw_response={**raw, "delivery_ambiguous": True})


def _receipt_carriages(receipt: SendResult, attempted_content: str | None) -> list[dict]:
    raw = receipt.raw_response if isinstance(receipt.raw_response, dict) else {}
    if raw.get("skipped"):
        return []
    nested = raw.get("ambient_carriages")
    if isinstance(nested, (tuple, list)):
        return [item for part in nested for item in _receipt_carriages(part, None)]
    if not receipt.success and not raw.get("partial_overflow") and not receipt.delivered_text:
        return []
    pairs = raw.get("acknowledged_carriages")
    if isinstance(pairs, (list, tuple)):
        return [{"message_id": str(item["message_id"]) if item.get("message_id") is not None else None,
                 "text": item["text"]} for item in pairs
                if isinstance(item, dict) and isinstance(item.get("text"), str)]
    ids = raw.get("message_ids")
    if not isinstance(ids, (list, tuple)):
        ids = (receipt.message_id,) if receipt.message_id else ()
    chunks = raw.get("acknowledged_chunks")
    if isinstance(chunks, (list, tuple)) and len(chunks) == len(ids):
        return [{"message_id": str(mid) if mid is not None else None, "text": text}
                for mid, text in zip(ids, chunks)]
    text = receipt.delivered_text
    if text is None and receipt.success:
        text = attempted_content
    if len(ids) == 1:
        return [{"message_id": str(ids[0]) if ids[0] is not None else None,
                 "text": text if isinstance(text, str) else ""}]
    if ids:
        # Flattened text does not identify which ID carried which bytes.
        return [{"message_id": str(mid) if mid is not None else None, "text": ""}
                for mid in ids]
    return [{"message_id": None, "text": text}] if isinstance(text, str) and text else []


def delivery_handoff_from_receipt(
    source_content: str, receipt: SendResult, *,
    previous: StreamDeliveryHandoff | None = None,
    acknowledged: Iterable[SendResult] = (), later_chunks: tuple[str, ...] = (),
    cursor: str = "", edited_message_id: str | None = None,
    attempted_content: str | None = None, resolve_attempt: bool = False,
    active_model_frame: str | None = None,
) -> StreamDeliveryHandoff:
    """Reduce a transport receipt into the turn owner; preserve the first identity."""
    from gateway.platforms.base import SendResult as _SendResult
    if not isinstance(receipt, _SendResult):
        receipt = _SendResult(
            success=bool(getattr(receipt, "success", False)),
            message_id=getattr(receipt, "message_id", None),
            error=getattr(receipt, "error", None),
            raw_response=getattr(receipt, "raw_response", None),
            retryable=bool(getattr(receipt, "retryable", False)),
            retry_after=getattr(receipt, "retry_after", None),
            continuation_message_ids=getattr(receipt, "continuation_message_ids", ()),
            error_kind=getattr(receipt, "error_kind", None),
            delivered_text=getattr(receipt, "delivered_text", None),
        )
    receipt = normalize_delivery_result(receipt)
    declined = declined_send(receipt) or bool(previous and declined_send(previous.result))
    raw = receipt.raw_response if isinstance(receipt.raw_response, dict) else {}
    if raw.get("skipped") and previous is not None:
        return previous
    prior = previous.result if previous else None
    prior_raw = prior.raw_response if prior and isinstance(prior.raw_response, dict) else {}
    carriages: list[dict] = [dict(item) for item in prior_raw.get("acknowledged_carriages", ())]
    if not carriages and prior:
        carriages = _receipt_carriages(prior, None)
    # A confirmed platform deletion removes exactly those carriages; any other
    # visible message (an independently sent GIF) stays acknowledged.
    deleted = {str(mid) for mid in raw.get("deleted_message_ids") or ()}
    if deleted:
        carriages = [item for item in carriages if item["message_id"] not in deleted]
    if not raw.get("skipped"):
        for result in (*acknowledged, receipt):
            for item in _receipt_carriages(result, attempted_content if result is receipt else None):
                message_id = (edited_message_id if item["message_id"] is None
                              and edited_message_id and result is receipt and receipt.success
                              else item["message_id"])
                if message_id != item["message_id"]:
                    item = {**item, "message_id": message_id}
                if edited_message_id and message_id == edited_message_id:
                    index = next((i for i, existing in enumerate(carriages)
                                  if existing["message_id"] == edited_message_id), None)
                    if index is not None:
                        carriages[index] = item
                        continue
                if message_id and any(existing["message_id"] == message_id for existing in carriages):
                    continue
                carriages.append(item)
    ids = [item["message_id"] for item in carriages if item["message_id"]]
    texts = [item["text"] for item in carriages if item["text"]]
    effective_cursor = cursor or prior_raw.get("stream_cursor", "")
    model_frame = active_model_frame if active_model_frame is not None else source_content
    appended_cursor = bool(
        cursor and attempted_content and attempted_content.endswith(cursor)
        and model_frame.startswith(attempted_content[:-len(cursor)])
        and not model_frame.startswith(attempted_content))
    cursor_carriage = prior_raw.get("cursor_carriage")
    if appended_cursor:
        cursor_carriage = next(
            (dict(item) for item in reversed(carriages)
             if item["text"] == attempted_content), cursor_carriage)
    if cursor_carriage not in carriages:
        cursor_carriage = None
    # The carriage stays exactly as rendered; only the reconciliation prefix
    # excludes a cursor known to have been appended by this consumer.
    seen = "".join(
        item["text"][:-len(effective_cursor)]
        if item == cursor_carriage and effective_cursor
        and item["text"].endswith(effective_cursor) else item["text"]
        for item in carriages)
    refused_replacement = bool(
        edited_message_id and not receipt.success and not raw.get("partial_overflow")
        and not raw.get("delivery_ambiguous") and seen
        and source_content.startswith(seen) and attempted_content)
    remainder = raw.get("remaining_content")
    if not isinstance(remainder, str):
        if receipt.success:
            remainder = ""
        elif edited_message_id and seen and source_content.startswith(seen):
            # A refused in-place edit leaves the acknowledged carriage visible;
            # only the bytes beyond it are owed, never the whole attempted frame.
            remainder = source_content[len(seen):]
        else:
            remainder = (prior_raw.get("remaining_content") or attempted_content or
                         (source_content[len(seen):] if seen and source_content.startswith(seen)
                          else source_content))
    if refused_replacement and remainder == attempted_content:
        remainder = source_content[len(seen):]
    source = source_content
    model = raw.get("model_remaining_content")
    attachment = raw.get("attachment_remaining_content")
    if not isinstance(model, str) or not isinstance(attachment, str):
        if (not receipt.success and previous and remainder == prior_raw.get("remaining_content")
                and isinstance(prior_raw.get("model_remaining_content"), str)
                and isinstance(prior_raw.get("attachment_remaining_content"), str)):
            model = prior_raw["model_remaining_content"]
            attachment = prior_raw["attachment_remaining_content"]
        else:
            unsent = source_content[len(seen):] if source_content.startswith(seen) else ""
            if appended_cursor and remainder == attempted_content:
                model, attachment = source_content, ""
            elif unsent and remainder.startswith(unsent):
                model, attachment = unsent, remainder[len(unsent):]
            elif not texts and source_content and remainder.startswith(source_content):
                model, attachment = source_content, remainder[len(source_content):]
            else:
                model, attachment = (remainder, "") if remainder == (
                    attempted_content or source_content) else ("", remainder)
    if refused_replacement and model == attempted_content:
        model = source_content[len(seen):]
    # The attempted frame, not the model suffix, identifies the one cursor
    # appended by this consumer. Never strip a model-authored lookalike.
    if appended_cursor and model == attempted_content:
        model = model[:-len(cursor)]
    if later_chunks:
        model = "\n".join(part for part in (model, *later_chunks) if part)
    owed = model + ("\n" if model and attachment and not attachment.startswith(("\n", " ")) else "") + attachment
    transport_chunks = raw.get("transport_chunks")
    if transport_chunks is None and previous and owed == previous.owed:
        transport_chunks = prior_raw.get("transport_chunks")
    ambiguous = bool(raw.get("ambiguous") or raw.get("delivery_ambiguous") or
                     (not resolve_attempt and prior_raw.get("delivery_ambiguous")))
    content_covered = bool(previous and previous.complete and refused_replacement
                           and not owed and not ambiguous)
    aggregate = replace(
        receipt, success=(receipt.success or content_covered) and not owed and not ambiguous and not declined,
        delivered_text="\n".join(texts) if texts else (
            "" if receipt.success and receipt.delivered_text == "" else None),
        raw_response={
            **{key: value for key, value in raw.items()
               if key not in ("ambient_carriages", "transport_chunks", "cursor_carriage")},
            **({"code": EGRESS_DECLINE_CODE, "success": False, "ambiguous": False}
               if declined else {}),
            **({"transport_chunks": transport_chunks}
               if isinstance(transport_chunks, (list, tuple))
               and all(isinstance(chunk, str) for chunk in transport_chunks)
               and "\n".join(transport_chunks) == owed else {}),
            "stream_transport_remainder": remainder,
            "stream_initial_remaining_content": owed,
            **({"cursor_carriage": cursor_carriage} if cursor_carriage else {}),
            "stream_cursor": effective_cursor,
            "model_remaining_content": model,
            "attachment_remaining_content": attachment,
            "remaining_content": owed, "message_ids": ids,
            "acknowledged_carriages": carriages, "acknowledged_texts": texts,
            "partial_overflow": bool((owed or not (receipt.success or content_covered)) and carriages),
            "delivery_ambiguous": ambiguous,
            "stream_handoff": True,
            "pending_chunks": bool(receipt.success and later_chunks),
            "delivery_error": (None if receipt.success else
                               (raw.get("delivery_error") or receipt.error or
                                (None if resolve_attempt else prior_raw.get("delivery_error")))),
        },
    )
    return StreamDeliveryHandoff(source, aggregate, previous.settled if previous else False,
                                 dict(previous.address) if previous and previous.address else None)


def reconcile_stream_delivery_handoff(
    handoff: StreamDeliveryHandoff, final_content: str,
) -> StreamDeliveryHandoff:
    """Append only genuinely new model bytes, never an already-carried attachment."""
    source = handoff.source_content
    raw = handoff.result.raw_response
    model = raw.get("model_remaining_content", "")
    attachment = raw.get("attachment_remaining_content", "")
    final = final_content
    if final.startswith(source):
        suffix = final[len(source):]
        model += suffix
        if attachment and suffix.endswith(attachment):
            attachment = ""
    elif final != source:
        acknowledged = "".join(raw.get("acknowledged_texts", ()))
        remaining_after_ack = (
            final[len(acknowledged):] if acknowledged and final.startswith(acknowledged) else None
        )
        if remaining_after_ack is not None and remaining_after_ack.startswith(model):
            # A segment can report only its own ledger, while earlier segments
            # already acknowledged the prefix of the final turn.
            model = remaining_after_ack
        else:
            model = final
        if attachment and attachment in final:
            attachment = ""
    owed = model + ("\n" if model and attachment and not attachment.startswith(("\n", " ")) else "") + attachment
    return replace(handoff, source_content=final, result=replace(
        handoff.result, success=not owed and not handoff.ambiguous and handoff.result.success,
        raw_response={**raw, "model_remaining_content": model,
                      "attachment_remaining_content": attachment,
                      "remaining_content": owed,
                      "partial_overflow": bool(owed and
                                               (raw.get("message_ids") or handoff.result.delivered_text))},
    ))
