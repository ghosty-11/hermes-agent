"""Strict ``hermes.strict_run.v1`` /v1/runs admission helpers.

The strict input object is a discriminated protocol: the legacy runs handler
only accepts string/list input and rejects a dict before creating an agent,
so older gateways fail closed for avatar clients before any work is
reserved. This module owns:

* exact-shape parsing and validation of the strict input object (before any
  idempotency reservation, concurrency admission, or agent creation),
* reuse of the existing multimodal normalizer plus strict image bounds
  (data URLs only, PNG/JPEG, 3 MiB decoded, 4 MP, magic-byte match) and a
  full decode of the frame,
* lock/conflict resolution against real explicit locks (confirmed session
  model locks and gateway ``/model`` overrides — never a historical
  last-used model), and
* confirmation of the CONSTRUCTED runtime (agent model/provider and the
  client's dialed base URL) against the requested pair, producing the
  descriptor emitted as ``run.runtime`` before any provider output and
  attached to terminal status.

Strict errors are safe codes/messages: they never echo input, data URLs,
credentials, or provider bodies. Legacy string/list input never reaches any
of this code.
"""

import base64
import binascii
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

# Shared constants (avatar-cloud contract). Golden vectors live in
# tests/fixtures/avatar-cloud-protocol.json.
STRICT_RUN_TYPE = "hermes.strict_run.v1"
STRICT_RUN_VERSION = 1
MAX_STRICT_IMAGE_BYTES = 3_145_728
MAX_STRICT_IMAGE_PIXELS = 4_000_000
STRICT_TOOL_POLICIES = frozenset({"inherit", "none"})

# Top-level keys a strict run body may carry. Internal gateway-injected room
# keys are excluded from the unknown-field check (they never come from the
# client and are consumed by _normalize_room_dispatch before this module).
_STRICT_ALLOWED_BODY_KEYS = frozenset({
    "input",
    "session_id",
    "model",
    "provider",
    "require_model_lock",
    "tool_policy",
    "require_image_input",
})
_STRICT_INTERNAL_BODY_KEYS = frozenset({"hosted_room_dispatch", "_room_execution_policy"})

_STRICT_ALLOWED_INPUT_KEYS = frozenset({"type", "content"})
_STRICT_ALLOWED_PART_KEYS = {
    "text": frozenset({"type", "text"}),
    "image_url": frozenset({"type", "image_url"}),
}

# magic-byte signatures for the two admitted formats
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class StrictRunError(ValueError):
    """A strict-schema violation with a safe wire code and message.

    ``code`` is an OpenAI-style error code; ``message`` is fixed text that
    never quotes client input.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def is_strict_run_input(raw_input: Any) -> bool:
    """True when *raw_input* uses the discriminated strict object shape.

    Only a dict is strict; legacy string/list input keeps its documented
    behavior untouched.
    """
    return isinstance(raw_input, dict)


def _reject(code: str, message: str) -> "StrictRunError":
    return StrictRunError(code, message)


def _clean_id(value: Any, *, max_len: int = 200) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or len(text) > max_len:
        return ""
    if re.search(r"[\r\n\x00]", text):
        return ""
    return text


def parse_strict_input(raw_input: Any) -> Dict[str, Any]:
    """Validate the strict input object exactly; return parsed fields.

    Rejects malformed/unknown shapes before any reservation or agent work.
    """
    if not isinstance(raw_input, dict):
        raise _reject("invalid_strict_input", "Strict run input must be an object.")
    if set(raw_input) - _STRICT_ALLOWED_INPUT_KEYS:
        raise _reject("unknown_strict_field", "Strict run input contains unknown fields.")
    if raw_input.get("type") != STRICT_RUN_TYPE:
        raise _reject(
            "invalid_strict_input",
            f"Strict run input type must be {STRICT_RUN_TYPE!r}.",
        )
    content = raw_input.get("content")
    if not isinstance(content, list) or not content:
        raise _reject(
            "invalid_strict_content",
            "Strict run input requires a non-empty content array.",
        )

    text_parts: List[Dict[str, Any]] = []
    image_parts: List[Dict[str, Any]] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise _reject(
                "invalid_content_part",
                "Strict run content parts must be objects.",
            )
        part_type = part.get("type")
        if part_type == "text":
            if set(part) - _STRICT_ALLOWED_PART_KEYS["text"]:
                raise _reject(
                    "unknown_strict_field",
                    "Strict run text part contains unknown fields.",
                )
            text = part.get("text")
            if not isinstance(text, str) or not text.strip():
                raise _reject(
                    "invalid_content_part",
                    "Strict run text parts require non-empty text.",
                )
            text_parts.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            if set(part) - _STRICT_ALLOWED_PART_KEYS["image_url"]:
                raise _reject(
                    "unknown_strict_field",
                    "Strict run image part contains unknown fields.",
                )
            image_ref = part.get("image_url")
            if not isinstance(image_ref, dict):
                raise _reject(
                    "invalid_image_url",
                    "Strict run image parts require an image_url object.",
                )
            if set(image_ref) - {"url"}:
                raise _reject(
                    "unknown_strict_field",
                    "Strict run image reference contains unknown fields.",
                )
            url_value = image_ref.get("url")
            if not isinstance(url_value, str) or not url_value.strip():
                raise _reject(
                    "invalid_image_url",
                    "Strict run image parts require a non-empty image URL.",
                )
            image_parts.append({"type": "image_url", "image_url": {"url": url_value.strip()}})
            continue
        raise _reject(
            "unsupported_content_type",
            "Strict run content supports only text and image_url parts.",
        )

    if not text_parts:
        raise _reject(
            "invalid_strict_content",
            "Strict run input requires at least one non-empty text part.",
        )
    if len(image_parts) > 1:
        raise _reject(
            "too_many_images",
            "Strict run input accepts at most one image part.",
        )
    return {
        "text_parts": text_parts,
        "image_url": image_parts[0]["image_url"]["url"] if image_parts else None,
    }


def parse_strict_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the strict request body around the input object.

    Requires an explicit model/provider pair, an exactly-boolean
    ``require_model_lock: true``, known tool policies, and correctly typed
    image flags. Unknown top-level fields are rejected so an old gateway's
    silently-ignored safety flags cannot masquerade as support.
    """
    unknown = {
        key
        for key in body
        if key not in _STRICT_ALLOWED_BODY_KEYS
        and key not in _STRICT_INTERNAL_BODY_KEYS
    }
    if unknown:
        raise _reject("unknown_strict_field", "Strict run request contains unknown fields.")

    model = _clean_id(body.get("model"))
    provider = _clean_id(body.get("provider"), max_len=80)
    if not model or not provider:
        raise _reject(
            "missing_model",
            "Strict runs require an explicit model and provider pair.",
        )

    if body.get("require_model_lock") is not True:
        raise _reject(
            "model_lock_required",
            "Strict runs require require_model_lock to be exactly true.",
        )
    tool_policy = body.get("tool_policy")
    if not isinstance(tool_policy, str) or tool_policy not in STRICT_TOOL_POLICIES:
        raise _reject(
            "invalid_tool_policy",
            "Strict runs require tool_policy of 'inherit' or 'none'.",
        )
    require_image_input = body.get("require_image_input")
    if not isinstance(require_image_input, bool):
        raise _reject(
            "invalid_image_flag",
            "Strict runs require require_image_input to be a boolean.",
        )

    parsed_input = parse_strict_input(body.get("input"))
    has_image = parsed_input["image_url"] is not None
    if has_image:
        if not require_image_input:
            raise _reject(
                "unexpected_image",
                "Strict runs without require_image_input must not carry an image.",
            )
        if tool_policy != "none":
            raise _reject(
                "image_requires_no_tools",
                "Strict image turns require tool_policy 'none'.",
            )
    elif require_image_input:
        raise _reject(
            "missing_required_image",
            "Strict runs with require_image_input must include exactly one image.",
        )

    session_id = body.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise _reject("invalid_session_id", "Strict run session_id must be a string.")

    return {
        "model": model,
        "provider": provider,
        "tool_policy": tool_policy,
        "require_image_input": require_image_input,
        "has_image": has_image,
        "session_id": session_id,
        **parsed_input,
    }


def validate_strict_image_data_url(url_value: str) -> Dict[str, Any]:
    """Validate one strict screen image and return its normalized metadata.

    Data URLs only; declared PNG/JPEG only; bounded decoded bytes, bounded
    pixels, magic-byte agreement with the declared MIME, and then a full
    decode of the frame: a readable header over corrupt or truncated
    compressed data is not a valid frame. Errors carry safe fixed messages —
    the URL, its bytes, and any payload prefix never reach the response or
    logs.
    """
    if not isinstance(url_value, str) or not url_value.lower().startswith("data:"):
        raise _reject(
            "invalid_image_url",
            "Strict screen images must be inline data URLs.",
        )
    header, sep, encoded = url_value.partition(",")
    if not sep:
        raise _reject("invalid_image_url", "Strict screen image data URLs are malformed.")
    meta = header.partition(":")[2].lower()
    declared_mime = meta.split(";", 1)[0].strip()
    if declared_mime not in {"image/png", "image/jpeg"}:
        raise _reject(
            "unsupported_image_type",
            "Strict screen images must be PNG or JPEG.",
        )
    if not encoded:
        raise _reject("invalid_image", "Strict screen image payload is empty.")
    if len(encoded) > 4 * ((MAX_STRICT_IMAGE_BYTES + 2) // 3):
        # Cheap pre-decode bound: base64 encodes every 3 raw bytes as 4
        # characters, so a valid payload never exceeds 4*ceil(max/3) chars.
        raise _reject("image_too_large", "Strict screen image exceeds the size limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise _reject("invalid_image", "Strict screen image payload is not valid base64.")
    if not raw:
        raise _reject("invalid_image", "Strict screen image payload is empty.")
    if len(raw) > MAX_STRICT_IMAGE_BYTES:
        raise _reject("image_too_large", "Strict screen image exceeds the size limit.")

    sniffed = _sniff_image_mime(raw)
    if sniffed != declared_mime:
        raise _reject(
            "image_format_mismatch",
            "Strict screen image bytes do not match the declared format.",
        )
    dimensions = _image_dimensions(raw, declared_mime)
    if dimensions is None:
        raise _reject("invalid_image", "Strict screen image header is unreadable.")
    width, height = dimensions
    if width <= 0 or height <= 0 or width * height > MAX_STRICT_IMAGE_PIXELS:
        raise _reject(
            "image_too_large",
            "Strict screen image exceeds the pixel limit.",
        )
    if not _decodes_as_valid_frame(raw, declared_mime, (width, height)):
        raise _reject("invalid_image", "Strict screen image data is corrupt or truncated.")
    return {
        "mime_type": declared_mime,
        "width": width,
        "height": height,
        "decoded_bytes": len(raw),
    }


_PIL_FORMAT_FOR_MIME = {"image/png": "PNG", "image/jpeg": "JPEG"}


def _decodes_as_valid_frame(raw: bytes, mime: str, dimensions: Tuple[int, int]) -> bool:
    """Fully decode *raw* with Pillow and confirm it is the declared frame.

    Runs only after the byte/pixel bounds, so the decode is bounded work.
    ``verify()`` checks container integrity (PNG chunk CRCs, truncation);
    the second open + ``load()`` decodes the compressed pixel data itself,
    which is what a truncated JPEG scan or corrupt IDAT stream fails.
    """
    from io import BytesIO

    from PIL import Image

    expected_format = _PIL_FORMAT_FOR_MIME.get(mime)
    try:
        with Image.open(BytesIO(raw)) as probe:
            if probe.format != expected_format or tuple(probe.size) != tuple(dimensions):
                return False
            probe.verify()
        with Image.open(BytesIO(raw)) as frame:
            if frame.format != expected_format or tuple(frame.size) != tuple(dimensions):
                return False
            frame.load()
    except Exception:
        return False
    return True


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(_PNG_MAGIC):
        return "image/png"
    if raw.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    return None


def _image_dimensions(raw: bytes, mime: str) -> Optional[Tuple[int, int]]:
    """Read pixel dimensions from PNG IHDR / JPEG SOFn headers.

    Deterministic header parsing so the pixel bound is enforced BEFORE any
    decode of untrusted pixel data; the full decode follows only for frames
    already inside the byte and pixel limits.
    """
    if mime == "image/png":
        if len(raw) < 24:
            return None
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if mime == "image/jpeg":
        index = 2
        size = len(raw)
        while index + 9 < size:
            if raw[index] != 0xFF:
                index += 1
                continue
            marker = raw[index + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            if index + 4 > size:
                return None
            seg_len = int.from_bytes(raw[index + 2:index + 4], "big")
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if index + 9 > size:
                    return None
                height = int.from_bytes(raw[index + 5:index + 7], "big")
                width = int.from_bytes(raw[index + 7:index + 9], "big")
                return width, height
            if seg_len < 2:
                return None
            index += 2 + seg_len
        return None
    return None


def confirmed_base_url_for_metadata(url_value: Any) -> Optional[str]:
    """Return the normalized endpoint the constructed client dials, or None.

    Only a plain ``http(s)://host[:port]/path`` is confirmable. Unknown,
    unparseable, credential-bearing (userinfo), query-bearing or
    fragment-bearing URLs return None so the caller refuses the run: they
    are never sanitized into an apparently approved endpoint. The only
    normalization is dropping a trailing slash.
    """
    text = str(url_value or "").strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except Exception:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username is not None or parts.password is not None:
        return None
    if parts.query or parts.fragment or "?" in text or "#" in text:
        return None
    host = parts.hostname
    netloc = f"[{host}]" if ":" in host else host
    try:
        port = parts.port
    except ValueError:
        return None
    if port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/") or "", "", ""))


def confirm_strict_runtime(
    *,
    requested_model: str,
    requested_provider: str,
    actual_model: Any,
    actual_provider: Any,
    actual_base_url: Any,
    route_source: str,
    tool_policy: str,
    require_image_input: bool,
    has_image: bool,
) -> Dict[str, Any]:
    """Confirm the CONSTRUCTED runtime against the requested pair.

    ``actual_*`` come from the built agent and its client — never from the
    request. Raises :class:`StrictRunError` (``runtime_mismatch`` /
    ``runtime_unconfirmed``) so the caller refuses before inference; the
    mismatch message carries the exact resolved identity for escalation.
    Returns the frozen wire-shape descriptor on success. ``model_lock`` is
    filled in by the caller once the effective session lock is confirmed.
    """
    model = actual_model if isinstance(actual_model, str) else ""
    provider = actual_provider if isinstance(actual_provider, str) else ""
    if not model or not provider:
        raise _reject(
            "runtime_unconfirmed",
            "Strict runtime could not be confirmed: constructed model/provider unknown.",
        )
    if model != requested_model or provider != requested_provider:
        raise _reject(
            "runtime_mismatch",
            "Strict runtime mismatch: constructed runtime resolved provider "
            f"{provider!r} model {model!r}; refusing before inference.",
        )
    base_url = confirmed_base_url_for_metadata(actual_base_url)
    if base_url is None:
        raise _reject(
            "runtime_unconfirmed",
            "Strict runtime could not be confirmed: client base URL is unknown "
            "or carries credentials/query/fragment.",
        )
    return {
        "strict_run_version": STRICT_RUN_VERSION,
        "requested_provider": requested_provider,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "route_source": route_source or "raw_request",
        "model_lock": False,
        "tool_policy": tool_policy,
        "image_input_mode": "native" if has_image else "none",
        "require_image_input": bool(require_image_input),
    }


def strict_session_lock_conflict(
    *,
    requested_model: str,
    requested_provider: str,
    persisted_lock: Optional[Dict[str, Any]],
    session_model_override: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return a conflict reason when an EXPLICIT lock disagrees with the request.

    Only backend-acknowledged locks count: a confirmed ``browser_model_lock``
    on the session row, or a gateway ``/model`` override for the session key.
    A historical last-used model on the session row is NOT a lock and never
    conflicts.
    """
    def _differs(entry: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(entry, dict):
            return False
        model = _clean_id(entry.get("model"))
        provider = _clean_id(entry.get("provider"), max_len=80)
        if not model and not provider:
            return False
        model_match = (not model) or model == requested_model
        provider_match = (not provider) or provider == requested_provider
        return not (model_match and provider_match)

    if persisted_lock is not None:
        if _differs(persisted_lock):
            return (
                "Session has a confirmed model lock for a different "
                "model/provider; refusing the strict request."
            )
        return None
    if _differs(session_model_override):
        return (
            "Session has an explicit /model override for a different "
            "model/provider; refusing the strict request."
        )
    return None


def parse_persisted_browser_lock(model_config: Any) -> Optional[Dict[str, Any]]:
    """Extract the confirmed ``browser_model_lock`` from a session model_config."""
    if isinstance(model_config, str) and model_config.strip():
        try:
            model_config = json.loads(model_config)
        except Exception:
            return None
    if not isinstance(model_config, dict):
        return None
    lock = model_config.get("browser_model_lock")
    if not isinstance(lock, dict) or lock.get("confirmed") is not True:
        return None
    return lock
