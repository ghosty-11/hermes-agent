"""Server-created turn origin and response policy for companion surfaces.

A ``TurnOrigin`` records WHY a turn was admitted and whether a reply is REQUIRED or
discretionary. It is created only from trusted server state — adapter-supplied verified
event metadata or gateway lifecycle flags — and NEVER parsed from message text: public
text can neither set the policy nor the kind (a user typing ``[SILENT]``-adjacent prose or
a fake origin gains nothing).

Existing work surfaces (root, CLI, API, cron) default to ``required``: a silence
marker on them keeps today's fallback reply. Only a configured companion adapter that
explicitly returns a ``discretionary`` origin opts a surface into intentional silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("gateway.run")

# The approved origin vocabulary (plan S02): how the turn was admitted.
TURN_ORIGIN_KINDS = frozenset({"direct", "ambient", "catch_up", "scheduled", "resume"})

RESPONSE_POLICY_REQUIRED = "required"
RESPONSE_POLICY_DISCRETIONARY = "discretionary"
RESPONSE_POLICIES = frozenset({RESPONSE_POLICY_REQUIRED, RESPONSE_POLICY_DISCRETIONARY})

_VALID_POLICIES = {RESPONSE_POLICY_REQUIRED, RESPONSE_POLICY_DISCRETIONARY}


@dataclass(frozen=True)
class TurnOrigin:
    """Trusted per-turn admission metadata (immutable; validated on construction paths)."""

    event_id: str = ""
    kind: str = "direct"
    response_policy: str = RESPONSE_POLICY_REQUIRED
    # Informational surface label (e.g. the adapter that vouched for the origin). Never
    # consulted for authorization decisions.
    surface: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Plain-dict form for the ``pre_llm_call`` payload and result envelopes. An empty
        ``surface`` is omitted so the payload equals the adapter-supplied trusted fields."""
        payload = {
            "event_id": self.event_id,
            "kind": self.kind,
            "response_policy": self.response_policy,
        }
        if self.surface:
            payload["surface"] = self.surface
        return payload


def normalize_turn_origin(candidate: Any) -> Optional[TurnOrigin]:
    """Validate a server-side origin candidate; ``None`` unless every trusted field is
    well-typed. Unknown extra keys are ignored (they are never authority); a bad kind or
    policy fails CLOSED — the caller falls back to the ``required`` default, never to a
    partially trusted origin."""
    if isinstance(candidate, TurnOrigin):
        return candidate
    if not isinstance(candidate, dict):
        return None
    event_id = candidate.get("event_id")
    kind = candidate.get("kind", "direct")
    policy = candidate.get("response_policy", RESPONSE_POLICY_REQUIRED)
    surface = candidate.get("surface", "")
    if not isinstance(event_id, str) or not isinstance(kind, str) or not isinstance(policy, str):
        return None
    if kind not in TURN_ORIGIN_KINDS or policy not in _VALID_POLICIES:
        return None
    if surface is None:
        surface = ""
    if not isinstance(surface, str):
        surface = ""
    return TurnOrigin(event_id=event_id, kind=kind, response_policy=policy, surface=surface)


def turn_origin_allows_silence(origin: Any) -> bool:
    """Whether an intentional silence marker may stand as this turn's outcome.

    ``None``/malformed origins forbid silence (the work-surface default); only an explicit
    ``discretionary`` policy permits it. Machinery display kinds are a separate lane the
    gateway keeps unchanged (``is_machinery_display_kind``)."""
    if not isinstance(origin, dict):
        return False
    return origin.get("response_policy") == RESPONSE_POLICY_DISCRETIONARY


def default_turn_origin(*, scheduled: bool = False, resumed: bool = False,
                        event_id: str = "") -> TurnOrigin:
    """Gateway default when no adapter vouches for an origin: kind from trusted lifecycle
    flags only, policy always ``required`` (work surfaces unchanged)."""
    kind = "scheduled" if scheduled else "resume" if resumed else "direct"
    return TurnOrigin(event_id=event_id, kind=kind, response_policy=RESPONSE_POLICY_REQUIRED)
