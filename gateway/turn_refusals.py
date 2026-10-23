"""Turn-local, destination-scoped connector refusals shared by child asyncio tasks."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from gateway.relay.egress import declined_send


class _TurnRefusals:
    __slots__ = ("receipts", "active")

    def __init__(self) -> None:
        self.receipts: dict[tuple[str, str], Any] = {}
        self.active = True


_refusals: ContextVar[_TurnRefusals | None] = ContextVar(
    "gateway_turn_refusals", default=None)


def _key(platform: Any, chat_id: Any) -> tuple[str, str]:
    return str(getattr(platform, "value", platform)), str(chat_id)


@contextmanager
def begin_turn() -> Iterator[None]:
    """Install a fresh map; tasks spawned in this context share its lifetime."""
    state = _TurnRefusals()
    token = _refusals.set(state)
    try:
        yield
    finally:
        state.active = False
        state.receipts.clear()
        _refusals.reset(token)


def refused(platform: Any, chat_id: Any) -> Any | None:
    state = _refusals.get()
    return state.receipts.get(_key(platform, chat_id)) if state is not None and state.active else None


def record(platform: Any, chat_id: Any, receipt: Any) -> None:
    state = _refusals.get()
    if state is not None and state.active and declined_send(receipt):
        state.receipts.setdefault(_key(platform, chat_id), receipt)
