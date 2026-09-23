"""``SessionDB.record_message_delivery`` — actual-delivery metadata on the assistant row
(companion delivery contract).

Contracts:

- Records merge a ``delivery`` member into the assistant row's EXISTING
  ``display_metadata`` — reactions and other members survive untouched, and the row's
  content is never rewritten.
- The row must be an assistant row of the addressed session; anything else (user row,
  foreign session, missing row, non-int id) is refused without writing.
- The turn identity binds the record: a SECOND turn may not overwrite the first turn's
  delivery; receipts within the SAME turn aggregate idempotently (message ids union,
  disposition may refine unknown -> text, no duplicate delivery member).
- Dispositions are the fixed vocabulary text/media/silence/failure/unknown.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return SessionDB(db_path=tmp_path / "state.db")


@pytest.fixture
def session(db):
    key = db.create_session("delivery-test", "test")
    db.append_message(key, "user", "what is the answer")
    db.append_message(key, "assistant", "42 [SILENT]")  # raw draft kept as audit data
    rows = [m["_row_id"] for m in db.get_messages_as_conversation(key, include_row_ids=True)]
    return key, rows


def _row_meta(db, key, row_id):
    return next(
        m.get("display_metadata") for m in db.get_messages_as_conversation(key, include_row_ids=True)
        if m["_row_id"] == row_id
    )


def test_records_delivery_on_the_assistant_row(db, session):
    key, rows = session
    ok = db.record_message_delivery(
        key, rows[1], "turn-1",
        {"disposition": "text", "text": "42", "message_ids": ["m-1"], "error": None},
    )
    assert ok is True
    meta = _row_meta(db, key, rows[1])
    delivery = meta["delivery"]
    assert delivery["disposition"] == "text"
    assert delivery["text"] == "42"
    assert delivery["turn_id"] == "turn-1"
    assert delivery["message_ids"] == ["m-1"]
    # The raw draft stays as audit data; the row's own content is untouched.
    assert [m["content"] for m in db.get_messages_as_conversation(key)][1] == "42 [SILENT]"


def test_preserves_reactions_and_other_metadata(db, session):
    key, rows = session
    db.set_message_reaction(key, rows[1], "🔥", author="user")
    db.record_message_delivery(
        key, rows[1], "turn-1", {"disposition": "silence"},
    )
    meta = _row_meta(db, key, rows[1])
    assert [r["emoji"] for r in meta[db.REACTIONS_METADATA_KEY]] == ["🔥"]
    assert meta["delivery"]["disposition"] == "silence"
    # And the reverse order: a reaction added AFTER delivery keeps the delivery member.
    db.set_message_reaction(key, rows[1], "👍", author="agent")
    meta = _row_meta(db, key, rows[1])
    assert meta["delivery"]["disposition"] == "silence"
    assert {r["emoji"] for r in meta[db.REACTIONS_METADATA_KEY]} == {"🔥", "👍"}


def test_same_turn_receipts_aggregate_idempotently(db, session):
    key, rows = session
    db.record_message_delivery(
        key, rows[1], "turn-1", {"disposition": "unknown", "message_ids": ["m-1"]},
    )
    db.record_message_delivery(
        key, rows[1], "turn-1",
        {"disposition": "text", "text": "delivered text", "message_ids": ["m-1", "m-2"]},
    )
    # A repeated identical receipt is a no-op, not a duplicate entry.
    db.record_message_delivery(
        key, rows[1], "turn-1",
        {"disposition": "text", "text": "delivered text", "message_ids": ["m-1", "m-2"]},
    )
    meta = _row_meta(db, key, rows[1])
    assert len(meta["delivery"]["message_ids"]) == len(set(meta["delivery"]["message_ids"])) == 2
    assert meta["delivery"]["disposition"] == "text"
    assert meta["delivery"]["text"] == "delivered text"


def test_refuses_cross_turn_overwrite(db, session):
    key, rows = session
    db.record_message_delivery(key, rows[1], "turn-1", {"disposition": "text", "text": "first"})
    ok = db.record_message_delivery(
        key, rows[1], "turn-2", {"disposition": "text", "text": "second"})
    assert ok is False
    assert _row_meta(db, key, rows[1])["delivery"]["text"] == "first"


def test_refuses_non_assistant_rows(db, session):
    key, rows = session
    assert db.record_message_delivery(key, rows[0], "turn-1", {"disposition": "text"}) is False
    assert _row_meta(db, key, rows[0]) is None or "delivery" not in (_row_meta(db, key, rows[0]) or {})


def test_refuses_missing_row_or_session(db, session):
    key, _rows = session
    assert db.record_message_delivery(key, 999_999, "turn-1", {"disposition": "text"}) is False
    assert db.record_message_delivery("no-such-session", 1, "turn-1", {"disposition": "text"}) is False


def test_refuses_invalid_outcomes(db, session):
    key, rows = session
    assert db.record_message_delivery(key, rows[1], "turn-1", {"disposition": "definitely-sent"}) is False
    assert db.record_message_delivery(key, rows[1], "", {"disposition": "text"}) is False
    assert db.record_message_delivery(key, rows[1], "turn-1", "sent!") is False


def test_failure_outcome_keeps_error_and_silence_keeps_absence(db, session):
    key, rows = session
    db.record_message_delivery(
        key, rows[1], "turn-1",
        {"disposition": "failure", "error": "HTTP 429: flood control", "message_ids": []},
    )
    delivery = _row_meta(db, key, rows[1])["delivery"]
    assert delivery["disposition"] == "failure"
    assert delivery["error"] == "HTTP 429: flood control"

    key2 = db.create_session("delivery-test-2", "test")
    db.append_message(key2, "user", "hi")
    row2 = db.append_message(key2, "assistant", "[SILENT]")
    db.record_message_delivery(key2, row2, "turn-a", {"disposition": "silence"})
    delivery2 = _row_meta(db, key2, row2)["delivery"]
    assert delivery2["disposition"] == "silence"
    assert "error" not in delivery2 or delivery2.get("error") in (None, "")


def test_delivery_survives_reopen_and_compaction_addressing(db, tmp_path, session):
    key, rows = session
    db.record_message_delivery(key, rows[1], "turn-1", {"disposition": "text", "text": "42"})
    reopened = SessionDB(db_path=db.db_path)
    assert _row_meta(reopened, key, rows[1])["delivery"]["text"] == "42"
    reopened.close()
