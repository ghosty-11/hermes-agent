"""Companion-surface scope for session_search.

On a restricted companion surface (a verified ambient Discord turn on a profile
whose ``optmem.speaker_scoped`` shared/public posture is set) the tool's authority
is server-derived only: the caller's own profile database and the current
conversation's compaction lineage. A model-chosen ``profile=``, an embedded
``@session:<profile>/<id>`` link, a foreign session id, or a bare browse of the
profile's other conversations must be refused BEFORE a protected (foreign)
database is opened. Off-restriction callers (CLI, cron, nonambient work surfaces,
private profiles) keep the stock tool.
"""

import json
import time
from contextvars import ContextVar

import pytest

from hermes_state import SessionDB
from tools import session_search_tool as sst
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed(db):
    """s_root (compression ancestor) <- s_current; s_foreign is another conversation.

    Messages are appended BEFORE the compression end and the continuation link,
    mirroring the authentic rotation order (conversation happens, then the
    session compresses into a child) — never appending into a closed session.
    """
    now = int(time.time())
    db.create_session("s_root", source="cli")
    db.append_message("s_root", role="user", content="wombat kumquat harvest planning")
    db.append_message("s_root", role="assistant", content="wombat kumquat rows mapped")
    db.end_session("s_root", "compression")
    db.create_session("s_current", source="cli", parent_session_id="s_root")
    db.append_message("s_current", role="user", content="wombat lantern fish question")
    db.create_session("s_foreign", source="cli")
    db.append_message("s_foreign", role="user", content="wombat zeppelin ticket refund please")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = 's_root'",
        (now - 5000, "Kumquat Harvest"))
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = 's_current'",
        (now - 1000, "Lantern Fish Migration"))
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, title = ? WHERE id = 's_foreign'",
        (now - 2000, "Zeppelin Ticket Refund"))
    db._conn.commit()


@pytest.fixture
def companion(monkeypatch):
    """Bind the ambient identity attestation + discord session vars for one turn."""
    from gateway import session_context

    identity = ContextVar("fixture_ambient_turn_identity", default=None)
    monkeypatch.setattr(session_context, "_ambient_turn_identity", identity, raising=False)
    monkeypatch.setattr(sst, "_profile_restricted", lambda: True)
    tokens = []

    def bind(message_id="m-verified", user_id="111", attested=True):
        values = {
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_MESSAGE_ID": message_id,
            "HERMES_SESSION_USER_ID": user_id,
            "HERMES_SESSION_USER_NAME": "riverbend",
        }
        for name, value in values.items():
            tokens.append(session_context._VAR_MAP[name].set(value))
        tokens.append(identity.set((message_id, user_id) if attested else None))

    yield bind
    while tokens:
        token = tokens.pop()
        token.var.reset(token)


@pytest.fixture
def foreign_db_recorder(monkeypatch, tmp_path):
    """A real other-profile database that records any attempt to open it."""
    opened = []
    other = SessionDB(tmp_path / "other-state.db")
    other.create_session("s_other", source="cli")
    other.append_message("s_other", role="user", content="operator private work row")
    other._conn.commit()

    def _record(profile):
        opened.append(profile)
        return other

    monkeypatch.setattr(sst, "_resolve_profile_db", _record)
    return opened


# =========================================================================
# Cross-profile refusal — before any protected database is opened
# =========================================================================

class TestCrossProfileRefusal:
    def test_explicit_profile_refused_before_foreign_db_opened(
            self, db, companion, foreign_db_recorder):
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, current_session_id="s_current", profile="operator", session_id="s_other"))
        assert result["success"] is False
        assert "cross-profile" in result["error"]
        assert opened_is_empty(foreign_db_recorder)

    def test_explicit_profile_refused_for_browse_too(
            self, db, companion, foreign_db_recorder):
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, current_session_id="s_current", profile="operator"))
        assert result["success"] is False
        assert "cross-profile" in result["error"]
        assert opened_is_empty(foreign_db_recorder)

    def test_embedded_profile_link_refused_before_foreign_db_opened(
            self, db, companion, foreign_db_recorder):
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="@session:operator/s_other"))
        assert result["success"] is False
        assert "cross-profile" in result["error"]
        assert opened_is_empty(foreign_db_recorder)

    def test_slash_link_with_explicit_profile_also_refused(
            self, db, companion, foreign_db_recorder):
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, current_session_id="s_current", profile="companion", session_id="operator/s_other"))
        assert result["success"] is False
        assert "cross-profile" in result["error"]
        assert opened_is_empty(foreign_db_recorder)


def opened_is_empty(opened):
    return opened == []


# =========================================================================
# Conversation scope — foreign sessions in the SAME profile
# =========================================================================

class TestConversationScope:
    def test_foreign_session_read_refused(self, db, companion):
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_foreign"))
        assert result["success"] is False
        assert "current conversation" in result["error"]

    def test_foreign_session_scroll_refused(self, db, companion):
        _seed(db)
        companion()
        mid = db.get_messages("s_foreign")[0]["id"]
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_foreign",
            around_message_id=mid))
        assert result["success"] is False
        assert "current conversation" in result["error"]

    def test_scroll_anchor_from_foreign_session_refused(self, db, companion):
        _seed(db)
        companion()
        mid = db.get_messages("s_foreign")[0]["id"]
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_current",
            around_message_id=mid))
        assert result["success"] is False


class TestMissingIdentity:
    def test_unattested_turn_denies_browse_discovery_and_foreign_read(self, db, companion):
        _seed(db)
        companion(attested=False)
        browse = json.loads(session_search(db=db, current_session_id="s_current"))
        assert browse["success"] is False
        assert "not available on this surface" in browse["error"]

        discovery = json.loads(session_search(
            db=db, query="wombat", current_session_id="s_current"))
        assert discovery["success"] is False
        assert "verified speaker identity" in discovery["error"]

        foreign = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_foreign"))
        assert foreign["success"] is False

        profile = json.loads(session_search(
            db=db, current_session_id="s_current", profile="operator", session_id="s_other"))
        assert profile["success"] is False

    def test_unattested_turn_withholds_own_lineage_read_too(self, db, companion):
        """Missing authorization withholds protected reads entirely: a
        server-bound session id bounds the conversation but does not attest
        the speaker."""
        _seed(db)
        companion(attested=False)
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_root"))
        assert result["success"] is False
        assert "verified speaker identity" in result["error"]



class TestConversationPreservedPaths:
    def test_same_lineage_scroll_preserved(self, db, companion):
        _seed(db)
        companion()
        mid = db.get_messages("s_root")[0]["id"]
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="s_root",
            around_message_id=mid, window=2))
        assert result["success"] is True
        assert result["mode"] == "scroll"

    def test_browse_refused_on_companion_surface(self, db, companion):
        _seed(db)
        companion()
        result = json.loads(session_search(db=db, current_session_id="s_current"))
        assert result["success"] is False
        assert "not available on this surface" in result["error"]

    def test_discovery_limited_to_current_lineage(self, db, companion):
        """A query matching both conversations returns only the current lineage."""
        _seed(db)
        companion()
        result = json.loads(session_search(
            db=db, query="wombat", current_session_id="s_current"))
        assert result["success"] is True
        sids = [r["session_id"] for r in result["results"]]
        assert "s_foreign" not in sids
        assert sids, "the compression ancestor must remain discoverable"
        assert set(sids) <= {"s_root", "s_current"}



# =========================================================================
# A missing ambient attestation must not widen a restricted surface
# =========================================================================

class TestFailClosedSurface:
    def _bind_discord(self, monkeypatch):
        from gateway import session_context

        monkeypatch.delattr(session_context, "_ambient_turn_identity", raising=False)
        monkeypatch.setattr(sst, "_profile_restricted", lambda: True)
        tokens = [session_context._VAR_MAP[name].set(value) for name, value in {
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_MESSAGE_ID": "m-noattest",
            "HERMES_SESSION_USER_ID": "111",
        }.items()]
        return tokens

    def test_missing_attestation_attribute_still_restricts_every_shape(
            self, db, monkeypatch, foreign_db_recorder):
        """Ambent plugin absent entirely: posture + platform alone restrict."""
        from gateway import session_context

        tokens = self._bind_discord(monkeypatch)
        try:
            _seed(db)
            scope = sst._companion_turn_scope()
            assert scope is not None, "restricted surface must not fall back to stock"
            assert scope["verified"] is False

            browse = json.loads(session_search(db=db, current_session_id="s_current"))
            assert browse["success"] is False
            discovery = json.loads(session_search(
                db=db, query="wombat", current_session_id="s_current"))
            assert discovery["success"] is False
            assert "verified speaker identity" in discovery["error"]
            foreign = json.loads(session_search(
                db=db, current_session_id="s_current", session_id="s_foreign"))
            assert foreign["success"] is False
            profile = json.loads(session_search(
                db=db, current_session_id="s_current",
                profile="operator", session_id="s_other"))
            assert profile["success"] is False
            assert foreign_db_recorder == [], "no protected database may open"
            own = json.loads(session_search(
                db=db, current_session_id="s_current", session_id="s_root"))
            assert own["success"] is False
            assert "verified speaker identity" in own["error"]
        finally:
            for token in reversed(tokens):
                token.var.reset(token)

    def test_unreadable_session_context_refuses_conservatively(
            self, db, monkeypatch, foreign_db_recorder):
        """session_context import failure on a restricted profile denies browse."""
        import sys as _sys

        monkeypatch.setattr(sst, "_profile_restricted", lambda: True)
        monkeypatch.setitem(_sys.modules, "gateway", None)
        _seed(db)
        try:
            scope = sst._companion_turn_scope()
            assert scope == {"verified": False, "speaker_id": "", "message_id": ""}
            browse = json.loads(session_search(db=db, current_session_id="s_current"))
            assert browse["success"] is False
            assert foreign_db_recorder == []
        finally:
            monkeypatch.undo()

# =========================================================================
# Off-restriction callers keep the stock tool
# =========================================================================

class TestStockSurfaceUnchanged:
    def test_private_profile_keeps_cross_profile_route(self, db, monkeypatch, foreign_db_recorder):
        """Private work surface: attested turn, profile not speaker-scoped."""
        from gateway import session_context

        identity = ContextVar("fixture_operator_identity", default=None)
        monkeypatch.setattr(session_context, "_ambient_turn_identity", identity, raising=False)
        monkeypatch.setattr(sst, "_profile_restricted", lambda: False)
        tokens = [session_context._VAR_MAP[name].set(value) for name, value in {
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_MESSAGE_ID": "m-operator",
            "HERMES_SESSION_USER_ID": "999",
        }.items()]
        tokens.append(identity.set(("m-operator", "999")))
        try:
            _seed(db)
            result = json.loads(session_search(
                db=db, current_session_id="s_current", profile="companion", session_id="s_other"))
        finally:
            for token in reversed(tokens):
                token.var.reset(token)
        assert result["success"] is True
        assert result["mode"] == "read"
        assert foreign_db_recorder == ["companion"]

    def test_bare_session_link_resolution_still_splits_off_surface(
            self, db, monkeypatch, foreign_db_recorder):
        monkeypatch.setattr(sst, "_profile_restricted", lambda: False)
        _seed(db)
        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="companion/s_other"))
        assert result["success"] is True
        assert foreign_db_recorder == ["companion"]


# =========================================================================
# The schema tells the model the truth about the restricted surface
# =========================================================================

class TestSchemaTruthfulness:
    def test_profile_param_documents_companion_restriction(self):
        desc = SESSION_SEARCH_SCHEMA["parameters"]["properties"]["profile"]["description"]
        assert "companion" in desc.lower()

    def test_description_documents_server_derived_scope(self):
        desc = SESSION_SEARCH_SCHEMA["description"].lower()
        assert "current conversation" in desc
