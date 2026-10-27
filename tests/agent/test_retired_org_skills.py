"""Retired org skill-sync mirrors stay invisible to skill discovery and lookup.

The org sync plane was removed upstream; its on-disk ``_org/<org>/`` mirrors are
user data to preserve, never skills to discover or load — including mirrors that
still carry an active-org marker.
"""

from agent.skill_utils import TIER_LOCAL, is_excluded_skill_path, iter_skill_index_files


def _write_skill(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\n---\n", encoding="utf-8")


def _seed_skills_root(tmp_path):
    """One personal skill plus unmarked ``_org`` mirrors for two orgs."""
    personal = tmp_path / "weather" / "SKILL.md"
    _write_skill(personal, "weather")
    mirror_a = tmp_path / "_org" / "org-a" / "team-standup" / "SKILL.md"
    _write_skill(mirror_a, "team-standup")
    mirror_b = tmp_path / "_org" / "org-b" / "deploy-helper" / "SKILL.md"
    _write_skill(mirror_b, "deploy-helper")
    return personal, mirror_a, mirror_b


def test_iter_skill_index_files_ignores_unmarked_org_mirrors(tmp_path):
    """Stale ``_org`` mirrors are not offered to discovery even with no marker."""
    personal, mirror_a, mirror_b = _seed_skills_root(tmp_path)

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [personal]
    assert is_excluded_skill_path(mirror_a) is True
    assert is_excluded_skill_path(mirror_b) is True


def test_iter_skill_index_files_ignores_org_mirror_with_active_marker(tmp_path):
    """An active-org marker does not reactivate a retired ``_org`` mirror."""
    personal, mirror_a, _mirror_b = _seed_skills_root(tmp_path)
    (tmp_path / "_org" / "org-a" / ".active_org").write_text("", encoding="utf-8")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [personal]
    assert is_excluded_skill_path(mirror_a) is True


def test_iter_skill_index_files_keeps_personal_skills_and_support_dirs_hidden(tmp_path):
    """The reserved ``_org`` exclusion must not eat personal skills or the
    support-dir rule: a SKILL.md under a skill's references/ is still data."""
    personal, mirror_a, mirror_b = _seed_skills_root(tmp_path)
    archived = tmp_path / "weather" / "references" / "legacy-pack" / "SKILL.md"
    _write_skill(archived, "legacy-pack")

    found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

    assert found == [personal]
    assert is_excluded_skill_path(archived) is True
    assert mirror_a.exists() and mirror_b.exists()  # mirrors preserved on disk, just not indexed


def test_iter_skill_index_files_ignores_org_descriptions(tmp_path):
    """The reserved-dir rule covers the secondary DESCRIPTION.md index walk too."""
    _personal, _mirror_a, _mirror_b = _seed_skills_root(tmp_path)
    (tmp_path / "_org" / "org-a" / "team-standup" / "DESCRIPTION.md").write_text(
        "---\ndescription: stale\n---\n", encoding="utf-8")

    assert list(iter_skill_index_files(tmp_path, "DESCRIPTION.md")) == []


class TestConsumerExplicitLookup:
    """The consumer lookup refuses retired mirrors even when asked for them by
    name or by exact relative path; the personal skill beside them still loads."""

    def test_mirror_not_found_by_bare_name(self, tmp_path, monkeypatch):
        personal, _mirror_a, _mirror_b = _seed_skills_root(tmp_path)
        from tools import skills_tool

        monkeypatch.setattr(skills_tool, "find_all_skills", lambda **_: [])

        error, skill_dir, skill_md = skills_tool.locate_skill(
            "team-standup", None, [(TIER_LOCAL, tmp_path)])

        assert error is not None and skill_dir is None and skill_md is None
        assert personal.exists()

    def test_mirror_not_found_by_exact_relative_path(self, tmp_path, monkeypatch):
        _personal, _mirror_a, _mirror_b = _seed_skills_root(tmp_path)
        from tools import skills_tool

        monkeypatch.setattr(skills_tool, "find_all_skills", lambda **_: [])

        error, skill_dir, skill_md = skills_tool.locate_skill(
            "_org/org-a/team-standup", None, [(TIER_LOCAL, tmp_path)])

        assert error is not None and skill_dir is None and skill_md is None

    def test_personal_skill_still_resolves_beside_mirrors(self, tmp_path, monkeypatch):
        personal, _mirror_a, _mirror_b = _seed_skills_root(tmp_path)
        from tools import skills_tool

        monkeypatch.setattr(skills_tool, "find_all_skills", lambda **_: [])

        error, skill_dir, skill_md = skills_tool.locate_skill(
            "weather", None, [(TIER_LOCAL, tmp_path)])

        assert error is None
        assert skill_md == personal
        assert skill_dir == personal.parent
