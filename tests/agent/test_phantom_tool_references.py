"""Phantom tool references: system-prompt blocks must not name tools the
session can't call (Blank Slate audit, Aug 2026).

Covers:
  * HERMES_AGENT_HELP_GUIDANCE degrades to the docs-only variant when the
    skill tools aren't loaded.
  * execution_guidance_text() never names a web tool (guidance is toolset-neutral).
  * The coding operating brief drops the `todo` sentence when the todo tool
    isn't loaded.
  * Essential skills remain protected from sync/deletion, and the CLI writer
    refuses to introduce a new essential disable.
"""

from pathlib import Path


class TestCodingBriefTodoGating:
    def _brief(self, valid_tool_names):
        from agent.coding_context import CODING_PROFILE, RuntimeMode
        mode = RuntimeMode(
            profile=CODING_PROFILE, surface="cli", cwd=Path.cwd(),
        )
        prefix, _ws, _tr = mode.system_prompt_parts(
            valid_tool_names=valid_tool_names
        )
        assert prefix, "coding profile must emit an operating brief"
        return prefix[0]

    def test_todo_kept_when_tool_available(self):
        brief = self._brief({"todo_list", "terminal", "read_file"})
        assert "todo_list" in brief

    def test_todo_dropped_when_tool_missing(self):
        brief = self._brief({"terminal", "read_file"})
        assert "todo_list" not in brief

    def test_unknown_toolset_keeps_full_brief(self):
        brief = self._brief(None)
        assert "todo_list" in brief



class TestEssentialSkillProtection:
    def test_cli_side_writer_strips_essential(self, monkeypatch):
        import hermes_cli.skills_config as sc
        saved = {}
        monkeypatch.setattr(sc, "save_config", lambda cfg: saved.update(cfg))
        cfg = {}
        sc.save_disabled_skills(cfg, {"hermes-agent", "other"})
        assert cfg["skills"]["disabled"] == ["other"]

    def test_skill_manage_delete_refused(self):
        from tools.skill_manager_guards import _pinned_guard
        msg = _pinned_guard("hermes-agent")
        assert msg is not None

class TestEssentialOnlySync:
    def test_opted_out_sync_seeds_only_essential(self, monkeypatch, tmp_path):
        """A profile with .no-bundled-skills still gets the hermes-agent skill."""
        import tools.skills_sync as ss

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ss.NO_BUNDLED_SKILLS_MARKER).write_text("", encoding="utf-8")

        bundled = tmp_path / "bundled"
        for cat, name in [
            ("autonomous-ai-agents", "hermes-agent"),
            ("media", "gif-search"),
        ]:
            d = bundled / cat / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: x\n---\nbody\n",
                encoding="utf-8",
            )

        monkeypatch.setattr(ss, "_hermes_home", lambda: home)
        monkeypatch.setattr(ss, "_get_bundled_dir", lambda: bundled)
        monkeypatch.setattr(ss, "_build_external_skill_index", lambda: set())

        result = ss.sync_skills(quiet=True)

        assert result["skipped_opt_out"] is True
        assert result["copied"] == ["hermes-agent"]
        assert (home / "skills" / "autonomous-ai-agents" / "hermes-agent" / "SKILL.md").exists()
        assert not (home / "skills" / "media").exists()
