from __future__ import annotations

import json
from pathlib import Path

import pytest

from openai_cost_calculator.adapters.install import (
    check_claude,
    install_claude,
    uninstall_claude,
)


@pytest.fixture()
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _settings(config_dir: Path) -> dict:
    return json.loads((config_dir / "settings.json").read_text(encoding="utf-8"))


def test_install_is_idempotent_and_preserves_unrelated_settings(config_dir: Path):
    (config_dir / "settings.json").write_text(
        json.dumps({"unrelated": True, "permissions": {"allow": ["Read"]}}), encoding="utf-8"
    )

    install_claude("http://127.0.0.1:8100")
    install_claude("http://127.0.0.1:8100")

    settings = _settings(config_dir)
    assert settings["unrelated"] is True
    assert settings["permissions"] == {"allow": ["Read"]}
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8100"
    assert settings["env"]["OCC_PROXY_URL"] == "http://127.0.0.1:8100"
    assert settings["statusLine"]["command"] == "occ-claude-statusline"
    for event in ("UserPromptSubmit", "Stop", "SessionEnd"):
        assert json.dumps(settings["hooks"][event]).count("occ-claude-hook") == 1
    assert (config_dir / "occ-claude-install.json").exists()


def test_uninstall_restores_absent_state(config_dir: Path):
    install_claude("http://127.0.0.1:8100")
    uninstall_claude()
    assert _settings(config_dir) == {}
    assert not (config_dir / "occ-claude-install.json").exists()


def test_existing_statusline_is_not_destroyed_without_replace(config_dir: Path):
    original = {"statusLine": {"type": "command", "command": "my-status"}}
    (config_dir / "settings.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        install_claude("http://127.0.0.1:8100")
    # File is untouched by the refusal.
    assert _settings(config_dir)["statusLine"]["command"] == "my-status"

    install_claude("http://127.0.0.1:8100", replace_statusline=True)
    assert _settings(config_dir)["statusLine"]["command"] == "occ-claude-statusline"
    uninstall_claude()
    assert _settings(config_dir)["statusLine"]["command"] == "my-status"


def test_existing_gateway_base_url_is_preserved_and_restored(config_dir: Path):
    original = {"env": {"ANTHROPIC_BASE_URL": "https://gw.example.com", "MY_VAR": "keep"}}
    (config_dir / "settings.json").write_text(json.dumps(original), encoding="utf-8")

    messages = install_claude("http://127.0.0.1:8100")
    assert any("gw.example.com" in message for message in messages)
    settings = _settings(config_dir)
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8100"
    assert settings["env"]["MY_VAR"] == "keep"

    uninstall_claude()
    settings = _settings(config_dir)
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example.com"
    assert settings["env"]["MY_VAR"] == "keep"


def test_uninstall_preserves_user_added_hooks(config_dir: Path):
    install_claude("http://127.0.0.1:8100")
    settings = _settings(config_dir)
    settings["hooks"]["Stop"].append({"hooks": [{"type": "command", "command": "user-hook"}]})
    (config_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

    uninstall_claude()
    settings = _settings(config_dir)
    stop = json.dumps(settings.get("hooks", {}).get("Stop", []))
    assert "user-hook" in stop
    assert "occ-claude-hook" not in stop


def test_install_refuses_symlinked_settings(config_dir: Path, tmp_path: Path):
    target = tmp_path / "real-settings.json"
    target.write_text("{}", encoding="utf-8")
    (config_dir / "settings.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlinked configuration"):
        install_claude("http://127.0.0.1:8100")
    assert target.read_text(encoding="utf-8") == "{}"


def test_check_reports_state_and_project_conflict(config_dir: Path, tmp_path: Path, monkeypatch):
    install_claude("http://127.0.0.1:8100")
    project = tmp_path / "proj"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://other"}}), encoding="utf-8"
    )
    monkeypatch.chdir(project)
    report = check_claude()
    assert report["statusline_installed"] is True
    assert set(report["hook_events_installed"]) == {"UserPromptSubmit", "Stop", "SessionEnd"}
    assert report["anthropic_base_url"] == "http://127.0.0.1:8100"
    assert any("project" in conflict for conflict in report["conflicts"])


def test_compose_statusline_wraps_existing_and_restores(config_dir: Path):
    original = {"statusLine": {"type": "command", "command": "my-status --flag"}}
    (config_dir / "settings.json").write_text(json.dumps(original), encoding="utf-8")

    install_claude("http://127.0.0.1:8100", compose_statusline=True)
    command = _settings(config_dir)["statusLine"]["command"]
    assert command.startswith("occ-claude-statusline --compose ")

    from openai_cost_calculator.adapters.claude_proxy import encode_previous_statusline

    assert encode_previous_statusline("my-status --flag") in command

    # Reinstall is idempotent (does not double-wrap).
    install_claude("http://127.0.0.1:8100", compose_statusline=True)
    assert _settings(config_dir)["statusLine"]["command"] == command

    uninstall_claude()
    assert _settings(config_dir)["statusLine"] == {"type": "command", "command": "my-status --flag"}
