from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from asgi_client import ASGITestClient

from openai_cost_calculator.adapters.claude_code import (
    statusline_text as claude_statusline_text,
    stop_hook_output,
)
from openai_cost_calculator.adapters.codex import (
    _codex_adapter_settings,
    checkpoint_text,
    notifier_diagnostics,
    notify_main,
    statusline_text as codex_statusline_text,
)
from openai_cost_calculator.adapters.install import (
    install_claude_code,
    install_codex,
    uninstall_claude_code,
    uninstall_codex,
)
from openai_cost_calculator.pricing import (
    add_pricing_entry,
    clear_local_pricing,
    set_offline_mode,
)
from openai_cost_calculator.proxy.app import create_app
from openai_cost_calculator.proxy.registry import TrackerRegistry


@pytest.fixture(autouse=True)
def pinned_pricing():
    clear_local_pricing()
    set_offline_mode(True)
    add_pricing_entry(
        "gpt-test",
        "2025-01-01",
        input_price=1.0,
        output_price=2.0,
        cached_input_price=0.5,
    )
    yield
    clear_local_pricing()
    set_offline_mode(False)


def test_claude_statusline_formats_tokens_and_handles_null_usage():
    payload = {
        "model": {"display_name": "Sonnet 4.6"},
        "cost": {"total_cost_usd": "0.01234"},
        "context_window": {
            "total_input_tokens": 28_000,
            "context_window_size": 200_000,
            "current_usage": {
                "input_tokens": 8_500,
                "output_tokens": 1_200,
                "cache_read_input_tokens": 2_000,
            },
        },
    }

    assert claude_statusline_text(payload) == (
        "💰 $0.0123 session · last 8.5k->1.2k tok "
        "(cache 2.0k) · Sonnet 4.6 · ctx 14%"
    )

    payload["context_window"]["current_usage"] = None
    assert "last -- tok" in claude_statusline_text(payload)
    assert claude_statusline_text({}).startswith("💰 $0.0000 session")


def test_claude_stop_hook_costs_latest_turn_and_dedupes(tmp_path: Path):
    transcript = tmp_path / "transcript.jsonl"
    rows = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "usage": {"input_tokens": 1_000, "output_tokens": 500},
                "cost_usd": "0.0041",
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    payload = {"session_id": "s1", "transcript_path": str(transcript)}

    first = stop_hook_output(payload, cache_dir=tmp_path / "cache")
    assert first == {"systemMessage": "💰 This turn cost $0.0041 (1.0k in / 500 out)"}
    second = stop_hook_output(payload, cache_dir=tmp_path / "cache")
    assert second == {}


def test_codex_adapters_render_and_silently_handle_network_errors(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        url = request.full_url
        if "/_occ/checkpoint" in url:
            return Response(
                {
                    "total_cost": "0.00320000",
                    "prompt_tokens": 1_200,
                    "completion_tokens": 500,
                    "models": {"gpt-5.5": {"total_cost": "0.00320000"}},
                }
            )
        return Response(
            {
                "sessions": {
                    "s1": {
                        "session_total": "0.01000000",
                        "turns": [{"total_cost": "0.00320000"}],
                    }
                },
                "grand_total": "0.01000000",
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert checkpoint_text({}, session="s1") == "💰 Turn $0.0032 · gpt-5.5 1.2k->500"
    assert codex_statusline_text(session="s1") == "💰 $0.0100 session · last $0.0032"

    def failing_urlopen(request, timeout):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    assert checkpoint_text({}, session="s1") is None
    assert codex_statusline_text(session="s1") == "💰 cost offline"


def test_codex_notifier_records_hidden_proxy_failure_without_failing_codex(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(OSError("Bearer secret")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "occ-codex-notify",
            json.dumps({"type": "agent-turn-complete"}),
        ],
    )

    assert notify_main() == 0
    assert capsys.readouterr().out == ""
    diagnostics = notifier_diagnostics()
    assert diagnostics[-1]["code"] == "checkpoint_unavailable"
    assert "secret" not in diagnostics[-1]["message"]
    log = tmp_path / "occ-notifier-diagnostics.jsonl"
    assert log.stat().st_mode & 0o777 == 0o600


def test_codex_notify_uses_installed_session_and_chains_previous_notifier(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    previous_log = tmp_path / "previous.log"
    previous = tmp_path / "previous.py"
    previous.write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(previous_log)!r}).write_text(sys.argv[-1])\n",
        encoding="utf-8",
    )
    (codex_home / "config.toml").write_text(
        "# >>> openai-cost-calculator\n"
        f"# previous_notify = notify = [\"python3\", \"{previous}\"]\n"
        'notify = ["occ-codex-notify"]\n'
        'model_provider = "openai_cost_calculator"\n'
        'occ_codex_proxy_url = "http://127.0.0.1:8100"\n'
        'occ_codex_session = "installed-session"\n'
        'occ_codex_statusline_command = "occ-codex-statusline"\n'
        "# <<< openai-cost-calculator\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    requested_urls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "total_cost": "0.00400000",
                    "prompt_tokens": 2_000,
                    "completion_tokens": 1_000,
                    "models": {"gpt-test": {"total_cost": "0.00400000"}},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return Response()

    notification = {
        "type": "agent-turn-complete",
        "thread-id": "thread-session",
    }
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("sys.argv", ["occ-codex-notify", json.dumps(notification)])

    assert notify_main() == 0
    assert "💰 Turn $0.0040 · gpt-test 2.0k->1.0k" in capsys.readouterr().out
    assert "session=installed-session" in requested_urls[0]
    assert json.loads(previous_log.read_text(encoding="utf-8")) == notification


def test_proxy_checkpoint_advances_cursor_and_costs_remain_cumulative():
    calls = [
        {
            "model": "gpt-test-2025-01-01",
            "usage": {"prompt_tokens": 1_000, "completion_tokens": 0},
        },
        {
            "model": "gpt-test-2025-01-01",
            "usage": {"prompt_tokens": 0, "completion_tokens": 1_000},
        },
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json=calls.pop(0))

    app = create_app(
        upstream="https://upstream.example/v1",
        transport=httpx.MockTransport(handler),
        registry=TrackerRegistry(),
    )
    client = ASGITestClient(app)
    for _ in range(2):
        client.post("/v1/responses", json={}, headers={"x-occ-session": "s1"})

    first = client.post("/_occ/checkpoint?session=s1").json()
    assert first["total_cost"] == "0.00300000"
    assert first["num_calls"] == 2
    assert first["models"]["gpt-test-2025-01-01"]["completion_tokens"] == 1_000
    second = client.post("/_occ/checkpoint?session=s1").json()
    assert second["total_cost"] == "0.00000000"
    assert client.get("/_occ/costs?session=s1").json()["grand_total"] == "0.00300000"


def test_installers_are_idempotent_and_reversible(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    claude_settings = claude_dir / "settings.json"
    claude_settings.write_text('{"unrelated": true}\n', encoding="utf-8")

    install_claude_code()
    install_claude_code()
    settings = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert settings["unrelated"] is True
    assert settings["statusLine"]["command"] == "occ-cc-statusline"
    stop_hooks = settings["hooks"]["Stop"]
    assert json.dumps(stop_hooks).count("occ-cc-stop-hook") == 1
    uninstall_claude_code()
    settings = json.loads(claude_settings.read_text(encoding="utf-8"))
    assert settings == {"unrelated": True}
    assert list(claude_dir.glob("*.occ-backup-*")) == []

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    codex_config = codex_dir / "config.toml"
    codex_config.write_text(
        'notify = ["existing-notifier"]\n'
        'model_provider = "existing-provider"\n'
        'model = "gpt-test"\n',
        encoding="utf-8",
    )
    install_codex("http://127.0.0.1:8100", "s1")
    install_codex("http://127.0.0.1:8100", "s1")
    text = codex_config.read_text(encoding="utf-8")
    assert text.count("openai-cost-calculator") == 4
    assert text.count("occ-codex-notify") == 1
    assert "occ-codex-statusline" in text
    assert 'model_provider = "openai_cost_calculator"' in text
    assert "[model_providers.openai_cost_calculator]" in text
    assert 'base_url = "http://127.0.0.1:8100/v1"' in text
    assert "requires_openai_auth = true" in text
    assert 'env_key = "OPENAI_API_KEY"' not in text
    assert "supports_websockets = false" in text
    assert 'http_headers = { "X-OCC-Session" = "s1" }' in text
    active_notify_lines = [
        line for line in text.splitlines() if line.startswith('notify = ["')
    ]
    assert active_notify_lines == ['notify = ["occ-codex-notify"]']
    assert "# occ-restore-notify = " in text
    assert "# occ-restore-model_provider = " in text
    assert 'model = "gpt-test"' in text
    uninstall_codex()
    assert codex_config.read_text(encoding="utf-8") == (
        'notify = ["existing-notifier"]\n'
        'model_provider = "existing-provider"\n'
        'model = "gpt-test"\n'
    )
    assert list(codex_dir.glob("*.occ-backup-*")) == []


def test_codex_installer_can_enable_websockets_explicitly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))

    install_codex(
        "http://127.0.0.1:8100",
        "s1",
        supports_websockets=True,
    )

    text = (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert "supports_websockets = true" in text


def test_codex_installer_refuses_invalid_config_without_modifying_it(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    original = 'model = "unterminated\n'
    config.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to modify invalid Codex configuration"):
        install_codex("http://127.0.0.1:8100", "s1")

    assert config.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob("config.toml.occ-backup-*")) == []
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


def test_codex_installer_refuses_conflicting_provider_and_malformed_markers(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    conflicting = (
        "[model_providers.openai_cost_calculator]\n"
        'name = "user-owned"\n'
    )
    config.write_text(conflicting, encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite existing Codex provider"):
        install_codex("http://127.0.0.1:8100", "s1")
    assert config.read_text(encoding="utf-8") == conflicting

    malformed = "# >>> openai-cost-calculator\nmodel = \"gpt-test\"\n"
    config.write_text(malformed, encoding="utf-8")
    with pytest.raises(ValueError, match="unmatched start marker"):
        uninstall_codex()
    assert config.read_text(encoding="utf-8") == malformed


def test_codex_installer_refuses_symlink_and_preserves_target(
    tmp_path: Path,
    monkeypatch,
):
    codex_home = tmp_path / "Codex üser"
    codex_home.mkdir()
    target = tmp_path / "actual-config.toml"
    target.write_text('model = "gpt-test"\n', encoding="utf-8")
    (codex_home / "config.toml").symlink_to(target)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(ValueError, match="symlinked configuration"):
        install_codex("http://127.0.0.1:8100", "session ü")

    assert target.read_text(encoding="utf-8") == 'model = "gpt-test"\n'


def test_codex_atomic_write_failure_preserves_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    original = 'model = "gpt-test"\n'
    config.write_text(original, encoding="utf-8")
    config.chmod(0o640)

    def interrupted_replace(source, destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr("os.replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        install_codex("http://127.0.0.1:8100", "s1")

    assert config.read_text(encoding="utf-8") == original
    assert config.stat().st_mode & 0o777 == 0o640
    assert list(tmp_path.glob(".config.toml.*.tmp")) == []


@pytest.mark.parametrize("final_newline", [True, False])
def test_codex_install_uninstall_preserves_unrelated_toml_bytes_exactly(
    tmp_path: Path,
    monkeypatch,
    final_newline: bool,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    original = (
        "# Leading user comment\r\n"
        'notify  = [ "previous-notifier", "--flag" ] # keep inline comment\r\n'
        'model_provider= "custom-provider"   # preserve spacing\r\n'
        'model = "gpt-test"\r\n'
        "\r\n"
        '[profiles."développement"]\r\n'
        'approval_policy = "never"'
    )
    if final_newline:
        original += "\r\n"
    config.write_bytes(original.encode("utf-8"))

    install_codex("http://127.0.0.1:8100", "session ü")
    installed = config.read_bytes()
    install_codex("http://127.0.0.1:8100", "session ü")
    assert config.read_bytes() == installed
    installed_text = installed.decode("utf-8")
    assert "# Leading user comment\r\n" in installed_text
    assert '[profiles."développement"]\r\n' in installed_text
    assert 'approval_policy = "never"' in installed_text

    uninstall_codex()
    assert config.read_bytes() == original.encode("utf-8")


def test_codex_uninstall_restores_legacy_managed_values(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    config.write_text(
        "# >>> openai-cost-calculator\n"
        '# previous_notify = notify = ["legacy"]\n'
        '# previous_model_provider = model_provider = "legacy-provider"\n'
        'notify = ["occ-codex-notify"]\n'
        'model_provider = "openai_cost_calculator"\n'
        "# <<< openai-cost-calculator\n\n"
        "# >>> openai-cost-calculator\n"
        "[model_providers.openai_cost_calculator]\n"
        'name = "managed"\n'
        "# <<< openai-cost-calculator\n\n"
        'model = "gpt-test"\n',
        encoding="utf-8",
    )

    uninstall_codex()
    restored = config.read_text(encoding="utf-8")
    assert 'notify = ["legacy"]' in restored
    assert 'model_provider = "legacy-provider"' in restored
    assert 'model = "gpt-test"' in restored


def test_codex_stashed_notifier_remains_chainable_and_user_edits_survive(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    original = (
        'notify = ["previous-notifier", "--flag"] # retained\n'
        'model_provider = "custom"\n'
        "[features]\n"
        "web_search = true\n"
    )
    config.write_text(original, encoding="utf-8")
    install_codex("http://127.0.0.1:8100", "s1")

    settings = _codex_adapter_settings()
    assert settings["previous_notify"] == (
        'notify = ["previous-notifier", "--flag"] # retained'
    )
    with config.open("a", encoding="utf-8") as handle:
        handle.write("# user edit after installation\n")

    uninstall_codex()
    assert config.read_text(encoding="utf-8") == (
        original + "# user edit after installation\n"
    )
