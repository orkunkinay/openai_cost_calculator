from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple


OCC_STATUSLINE = {"type": "command", "command": "occ-cc-statusline"}
OCC_STOP_HOOK = {"type": "command", "command": "occ-cc-stop-hook", "timeout": 5}
CODEX_BEGIN = "# >>> openai-cost-calculator"
CODEX_END = "# <<< openai-cost-calculator"


def install_claude_code(scope: str = "user") -> list[str]:
    path = _claude_settings_path(scope)
    settings = _read_json_object(path)
    changed: list[str] = []

    if settings.get("statusLine") != OCC_STATUSLINE:
        settings["statusLine"] = dict(OCC_STATUSLINE)
        changed.append("statusLine")

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks
        changed.append("hooks")
    stop_entries = hooks.setdefault("Stop", [])
    if not isinstance(stop_entries, list):
        stop_entries = []
        hooks["Stop"] = stop_entries
        changed.append("Stop hook")
    entry = {"matcher": "", "hooks": [dict(OCC_STOP_HOOK)]}
    if not _has_hook(stop_entries, "occ-cc-stop-hook"):
        stop_entries.append(entry)
        changed.append("Stop hook")

    if changed:
        _backup(path)
        _write_json(path, settings)
    return [f"{path}: installed {', '.join(dict.fromkeys(changed))}"] if changed else [f"{path}: already installed"]


def uninstall_claude_code(scope: str = "user") -> list[str]:
    path = _claude_settings_path(scope)
    settings = _read_json_object(path)
    changed: list[str] = []
    if settings.get("statusLine") == OCC_STATUSLINE:
        del settings["statusLine"]
        changed.append("statusLine")

    hooks = settings.get("hooks")
    if isinstance(hooks, dict):
        stop_entries = hooks.get("Stop")
        if isinstance(stop_entries, list):
            filtered = [_remove_hook(entry, "occ-cc-stop-hook") for entry in stop_entries]
            filtered = [entry for entry in filtered if entry is not None]
            if filtered != stop_entries:
                changed.append("Stop hook")
                if filtered:
                    hooks["Stop"] = filtered
                else:
                    hooks.pop("Stop", None)
        if not hooks:
            settings.pop("hooks", None)

    if changed:
        _backup(path)
        _write_json(path, settings)
    return [f"{path}: removed {', '.join(changed)}"] if changed else [f"{path}: not installed"]


def install_codex(proxy_url: str, session: str) -> list[str]:
    path = _codex_config_path()
    text = _read_text(path)
    _validate_toml(text, path)
    previous_notify = _managed_previous_key(text, "notify")
    previous_model_provider = _managed_previous_key(text, "model_provider")
    previous_openai_base_url = _managed_previous_key(text, "openai_base_url")
    base = _remove_managed_block(text)
    if (
        previous_openai_base_url
        and _extract_top_level_key(base, "openai_base_url")[0] is None
    ):
        base = f"{previous_openai_base_url}\n{base.lstrip()}"
    extracted_notify, base = _extract_top_level_key(base, "notify")
    extracted_model_provider, base = _extract_top_level_key(base, "model_provider")
    previous_notify = previous_notify or extracted_notify
    previous_model_provider = previous_model_provider or extracted_model_provider
    top_block, provider_block = _codex_blocks(
        proxy_url,
        session,
        previous_notify,
        previous_model_provider,
    )
    new_text = _insert_codex_blocks(base, top_block, provider_block)
    if new_text != text:
        _backup(path)
        _write_text(path, new_text)
        return [
            f"{path}: installed Codex cost adapter block",
            f"API-key login proxy: openai-cost-calculator proxy --port {_proxy_port(proxy_url)}",
            "ChatGPT login proxy: openai-cost-calculator proxy "
            f"--port {_proxy_port(proxy_url)} "
            "--upstream https://chatgpt.com/backend-api/codex",
            "Ensure Codex is logged in with ChatGPT or an OpenAI API key.",
        ]
    return [f"{path}: already installed"]


def uninstall_codex() -> list[str]:
    path = _codex_config_path()
    text = _read_text(path)
    _validate_toml(text, path)
    previous_notify = _managed_previous_key(text, "notify")
    previous_model_provider = _managed_previous_key(text, "model_provider")
    new_text = _remove_managed_block(text)
    if previous_notify and _extract_top_level_key(new_text, "notify")[0] is None:
        new_text = f"{previous_notify}\n{new_text.lstrip()}"
    if previous_model_provider and _extract_top_level_key(new_text, "model_provider")[0] is None:
        insert = previous_model_provider
        if not insert.endswith("\n"):
            insert = f"{insert}\n"
        new_text = f"{insert}{new_text.lstrip()}"
    if new_text != text:
        _backup(path)
        _write_text(path, new_text)
        return [f"{path}: removed openai-cost-calculator block"]
    return [f"{path}: not installed"]


def _claude_settings_path(scope: str) -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return Path.home() / ".claude" / "settings.json"


def _codex_config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "config.toml"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(previous_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_toml(text: str, path: Path) -> None:
    if not text.strip():
        return
    try:
        try:
            import tomllib  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]
        tomllib.loads(text)
    except Exception as exc:
        raise ValueError(
            f"Refusing to modify invalid Codex configuration at {path}: {exc}"
        ) from exc


def _backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.occ-backup-{stamp}")
    try:
        shutil.copy2(path, backup)
    except Exception:
        return


def _has_hook(entries: list[Any], command: str) -> bool:
    return any(_entry_has_hook(entry, command) for entry in entries)


def _entry_has_hook(entry: Any, command: str) -> bool:
    if not isinstance(entry, dict):
        return False
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return False
    return any(
        isinstance(hook, dict) and hook.get("command") == command
        for hook in hooks
    )


def _remove_hook(entry: Any, command: str) -> Any:
    if not isinstance(entry, dict):
        return entry
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return entry
    filtered = [
        hook
        for hook in hooks
        if not (isinstance(hook, dict) and hook.get("command") == command)
    ]
    if not filtered:
        return None
    next_entry = dict(entry)
    next_entry["hooks"] = filtered
    return next_entry


def _codex_blocks(
    proxy_url: str,
    session: str,
    previous_notify: Optional[str] = None,
    previous_model_provider: Optional[str] = None,
) -> tuple[str, str]:
    escaped_proxy = proxy_url.replace("\\", "\\\\").replace('"', '\\"')
    api_base = _proxy_api_base(proxy_url)
    escaped_api_base = api_base.replace("\\", "\\\\").replace('"', '\\"')
    escaped_session = session.replace("\\", "\\\\").replace('"', '\\"')
    previous = ""
    if previous_notify:
        previous += f"# previous_notify = {previous_notify}\n"
    if previous_model_provider:
        previous += f"# previous_model_provider = {previous_model_provider}\n"
    top_block = (
        f"{CODEX_BEGIN}\n"
        "# OpenAI Cost Calculator routes Codex through the local proxy\n"
        "# and prints a checkpoint after each turn.\n"
        f"{previous}"
        'notify = ["occ-codex-notify"]\n'
        'model_provider = "openai_cost_calculator"\n'
        f'occ_codex_proxy_url = "{escaped_proxy}"\n'
        f'occ_codex_session = "{escaped_session}"\n'
        'occ_codex_statusline_command = "occ-codex-statusline"\n'
        f"{CODEX_END}\n"
    )
    provider_block = (
        f"{CODEX_BEGIN}\n"
        "[model_providers.openai_cost_calculator]\n"
        'name = "OpenAI Cost Calculator Proxy"\n'
        f'base_url = "{escaped_api_base}"\n'
        "requires_openai_auth = true\n"
        'wire_api = "responses"\n'
        "supports_websockets = false\n"
        f'http_headers = {{ "X-OCC-Session" = "{escaped_session}" }}\n'
        f"{CODEX_END}\n"
    )
    return top_block, provider_block


def _prepend_managed_block(text: str, block: str) -> str:
    stripped = _remove_managed_block(text).rstrip()
    if stripped:
        return f"{block}\n{stripped}\n"
    return block


def _insert_codex_blocks(text: str, top_block: str, provider_block: str) -> str:
    base = _remove_managed_block(text).strip()
    if not base:
        return f"{top_block}\n{provider_block}"
    top_level, sections = _split_first_section(base)
    top_level = top_level.strip()
    sections = sections.strip()
    parts = [top_block.rstrip()]
    if top_level:
        parts.append(top_level)
    parts.append(provider_block.rstrip())
    if sections:
        parts.append(sections)
    return "\n\n".join(parts) + "\n"


def _remove_managed_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    in_block = False
    for line in lines:
        if line.strip() == CODEX_BEGIN:
            in_block = True
            continue
        if line.strip() == CODEX_END:
            in_block = False
            continue
        if not in_block:
            output.append(line)
    return "".join(output).rstrip() + ("\n" if output else "")


def _managed_previous_key(text: str, key: str) -> Optional[str]:
    marker = f"# previous_{key} = "
    in_block = False
    for line in text.splitlines():
        if line.strip() == CODEX_BEGIN:
            in_block = True
            continue
        if line.strip() == CODEX_END:
            return None
        if in_block and line.startswith(marker):
            value = line.removeprefix(marker).strip()
            return value or None
    return None


def _extract_top_level_key(text: str, key: str) -> Tuple[Optional[str], str]:
    section: Optional[str] = None
    found: Optional[str] = None
    output: list[str] = []
    prefix = f"{key} "
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        if (
            section is None
            and found is None
            and (stripped.startswith(f"{key}=") or stripped.startswith(prefix))
        ):
            found = stripped
            continue
        output.append(line)
    return found, "".join(output)


def _split_first_section(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return "".join(lines[:index]), "".join(lines[index:])
    return text, ""


def _proxy_api_base(proxy_url: str) -> str:
    base = proxy_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _proxy_port(proxy_url: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(proxy_url)
        return str(parsed.port or 8100)
    except Exception:
        return "8100"
