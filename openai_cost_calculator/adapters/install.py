from __future__ import annotations

import json
import shutil
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
    path = Path.home() / ".codex" / "config.toml"
    text = _read_text(path)
    previous_notify = _managed_previous_notify(text)
    base = _remove_managed_block(text)
    extracted_notify, base = _extract_top_level_key(base, "notify")
    previous_notify = previous_notify or extracted_notify
    block = _codex_block(proxy_url, session, previous_notify)
    new_text = _prepend_managed_block(base, block)
    if new_text != text:
        _backup(path)
        _write_text(path, new_text)
        return [
            f"{path}: installed notify/statusline adapter block",
            "Route Codex through the proxy with base_url http://127.0.0.1:8100/v1 and X-OCC-Session if your provider supports static headers.",
        ]
    return [f"{path}: already installed"]


def uninstall_codex() -> list[str]:
    path = Path.home() / ".codex" / "config.toml"
    text = _read_text(path)
    previous_notify = _managed_previous_notify(text)
    new_text = _remove_managed_block(text)
    if previous_notify and _extract_top_level_key(new_text, "notify")[0] is None:
        new_text = f"{previous_notify}\n{new_text.lstrip()}"
    if new_text != text:
        _backup(path)
        _write_text(path, new_text)
        return [f"{path}: removed openai-cost-calculator block"]
    return [f"{path}: not installed"]


def _claude_settings_path(scope: str) -> Path:
    if scope == "project":
        return Path.cwd() / ".claude" / "settings.json"
    return Path.home() / ".claude" / "settings.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _codex_block(
    proxy_url: str,
    session: str,
    previous_notify: Optional[str] = None,
) -> str:
    escaped_proxy = proxy_url.replace("\\", "\\\\").replace('"', '\\"')
    escaped_session = session.replace("\\", "\\\\").replace('"', '\\"')
    previous = f"# previous_notify = {previous_notify}\n" if previous_notify else ""
    return (
        f"{CODEX_BEGIN}\n"
        '# Codex notify is documented as an external program. Current Codex\n'
        '# docs expose TUI status_line as built-in item identifiers, so the\n'
        '# statusline command is stored here for wrappers that support it.\n'
        f"{previous}"
        'notify = ["occ-codex-notify"]\n'
        f'occ_codex_proxy_url = "{escaped_proxy}"\n'
        f'occ_codex_session = "{escaped_session}"\n'
        'occ_codex_statusline_command = "occ-codex-statusline"\n'
        f"{CODEX_END}\n"
    )


def _prepend_managed_block(text: str, block: str) -> str:
    stripped = _remove_managed_block(text).rstrip()
    if stripped:
        return f"{block}\n{stripped}\n"
    return block


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


def _managed_previous_notify(text: str) -> Optional[str]:
    in_block = False
    for line in text.splitlines():
        if line.strip() == CODEX_BEGIN:
            in_block = True
            continue
        if line.strip() == CODEX_END:
            return None
        if in_block and line.startswith("# previous_notify = "):
            value = line.removeprefix("# previous_notify = ").strip()
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
