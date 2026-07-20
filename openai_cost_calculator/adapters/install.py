from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple


OCC_STATUSLINE = {"type": "command", "command": "occ-cc-statusline"}
OCC_STOP_HOOK = {"type": "command", "command": "occ-cc-stop-hook", "timeout": 5}

# Proxy-backed Claude integration (occ-claude-*).
OCC_CLAUDE_STATUSLINE = {"type": "command", "command": "occ-claude-statusline"}
OCC_CLAUDE_HOOK = {"type": "command", "command": "occ-claude-hook", "timeout": 5}
OCC_CLAUDE_HOOK_EVENTS = ("UserPromptSubmit", "Stop", "SessionEnd")
OCC_CLAUDE_MANIFEST = "occ-claude-install.json"
OCC_CLAUDE_VERSION = 1
CODEX_BEGIN = "# >>> openai-cost-calculator"
CODEX_END = "# <<< openai-cost-calculator"
CODEX_RESTORE_PREFIX = "# occ-restore-"
CODEX_TRIM_FINAL_NEWLINE = "# occ-trim-final-newline = true"


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
        _write_json(path, settings)
    return [f"{path}: removed {', '.join(changed)}"] if changed else [f"{path}: not installed"]


def _claude_config_dir() -> Path:
    root = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(root) if root else Path.home() / ".claude"


def _claude_settings_file() -> Path:
    return _claude_config_dir() / "settings.json"


def _claude_manifest_file() -> Path:
    return _claude_config_dir() / OCC_CLAUDE_MANIFEST


def _is_occ_statusline(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("command"), str)
        and "occ-claude-statusline" in value["command"]
    )


def _is_command_statusline(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type", "command") == "command"
        and isinstance(value.get("command"), str)
        and bool(value["command"])
    )


def install_claude(
    proxy_url: str = "http://127.0.0.1:8100",
    *,
    replace_statusline: bool = False,
    compose_statusline: bool = False,
    upstream: Optional[str] = None,
) -> list[str]:
    """Install the proxy-backed Claude integration into settings.json.

    Merges ``env.ANTHROPIC_BASE_URL``/``env.OCC_PROXY_URL``, the OCC status line,
    and the OCC lifecycle hooks without disturbing unrelated settings.  Existing
    values are recorded in a non-sensitive manifest for exact restoration.
    """
    path = _claude_settings_file()
    _refuse_symlink(path)
    settings = _read_json_object(path)
    messages: list[str] = []
    changed: list[str] = []
    base = proxy_url.rstrip("/")

    env = settings.get("env")
    if not isinstance(env, dict):
        env = {}
    previous_base_url = env.get("ANTHROPIC_BASE_URL")
    if isinstance(previous_base_url, str) and previous_base_url and previous_base_url != base:
        # Preserve an existing gateway so the proxy can chain to it.
        messages.append(
            f"Existing ANTHROPIC_BASE_URL preserved for chaining: run the proxy "
            f"with --upstream {previous_base_url}"
        )
    previous_env = {
        "ANTHROPIC_BASE_URL": env.get("ANTHROPIC_BASE_URL"),
        "OCC_PROXY_URL": env.get("OCC_PROXY_URL"),
    }
    if env.get("ANTHROPIC_BASE_URL") != base or env.get("OCC_PROXY_URL") != base:
        env = dict(env)
        env["ANTHROPIC_BASE_URL"] = base
        env["OCC_PROXY_URL"] = base
        settings["env"] = env
        changed.append("env.ANTHROPIC_BASE_URL")

    previous_statusline = settings.get("statusLine")
    statusline_managed = False
    if _is_occ_statusline(previous_statusline):
        statusline_managed = True
    elif previous_statusline is None:
        settings["statusLine"] = dict(OCC_CLAUDE_STATUSLINE)
        statusline_managed = True
        changed.append("statusLine")
    elif compose_statusline and _is_command_statusline(previous_statusline):
        from openai_cost_calculator.adapters.claude_proxy import (
            encode_previous_statusline,
        )

        encoded = encode_previous_statusline(previous_statusline["command"])
        settings["statusLine"] = {
            "type": "command",
            "command": f"occ-claude-statusline --compose {encoded}",
        }
        statusline_managed = True
        changed.append("statusLine (composed with existing)")
    elif replace_statusline:
        settings["statusLine"] = dict(OCC_CLAUDE_STATUSLINE)
        statusline_managed = True
        changed.append("statusLine (replaced existing)")
    else:
        raise ValueError(
            "Refusing to overwrite an existing Claude status line; re-run with "
            "replace_statusline=True to replace it, or compose_statusline=True to "
            "keep it alongside the OCC status line"
        )

    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
    else:
        hooks = dict(hooks)
    hooks_changed = False
    for event in OCC_CLAUDE_HOOK_EVENTS:
        entries = hooks.get(event)
        entries = list(entries) if isinstance(entries, list) else []
        if not _has_hook(entries, "occ-claude-hook"):
            entries.append({"hooks": [dict(OCC_CLAUDE_HOOK)]})
            hooks_changed = True
        hooks[event] = entries
    if hooks_changed:
        settings["hooks"] = hooks
        changed.append("hooks")
    elif "hooks" not in settings:
        settings["hooks"] = hooks

    manifest = {
        "version": OCC_CLAUDE_VERSION,
        "config_path": str(path),
        "proxy_url": base,
        "managed": {
            "env": ["ANTHROPIC_BASE_URL", "OCC_PROXY_URL"],
            "statusLine": statusline_managed,
            "hook_events": list(OCC_CLAUDE_HOOK_EVENTS),
        },
        "previous": {
            "env": previous_env,
            "statusLine": None
            if _is_occ_statusline(previous_statusline)
            else previous_statusline,
        },
        "hash": _managed_hash(base),
    }

    if changed or not _claude_manifest_file().exists():
        _write_json(path, settings)
        _write_json(_claude_manifest_file(), manifest)
        messages.insert(0, f"{path}: installed {', '.join(dict.fromkeys(changed)) or 'manifest'}")
        messages.append(
            "Start the matching proxy: openai-cost-calculator proxy "
            f"--port {_proxy_port(base)} --protocol anthropic-messages"
        )
    else:
        messages.insert(0, f"{path}: already installed")
    return messages


def uninstall_claude() -> list[str]:
    """Remove the proxy-backed Claude integration, restoring previous values."""
    path = _claude_settings_file()
    _refuse_symlink(path)
    settings = _read_json_object(path)
    manifest = _read_json_object(_claude_manifest_file())
    previous = manifest.get("previous") if isinstance(manifest, dict) else {}
    previous = previous if isinstance(previous, dict) else {}
    changed: list[str] = []

    env = settings.get("env")
    if isinstance(env, dict):
        env = dict(env)
        previous_env = previous.get("env") if isinstance(previous.get("env"), dict) else {}
        for key in ("ANTHROPIC_BASE_URL", "OCC_PROXY_URL"):
            restore = previous_env.get(key)
            if restore is not None:
                if env.get(key) != restore:
                    env[key] = restore
                    changed.append(f"env.{key}")
            elif key in env:
                del env[key]
                changed.append(f"env.{key}")
        if env:
            settings["env"] = env
        else:
            settings.pop("env", None)

    if _is_occ_statusline(settings.get("statusLine")):
        restore = previous.get("statusLine")
        if restore is not None:
            settings["statusLine"] = restore
        else:
            settings.pop("statusLine", None)
        changed.append("statusLine")

    hooks = settings.get("hooks")
    if isinstance(hooks, dict):
        hooks = dict(hooks)
        for event in OCC_CLAUDE_HOOK_EVENTS:
            entries = hooks.get(event)
            if not isinstance(entries, list):
                continue
            filtered = [_remove_hook(entry, "occ-claude-hook") for entry in entries]
            filtered = [entry for entry in filtered if entry is not None]
            if filtered != entries:
                changed.append("hooks")
            if filtered:
                hooks[event] = filtered
            else:
                hooks.pop(event, None)
        if hooks:
            settings["hooks"] = hooks
        else:
            settings.pop("hooks", None)

    if changed:
        _write_json(path, settings)
    _claude_manifest_file().unlink(missing_ok=True)
    if changed:
        return [f"{path}: removed {', '.join(dict.fromkeys(changed))}"]
    return [f"{path}: not installed"]


def check_claude() -> dict[str, Any]:
    """Report the effective Claude integration state and likely conflicts."""
    path = _claude_settings_file()
    settings = _read_json_object(path)
    env = settings.get("env") if isinstance(settings.get("env"), dict) else {}
    hooks = settings.get("hooks") if isinstance(settings.get("hooks"), dict) else {}
    installed_events = [
        event
        for event in OCC_CLAUDE_HOOK_EVENTS
        if isinstance(hooks.get(event), list) and _has_hook(hooks[event], "occ-claude-hook")
    ]
    conflicts: list[str] = []
    project_settings = Path.cwd() / ".claude" / "settings.json"
    if project_settings.exists() and project_settings != path:
        project = _read_json_object(project_settings)
        project_env = project.get("env") if isinstance(project.get("env"), dict) else {}
        if project_env.get("ANTHROPIC_BASE_URL"):
            conflicts.append(
                "project .claude/settings.json overrides ANTHROPIC_BASE_URL"
            )
    return {
        "config_path": str(path),
        "settings_exists": path.exists(),
        "anthropic_base_url": env.get("ANTHROPIC_BASE_URL"),
        "occ_proxy_url": env.get("OCC_PROXY_URL"),
        "statusline_installed": _is_occ_statusline(settings.get("statusLine")),
        "hook_events_installed": installed_events,
        "manifest_present": _claude_manifest_file().exists(),
        "conflicts": conflicts,
    }


def _managed_hash(base: str) -> str:
    import hashlib

    material = json.dumps(
        {"proxy_url": base, "statusline": OCC_CLAUDE_STATUSLINE, "hook": OCC_CLAUDE_HOOK},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def install_codex(
    proxy_url: str,
    session: str,
    *,
    supports_websockets: bool = False,
) -> list[str]:
    path = _codex_config_path()
    _refuse_symlink(path)
    text = _read_text(path)
    _validate_toml(text, path)
    _validate_managed_blocks(text, path)
    previous_notify = _managed_previous_key(text, "notify")
    previous_model_provider = _managed_previous_key(text, "model_provider")
    previous_openai_base_url = _managed_previous_key(text, "openai_base_url")
    base = _remove_managed_block(text)
    _refuse_conflicting_provider(base, path)
    if (
        previous_openai_base_url
        and _extract_top_level_key(base, "openai_base_url")[0] is None
    ):
        base = f"{previous_openai_base_url}\n{base.lstrip()}"
    stashed_notify = _stashed_top_level_key(base, "notify")
    stashed_model_provider = _stashed_top_level_key(base, "model_provider")
    extracted_notify = None
    extracted_model_provider = None
    if stashed_notify is None:
        extracted_notify, base = _stash_top_level_key(base, "notify")
    if stashed_model_provider is None:
        extracted_model_provider, base = _stash_top_level_key(base, "model_provider")
    previous_notify = previous_notify or extracted_notify
    previous_model_provider = previous_model_provider or extracted_model_provider
    if stashed_notify is None and extracted_notify is None and previous_notify:
        base = _prepend_stashed_key(base, "notify", previous_notify)
        stashed_notify = previous_notify
    if (
        stashed_model_provider is None
        and extracted_model_provider is None
        and previous_model_provider
    ):
        base = _prepend_stashed_key(
            base,
            "model_provider",
            previous_model_provider,
        )
        stashed_model_provider = previous_model_provider
    _top_level, sections = _split_first_section(base)
    trim_final_newline = bool(
        base and not sections and not base.endswith(("\n", "\r"))
    )
    top_block, provider_block = _codex_blocks(
        proxy_url,
        session,
        None if stashed_notify or extracted_notify else previous_notify,
        None
        if stashed_model_provider or extracted_model_provider
        else previous_model_provider,
        supports_websockets=supports_websockets,
        trim_final_newline=trim_final_newline,
    )
    new_text = _insert_codex_blocks(base, top_block, provider_block)
    if new_text != text:
        _write_text(path, new_text)
        return [
            f"{path}: installed Codex cost adapter block",
            "Start the matching proxy: openai-cost-calculator proxy "
            f"--port {_proxy_port(proxy_url)} --auth-mode auto",
            "Ensure Codex is logged in with ChatGPT or an OpenAI API key.",
        ]
    return [f"{path}: already installed"]


def uninstall_codex() -> list[str]:
    path = _codex_config_path()
    _refuse_symlink(path)
    text = _read_text(path)
    _validate_toml(text, path)
    _validate_managed_blocks(text, path)
    previous_notify = _managed_previous_key(text, "notify")
    previous_model_provider = _managed_previous_key(text, "model_provider")
    new_text = _remove_managed_block(text)
    restored_notify, new_text = _restore_stashed_key(new_text, "notify")
    restored_model_provider, new_text = _restore_stashed_key(
        new_text,
        "model_provider",
    )
    if (
        not restored_notify
        and previous_notify
        and _extract_top_level_key(new_text, "notify")[0] is None
    ):
        new_text = f"{previous_notify}\n{new_text.lstrip()}"
    if (
        not restored_model_provider
        and previous_model_provider
        and _extract_top_level_key(new_text, "model_provider")[0] is None
    ):
        insert = previous_model_provider
        if not insert.endswith("\n"):
            insert = f"{insert}\n"
        new_text = f"{insert}{new_text.lstrip()}"
    if new_text != text:
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
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""


def _write_text(path: Path, text: str) -> None:
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    _refuse_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
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
        _fsync_directory(path.parent)
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


def _validate_managed_blocks(text: str, path: Path) -> None:
    depth = 0
    blocks = 0
    for line in text.splitlines():
        marker = line.strip()
        if marker == CODEX_BEGIN:
            if depth:
                raise ValueError(
                    f"Refusing to modify malformed managed blocks at {path}: nested start marker"
                )
            depth = 1
            blocks += 1
        elif marker == CODEX_END:
            if not depth:
                raise ValueError(
                    f"Refusing to modify malformed managed blocks at {path}: unmatched end marker"
                )
            depth = 0
    if depth:
        raise ValueError(
            f"Refusing to modify malformed managed blocks at {path}: unmatched start marker"
        )
    if blocks not in {0, 2}:
        raise ValueError(
            f"Refusing to modify unexpected managed blocks at {path}: found {blocks}, expected 0 or 2"
        )


def _refuse_conflicting_provider(text: str, path: Path) -> None:
    if "[model_providers.openai_cost_calculator]" in {
        line.strip() for line in text.splitlines()
    }:
        raise ValueError(
            "Refusing to overwrite existing Codex provider "
            f"'openai_cost_calculator' at {path}"
        )


def _refuse_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing to replace symlinked configuration at {path}")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    *,
    supports_websockets: bool = False,
    trim_final_newline: bool = False,
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
    if trim_final_newline:
        previous += f"{CODEX_TRIM_FINAL_NEWLINE}\n"
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
        f"supports_websockets = {'true' if supports_websockets else 'false'}\n"
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
    base = _remove_managed_block(text)
    if not base:
        return f"{top_block}{provider_block}"
    top_level, sections = _split_first_section(base)
    if sections:
        return f"{top_block}{top_level}{provider_block}{sections}"
    separator = "" if base.endswith(("\n", "\r")) else "\n"
    return f"{top_block}{base}{separator}{provider_block}"


def _remove_managed_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    in_block = False
    trim_final_newline = False
    for line in lines:
        if line.strip() == CODEX_BEGIN:
            in_block = True
            continue
        if line.strip() == CODEX_END:
            in_block = False
            continue
        if in_block and line.strip() == CODEX_TRIM_FINAL_NEWLINE:
            trim_final_newline = True
        if not in_block:
            output.append(line)
    result = "".join(output)
    if trim_final_newline and result.endswith("\n"):
        result = result[:-1]
    return result


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


def _stash_top_level_key(text: str, key: str) -> Tuple[Optional[str], str]:
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
            output.append(_stashed_line(key, line))
            continue
        output.append(line)
    return found, "".join(output)


def _prepend_stashed_key(text: str, key: str, assignment: str) -> str:
    raw_line = assignment if assignment.endswith(("\n", "\r")) else f"{assignment}\n"
    return f"{_stashed_line(key, raw_line)}{text}"


def _stashed_line(key: str, raw_line: str) -> str:
    if raw_line.endswith("\r\n"):
        ending = "\r\n"
    elif raw_line.endswith(("\n", "\r")):
        ending = raw_line[-1]
    else:
        ending = ""
    encoded = base64.urlsafe_b64encode(raw_line.encode("utf-8")).decode("ascii")
    return f"{CODEX_RESTORE_PREFIX}{key} = {encoded}{ending}"


def _stashed_top_level_key(text: str, key: str) -> Optional[str]:
    marker = f"{CODEX_RESTORE_PREFIX}{key} = "
    matches = [line for line in text.splitlines() if line.startswith(marker)]
    if len(matches) > 1:
        raise ValueError(f"Refusing duplicate restoration placeholders for {key}")
    if not matches:
        return None
    raw = _decode_stashed_line(matches[0], marker)
    return raw.strip()


def _restore_stashed_key(text: str, key: str) -> Tuple[bool, str]:
    marker = f"{CODEX_RESTORE_PREFIX}{key} = "
    restored = False
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(marker):
            if restored:
                raise ValueError(f"Refusing duplicate restoration placeholders for {key}")
            output.append(_decode_stashed_line(line.rstrip("\r\n"), marker))
            restored = True
        else:
            output.append(line)
    return restored, "".join(output)


def _decode_stashed_line(line: str, marker: str) -> str:
    encoded = line.removeprefix(marker).strip()
    try:
        return base64.b64decode(encoded, altchars=b"-_", validate=True).decode("utf-8")
    except Exception as exc:
        raise ValueError("Refusing malformed Codex restoration placeholder") from exc


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
