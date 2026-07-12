#!/usr/bin/env python3
"""Run a real, isolated Codex request through the local OCC proxy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any


FIXED_RESPONSE = "OCC_SELF_TEST_OK"
PROMPT = f"Reply with exactly `{FIXED_RESPONSE}`. Do not inspect or modify files."
ROOT = Path(__file__).resolve().parents[1]
PRICING_CSV = ROOT / "data" / "gpt_pricing_data.csv"


class SelfTestError(RuntimeError):
    """Raised when the end-to-end evidence violates an acceptance criterion."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument(
        "--model",
        help="Model override; defaults to the model in the source Codex config.",
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--session")
    parser.add_argument(
        "--upstream",
        help="Proxy upstream; defaults from the isolated Codex login mode.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = args.session or f"occ-self-test-{int(time.time())}"
    port = args.port or _free_port()
    temp_owner: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="occ-codex-self-test-"))
    else:
        temp_owner = tempfile.TemporaryDirectory(prefix="occ-codex-self-test-")
        temp_root = Path(temp_owner.name)

    proxy: subprocess.Popen[str] | None = None
    try:
        codex_home = temp_root / "codex-home"
        codex_home.mkdir(mode=0o700)
        auth_method, auth_mode = _prepare_auth(codex_home, Path(args.codex))
        model = args.model or _source_codex_model()
        upstream = args.upstream or _default_upstream(auth_mode)
        env = _isolated_env(codex_home)
        proxy_url = f"http://127.0.0.1:{port}"

        install = _run(
            [
                sys.executable,
                "-m",
                "openai_cost_calculator.cli",
                "install",
                "codex",
                "--proxy-url",
                proxy_url,
                "--session",
                session,
            ],
            env=env,
            cwd=ROOT,
        )
        if install.returncode != 0:
            raise SelfTestError(f"installer failed: {install.stderr.strip()}")
        _validate_install(codex_home / "config.toml", proxy_url, session)

        # Reinstallation must be byte-for-byte idempotent.
        installed_text = (codex_home / "config.toml").read_text(encoding="utf-8")
        reinstall = _run(install.args, env=env, cwd=ROOT)
        if reinstall.returncode != 0:
            raise SelfTestError(f"installer re-run failed: {reinstall.stderr.strip()}")
        if (codex_home / "config.toml").read_text(encoding="utf-8") != installed_text:
            raise SelfTestError("installer re-run changed config.toml")

        proxy_log = (temp_root / "proxy.log").open("w", encoding="utf-8")
        proxy_command = [
            sys.executable,
            "-m",
            "openai_cost_calculator.cli",
            "proxy",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--upstream",
            upstream,
        ]
        proxy = subprocess.Popen(
            proxy_command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
        )
        _wait_ready(proxy_url, proxy)

        baseline = _get_json(f"{proxy_url}/_occ/costs?{_query(session)}")
        if baseline != {"sessions": {}, "grand_total": "0.00000000"}:
            raise SelfTestError(f"unique session did not have a clean baseline: {baseline}")

        output_file = temp_root / "last-message.txt"
        workdir = temp_root / "empty-workdir"
        workdir.mkdir()
        codex_command = [
            str(args.codex),
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--config",
            'notify=[]',
            "--config",
            'model_reasoning_effort="low"',
            "--output-last-message",
            str(output_file),
            "--cd",
            str(workdir),
            PROMPT,
        ]
        if model:
            codex_command[9:9] = ["--model", model]
        nested = _run(codex_command, env=env, cwd=workdir, timeout=180)
        nested_output = output_file.read_text(encoding="utf-8").strip() if output_file.exists() else ""
        if nested.returncode != 0:
            diagnostic = _last_nonempty_line(nested.stderr) or "no stderr diagnostic"
            raise SelfTestError(
                f"nested Codex failed with exit {nested.returncode}: {diagnostic}"
            )
        if nested_output != FIXED_RESPONSE:
            raise SelfTestError(f"unexpected nested response: {nested_output!r}")

        after = _get_json(f"{proxy_url}/_occ/costs?{_query(session)}")
        session_data = after.get("sessions", {}).get(session)
        if not isinstance(session_data, dict):
            raise SelfTestError(f"proxy recorded no data for session {session!r}: {after}")
        turns = session_data.get("turns")
        if not isinstance(turns, list) or len(turns) != 1:
            raise SelfTestError(f"expected one aggregated turn, got: {turns!r}")
        turn = turns[0]
        if turn.get("num_calls") != 1:
            raise SelfTestError(f"expected exactly one upstream request, got: {turn!r}")

        model_costs = turn.get("cost_by_model")
        if not isinstance(model_costs, dict) or len(model_costs) != 1:
            raise SelfTestError(f"expected exactly one observed model, got: {model_costs!r}")
        observed_model = next(iter(model_costs))
        prompt_tokens = _positive_int(turn, "prompt_tokens")
        completion_tokens = _positive_int(turn, "completion_tokens")
        cached_tokens = _nonnegative_int(turn, "cached_tokens")
        if cached_tokens > prompt_tokens:
            raise SelfTestError("cached input tokens exceed total input tokens")

        expected = _independent_cost(
            observed_model,
            prompt_tokens,
            cached_tokens,
            completion_tokens,
        )
        recorded = Decimal(str(turn["total_cost"]))
        if expected["total"].quantize(Decimal("0.00000001")) != recorded:
            raise SelfTestError(f"independent cost {expected['total']} != recorded {recorded}")
        if Decimal(str(session_data["session_total"])) != recorded:
            raise SelfTestError("session total does not equal the only request/turn cost")
        if Decimal(str(after["grand_total"])) != recorded:
            raise SelfTestError("filtered grand total does not equal session total")

        statusline = _run(
            [str(Path(sys.executable).parent / "occ-codex-statusline")],
            env=env,
            cwd=workdir,
        )
        if statusline.returncode != 0 or "cost offline" in statusline.stdout:
            raise SelfTestError(f"Codex status line failed: {statusline.stderr.strip()}")
        if "session" not in statusline.stdout or "$0.0000" in statusline.stdout:
            raise SelfTestError(f"Codex status line did not expose captured cost: {statusline.stdout!r}")

        first_checkpoint = _post_json(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        second_checkpoint = _post_json(f"{proxy_url}/_occ/checkpoint?{_query(session)}")
        after_repeat = _get_json(f"{proxy_url}/_occ/costs?{_query(session)}")
        if first_checkpoint.get("num_calls") != 1:
            raise SelfTestError(f"checkpoint did not contain the new request: {first_checkpoint}")
        if Decimal(str(first_checkpoint.get("total_cost"))) != recorded:
            raise SelfTestError("checkpoint cost does not equal request cost")
        if second_checkpoint.get("num_calls") != 0 or second_checkpoint.get("total_cost") != "0.00000000":
            raise SelfTestError(f"repeated checkpoint double counted: {second_checkpoint}")
        if after_repeat != after:
            raise SelfTestError("status/checkpoint reads mutated cumulative costs")

        result = {
            "result": "PASS",
            "session": session,
            "proxy_port": port,
            "proxy_command": _sanitized_command(proxy_command, temp_root),
            "codex_command": _sanitized_command(codex_command, temp_root),
            "codex_exit_status": nested.returncode,
            "codex_response": nested_output,
            "codex_statusline": statusline.stdout.strip(),
            "auth_method": auth_method,
            "auth_mode": auth_mode,
            "isolated_codex_home": True,
            "model": observed_model,
            "input_tokens": prompt_tokens,
            "cached_input_tokens": cached_tokens,
            "uncached_input_tokens": prompt_tokens - cached_tokens,
            "output_tokens": completion_tokens,
            "input_price_per_million": str(expected["input_rate"]),
            "cached_input_price_per_million": str(expected["cached_rate"]),
            "output_price_per_million": str(expected["output_rate"]),
            "pricing_date": expected["pricing_date"],
            "pricing_minimum_tokens": expected["minimum_tokens"],
            "uncached_input_cost": str(expected["uncached_cost"]),
            "cached_input_cost": str(expected["cached_cost"]),
            "output_cost": str(expected["output_cost"]),
            "independent_total_cost": str(expected["total"]),
            "recorded_request_cost": str(recorded),
            "session_total_before": baseline["grand_total"],
            "session_total_after": session_data["session_total"],
            "checkpoint_cost": first_checkpoint["total_cost"],
            "second_checkpoint_cost": second_checkpoint["total_cost"],
            "installer_idempotent": True,
            "values_match": True,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (SelfTestError, subprocess.TimeoutExpired) as exc:
        print(f"SELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait(timeout=5)
        if temp_owner is not None:
            temp_owner.cleanup()
        elif args.keep_temp:
            print(f"Temporary files retained at {temp_root}", file=sys.stderr)


def _prepare_auth(codex_home: Path, codex: Path) -> tuple[str, str]:
    source_auth = Path.home() / ".codex" / "auth.json"
    if source_auth.is_file():
        try:
            auth_mode = json.loads(source_auth.read_text(encoding="utf-8")).get("auth_mode")
        except (OSError, ValueError):
            auth_mode = None
        if auth_mode not in {"chatgpt", "api"}:
            raise SelfTestError("existing Codex auth.json has an unsupported or missing auth_mode")
        destination = codex_home / "auth.json"
        shutil.copy2(source_auth, destination)
        destination.chmod(0o600)
        return "isolated copy of existing Codex login", auth_mode

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SelfTestError(
            "no ~/.codex/auth.json or OPENAI_API_KEY is available for isolated Codex authentication"
        )
    env = _isolated_env(codex_home)
    login = subprocess.run(
        [str(codex), "login", "--with-api-key"],
        input=api_key,
        text=True,
        capture_output=True,
        env=env,
        timeout=30,
        check=False,
    )
    if login.returncode != 0:
        raise SelfTestError("Codex API-key login failed in the isolated home")
    return "OPENAI_API_KEY imported into isolated Codex login", "api"


def _source_codex_model() -> str | None:
    path = Path.home() / ".codex" / "config.toml"
    try:
        model = tomllib.loads(path.read_text(encoding="utf-8")).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return model if isinstance(model, str) and model else None


def _default_upstream(auth_mode: str) -> str:
    if auth_mode == "chatgpt":
        return "https://chatgpt.com/backend-api/codex"
    return "https://api.openai.com/v1"


def _isolated_env(codex_home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CODEX_HOME"] = str(codex_home)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def _validate_install(path: Path, proxy_url: str, session: str) -> None:
    text = path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    if data.get("model_provider") != "openai_cost_calculator":
        raise SelfTestError("installer did not select the managed provider")
    if data.get("notify") != ["occ-codex-notify"]:
        raise SelfTestError("installer did not configure the Codex notifier")
    if data.get("occ_codex_session") != session:
        raise SelfTestError("installer session metadata is incorrect")
    provider = data.get("model_providers", {}).get("openai_cost_calculator", {})
    expected = {
        "base_url": f"{proxy_url}/v1",
        "requires_openai_auth": True,
        "wire_api": "responses",
        "supports_websockets": False,
        "http_headers": {"X-OCC-Session": session},
    }
    for key, value in expected.items():
        if provider.get(key) != value:
            raise SelfTestError(f"installed provider {key!r} is {provider.get(key)!r}, expected {value!r}")
    if "env_key" in provider or "experimental_bearer_token" in provider:
        raise SelfTestError("installed provider contains an unexpected credential reference")


def _independent_cost(model: str, prompt: int, cached: int, output: int) -> dict[str, Any]:
    model_name, requested_date = _split_model_date(model)
    rows = []
    with PRICING_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_date = row["Model Date"]
            if row["Model Name"] == model_name and (not row_date or row_date <= requested_date):
                rows.append(row)
    if not rows:
        raise SelfTestError(f"working-tree pricing has no row for observed model {model!r}")
    newest_date = max(row["Model Date"] for row in rows)
    dated_rows = [row for row in rows if row["Model Date"] == newest_date]
    applicable = [row for row in dated_rows if int(row.get("Minimum Tokens") or 0) <= prompt]
    if not applicable:
        raise SelfTestError(f"pricing for {model!r} has no tier applicable to {prompt} input tokens")
    row = max(applicable, key=lambda item: int(item.get("Minimum Tokens") or 0))
    input_rate = Decimal(row["Input Price"])
    cached_rate = Decimal(row["Cached Input Price"] or row["Input Price"])
    output_rate = Decimal(row["Output Price"])
    million = Decimal(1_000_000)
    uncached_cost = Decimal(prompt - cached) * input_rate / million
    cached_cost = Decimal(cached) * cached_rate / million
    output_cost = Decimal(output) * output_rate / million
    return {
        "input_rate": input_rate,
        "cached_rate": cached_rate,
        "output_rate": output_rate,
        "pricing_date": newest_date or "undated alias",
        "minimum_tokens": int(row.get("Minimum Tokens") or 0),
        "uncached_cost": uncached_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total": uncached_cost + cached_cost + output_cost,
    }


def _split_model_date(model: str) -> tuple[str, str]:
    parts = model.rsplit("-", 3)
    if len(parts) == 4 and all(part.isdigit() for part in parts[-3:]):
        return parts[0], "-".join(parts[-3:])
    return model, time.strftime("%Y-%m-%d", time.gmtime())


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = _nonnegative_int(data, key)
    if value <= 0:
        raise SelfTestError(f"{key} was not positive: {value}")
    return value


def _nonnegative_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SelfTestError(f"{key} was not a non-negative integer: {value!r}")
    return value


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, proxy: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if proxy.poll() is not None:
            raise SelfTestError(f"proxy exited before readiness with status {proxy.returncode}")
        try:
            _get_json(f"{base_url}/_occ/costs")
            return
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(0.05)
    raise SelfTestError("proxy did not become ready within 10 seconds")


def _get_json(url: str) -> dict[str, Any]:
    return _request_json(url, "GET")


def _post_json(url: str) -> dict[str, Any]:
    return _request_json(url, "POST")


def _request_json(url: str, method: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise SelfTestError(f"{method} {url} did not return a JSON object")
    return payload


def _query(session: str) -> str:
    return urllib.parse.urlencode({"session": session})


def _run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _last_nonempty_line(text: str) -> str:
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def _sanitized_command(command: list[str], temp_root: Path) -> str:
    replacements = sorted(
        (
            (str(ROOT), "$REPO"),
            (str(temp_root), "$TMP"),
            (str(Path.home()), "$HOME"),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    rendered = []
    for item in command:
        for source, replacement in replacements:
            item = item.replace(source, replacement)
        rendered.append(json.dumps(item))
    return " ".join(rendered)


if __name__ == "__main__":
    raise SystemExit(main())
