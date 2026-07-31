"""scripts/check_json_logs.py — Verify that service logs emit valid JSON lines.

Usage (run from backend/ or project root):
    python scripts/check_json_logs.py [--service api|worker] [--lines 50]

Requires docker compose to be running. No extra dependencies beyond stdlib.

Example:
    python scripts/check_json_logs.py --service api --lines 100
    python scripts/check_json_logs.py --service worker --lines 50
"""

import argparse
import json
import subprocess
import sys

SERVICE_NAMES = {
    "api": "video-understanding-api-1",  # adjust if compose service name differs
    "worker": "video-understanding-worker-1",
}

# Fallback: try plain service name as defined in docker-compose.yml
FALLBACK_NAMES = {
    "api": "api",
    "worker": "worker",
}


def fetch_log_lines(service_label: str, n: int) -> list[str]:
    """Run `docker compose logs --tail N <service>` and return lines."""
    for name in (SERVICE_NAMES[service_label], FALLBACK_NAMES[service_label]):
        result = subprocess.run(
            ["docker", "compose", "logs", f"--tail={n}", "--no-log-prefix", name],
            capture_output=True,
            text=True,
            cwd=_find_compose_dir(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.splitlines()
    return []


def _find_compose_dir() -> str:
    """Walk up from this script's location to find docker-compose.yml."""
    import os

    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(d, "docker-compose.yml")):
            return d
        d = os.path.dirname(d)
    return "."


def check_lines(lines: list[str]) -> None:
    ok = bad = skipped = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            skipped += 1
            continue
        try:
            json.loads(line)
            ok += 1
        except json.JSONDecodeError:
            bad += 1
            print(f"  [NOT JSON] {line[:120]}")

    total = ok + bad
    print(f"\nResult: {ok}/{total} lines are valid JSON  ({skipped} blank lines skipped)")
    if bad == 0 and total > 0:
        print("✓ All non-blank log lines parse as JSON.")
    elif total == 0:
        print("⚠ No log lines found — is the service running?")
    else:
        print(f"✗ {bad} lines failed JSON parse — logging not fully structured.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check service log lines for valid JSON.")
    parser.add_argument("--service", choices=["api", "worker"], default="api")
    parser.add_argument("--lines", type=int, default=50)
    args = parser.parse_args()

    print(f"Checking last {args.lines} log lines for service '{args.service}' ...")
    lines = fetch_log_lines(args.service, args.lines)
    if not lines:
        # Fallback: if docker compose logs returns nothing, try reading from local process
        print("  (No docker logs found — service may be running outside docker)")
        sys.exit(1)
    check_lines(lines)


if __name__ == "__main__":
    main()
