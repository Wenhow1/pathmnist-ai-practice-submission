"""Low-frequency filesystem monitor; imports no model or dataset code."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def classify(status, stale_seconds):
    state = status["status"]
    if state in ("budget_complete", "early_stopped", "paused"):
        return state
    if state != "running":
        return "unexpected_status"
    modified = datetime.fromisoformat(status["updated_utc"])
    if not process_alive(status.get("pid")):
        return "process_missing"
    if (datetime.now(UTC) - modified).total_seconds() > stale_seconds:
        return "progress_stale"
    return "running"


def check_once(run, stale_seconds):
    run = Path(run)
    status_file = run / "status.json"
    event = "status_missing" if not status_file.is_file() else classify(
        json.loads(status_file.read_text(encoding="utf-8")), stale_seconds)
    state_file = run / "monitor_state.json"
    previous = json.loads(state_file.read_text(encoding="utf-8"))["event"] if state_file.is_file() else None
    if event != previous:
        item = {"utc": datetime.now(UTC).isoformat(), "event": event}
        with open(run / "monitor_events.jsonl", "a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary = state_file.with_name(state_file.name + ".tmp")
        temporary.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, state_file)
    return event


def main():
    parser = argparse.ArgumentParser(description="No-model low-frequency PathMNIST run monitor")
    parser.add_argument("--run", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--stale-seconds", type=int, default=2400)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.stale_seconds < 1:
        parser.error("Intervals must be positive")
    while True:
        event = check_once(args.run, args.stale_seconds)
        if args.once or event in ("budget_complete", "early_stopped"):
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
