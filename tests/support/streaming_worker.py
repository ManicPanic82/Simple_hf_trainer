"""Worker that splits a stderr line across two writes.

Used to prove that ``WorkerProcess`` reassembles partial stderr lines instead
of showing two unrelated log entries.
"""

from __future__ import annotations

import json
import sys
import time


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def log_line(text: str, job_id: str = "stream-1") -> None:
    sys.stdout.write(
        json.dumps(
            {
                "protocol": 1,
                "type": "log",
                "job_id": job_id,
                "payload": {"message": text, "level": "info", "stream": "stdout"},
            }
        )
        + "\n"
    )
    sys.stdout.flush()


emit({"protocol": 1, "type": "hello", "payload": {"role": "worker"}})

# One logical stderr line, delivered in two reads with a pause in between.
sys.stderr.write("Loading mo")
sys.stderr.flush()
time.sleep(0.2)
sys.stderr.write("del from cache\n")
sys.stderr.flush()

# A stdout protocol log line, as the real interceptor emits it.
log_line("stdout protocol line")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    if message.get("type") == "cancel":
        job_id = message.get("job_id", "unknown")
        emit(
            {
                "protocol": 1,
                "type": "job_complete",
                "job_id": job_id,
                "payload": {"success": True},
            }
        )
        break
