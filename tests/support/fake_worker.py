"""Minimal worker used to exercise the real QProcess + protocol path in tests."""

from __future__ import annotations

import json
import os
import sys


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


emit(
    {
        "protocol": 1,
        "type": "hello",
        "payload": {"role": "worker", "cwd": os.getcwd()},
    }
)
sys.stdout.write("this is not json\n")
sys.stdout.flush()

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
        emit({"protocol": 1, "type": "progress", "job_id": job_id, "payload": {"fraction": 0.5}})
        emit(
            {
                "protocol": 1,
                "type": "job_complete",
                "job_id": job_id,
                "payload": {"success": True},
            }
        )
        break
