"""Worker fixture that deliberately ignores stdin cancellation requests."""

from __future__ import annotations

import json
import os
import sys
import time

sys.stdout.write(
    json.dumps(
        {
            "protocol": 1,
            "type": "hello",
            "payload": {"role": "unresponsive-test-worker", "pid": os.getpid()},
        }
    )
    + "\n"
)
sys.stdout.flush()

while True:
    time.sleep(1)
