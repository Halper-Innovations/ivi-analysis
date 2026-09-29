"""Python-side push alerts.

Always append to data/outputs/cron/alerts.log; push via ntfy
when VOE_ALERT_NTFY_TOPIC is set, Pushover when its pair is set; never
raise. For code paths (like the held-book pass) where a signal must page
the operator without marking a heartbeat step failed.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_config


def post_alert(title: str, body: str) -> None:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        log_path = Path(get_config().outputs_dir) / "cron" / "alerts.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}\t{title}\t{body}\n")
    except Exception:  # noqa: BLE001
        pass

    # VOE_NET_PROVIDER=disabled: the local log above is the whole alert.
    try:
        from app.util.http import network_disabled

        if network_disabled():
            return
    except Exception:  # noqa: BLE001
        pass

    topic = os.getenv("VOE_ALERT_NTFY_TOPIC", "").strip()
    if topic:
        try:
            request = urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=body.encode("utf-8"),
                headers={"Title": title, "Priority": "high"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=10)
        except Exception:  # noqa: BLE001
            pass

    token = os.getenv("VOE_ALERT_PUSHOVER_TOKEN", "").strip()
    user = os.getenv("VOE_ALERT_PUSHOVER_USER", "").strip()
    if token and user:
        try:
            payload = json.dumps(
                {"token": token, "user": user, "title": title, "message": body}
            ).encode("utf-8")
            request = urllib.request.Request(
                "https://api.pushover.net/1/messages.json",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=10)
        except Exception:  # noqa: BLE001
            pass


__all__ = ["post_alert"]
