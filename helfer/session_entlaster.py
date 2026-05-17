"""Session-Entlaster – Kontext-Snapshots bei langen Sessions.

Als Claude Code Hook (UserPromptSubmit). Zählt Nachrichten, warnt bei >60.
"""

import json
from datetime import datetime
from pathlib import Path

COUNTER_FILE = Path("/tmp/helfer_session_counter.json")
SNAPSHOT_FILE = Path("/tmp/helfer_session_snapshot.md")
WARN_THRESHOLD = 60
ALERT_THRESHOLD = 80


def increment_and_check() -> dict:
    if COUNTER_FILE.exists():
        data = json.loads(COUNTER_FILE.read_text())
    else:
        data = {"count": 0, "started": datetime.now().isoformat()}
    data["count"] += 1
    data["last"] = datetime.now().isoformat()
    COUNTER_FILE.write_text(json.dumps(data))
    return data


def main():
    data = increment_and_check()
    count = data["count"]
    if count == WARN_THRESHOLD:
        print(f"⚠️ {count} Nachrichten – Session wird lang. Snapshot gespeichert.")
        SNAPSHOT_FILE.write_text(
            f"# Session Snapshot ({datetime.now().strftime('%H:%M')})\n\n"
            f"Nachrichten: {count}\nGestartet: {data['started']}\n\n"
            f"→ Empfehlung: /sp und neue Session für nächstes Thema\n"
        )
    elif count == ALERT_THRESHOLD:
        print(
            f"🔴 {count} Nachrichten! Context-Overflow-Risiko. Bitte Session splitten."
        )
    elif count > ALERT_THRESHOLD and count % 20 == 0:
        print(f"🔴 {count} Nachrichten! Session dringend splitten.")


if __name__ == "__main__":
    main()
