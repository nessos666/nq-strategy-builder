"""Status Board – Alle Helfer-Status auf einen Blick."""

import json

import typer
from loguru import logger

from helfer.common import HELFER_BASE

app = typer.Typer(help="Status Board: Alle Helfer auf einen Blick")


def collect_status() -> dict:
    status_dir = HELFER_BASE / "status"
    alerts_dir = HELFER_BASE / "alerts"
    statuses = {}
    if status_dir.exists():
        for f in sorted(status_dir.glob("*.json")):
            statuses[f.stem] = json.loads(f.read_text())
    alerts = []
    if alerts_dir.exists():
        for f in sorted(alerts_dir.glob("*.md"), reverse=True)[:5]:
            alerts.append({"file": f.name, "content": f.read_text()[:200]})
    return {"statuses": statuses, "alerts": alerts}


@app.command()
def show():
    """Zeigt Status im Terminal."""
    data = collect_status()
    print("\n=== HELFER STATUS ===\n")
    for name, status in data["statuses"].items():
        state = status.get("state", "?")
        icon = "✅" if state in ("ok", "running", "go", "done") else "⚠️"
        print(f"  {icon} {name}: {state}")
    if data["alerts"]:
        print(f"\n=== ALERTS ({len(data['alerts'])}) ===\n")
        for a in data["alerts"]:
            print(f"  📋 {a['file']}")
    print()


@app.command()
def serve(port: int = 5050):
    """Startet Web-Dashboard (Flask)."""
    try:
        from flask import Flask, jsonify, render_template_string
    except ImportError:
        print("Flask nicht installiert. Nutze 'show' statt 'serve'.")
        raise typer.Exit(1)

    web = Flask(__name__)
    TEMPLATE = """<!DOCTYPE html>
<html><head><title>Helfer Status</title>
<meta http-equiv="refresh" content="30">
<style>body{font-family:monospace;background:#1e1e1e;color:#ddd;padding:20px}
.ok{color:#4ec9b0}.warn{color:#ce9178}.card{background:#2d2d2d;padding:15px;margin:10px 0;border-radius:8px}
h1{color:#569cd6}</style></head>
<body><h1>Helfer Status Board</h1>
{% for name, s in statuses.items() %}
<div class="card">
<strong class="{{ 'ok' if s.state in ['ok','running','go','done'] else 'warn' }}">
{{ name }}</strong>: {{ s.state }}
{% if s.get('issues') %}<br>Issues: {{ s.issues }}{% endif %}
{% if s.get('reason') %}<br>Reason: {{ s.reason }}{% endif %}
<br><small>Updated: {{ s.get('updated', '?') }}</small>
</div>{% endfor %}
{% if alerts %}<h2>Alerts</h2>
{% for a in alerts %}<div class="card warn">{{ a.file }}<br><small>{{ a.content }}</small></div>{% endfor %}
{% endif %}</body></html>"""

    @web.route("/")
    def index():
        return render_template_string(TEMPLATE, **collect_status())

    @web.route("/api/status")
    def api():
        return jsonify(collect_status())

    logger.info(f"Status Board auf http://localhost:{port}")
    web.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    app()
