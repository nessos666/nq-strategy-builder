# tests/test_integration_dry.py
"""Integration-Test: Template laden + rendern ohne echtes TradingView."""

import re
from unittest.mock import patch

import pytest
pytest.importorskip("tv_kombinatorik")

from tv_kombinatorik.mcp_bridge import MCPBridge
from tv_kombinatorik.param_space import get_param_space
from tv_kombinatorik.pine_template import load_template, render_pine
from tv_kombinatorik.runner import KombinatorikRunner
from tv_kombinatorik.tv_client import TVClient


# Default-Parameter für S1.8 Reborn (entspricht v2 Test: PF=0.897)
DEFAULT_PARAMS = {
    "atr_per": 14,
    "disp_mult": 1.5,
    "min_body": 0.5,
    "sw_lookback": 10,
    "ob_max_age": 10,
    "sl_mult": 1.5,
    "tp_mult": 2.0,
}


def test_template_loads_and_renders():
    """Template laden + mit Default-Params rendern – keine Platzhalter im Ergebnis."""
    template = load_template("s1_8_reborn")
    result = render_pine(template, DEFAULT_PARAMS)
    assert "{{" not in result
    assert "//@version=6" in result


def test_rendered_pine_contains_correct_values():
    """Gerenderte Werte stimmen mit Params überein."""
    template = load_template("s1_8_reborn")
    result = render_pine(template, DEFAULT_PARAMS)
    # Template-Format: "atr_per     = {{atr_per}}" → int bleibt "14"
    assert "atr_per     = 14" in result
    # float-Werte werden mit .4f formatiert
    assert "tp_mult     = 2.0000" in result
    assert "sl_mult     = 1.5000" in result


def test_full_pipeline_with_mock_bridge(tmp_path):
    """Vollständiger Pipeline-Test mit Mock-Bridge (kein echtes TradingView)."""
    bridge = MCPBridge.mock(pf=1.35, wr=62.0, trades=110, max_dd=0.4)
    client = TVClient(mcp_tools=bridge)
    runner = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=5,
        min_trades=20,
        top_n=3,
        result_dir=tmp_path / "results",
    )
    # time.sleep patchen – sonst 5 Trials × 3 Sek = 15 Sek Wartezeit
    with patch("tv_kombinatorik.tv_client.time.sleep"):
        results = runner.run(storage=None)
    assert len(results) > 0
    assert results[0]["pf"] == pytest.approx(1.35, rel=0.01)


def test_param_space_covers_all_template_placeholders():
    """Alle Platzhalter im Template haben einen entsprechenden Parameter-Raum."""
    template = load_template("s1_8_reborn")
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", template))
    space = get_param_space("s1_8_reborn")
    assert placeholders == set(space.keys()), (
        f"Mismatch: Template hat {placeholders}, Space hat {set(space.keys())}"
    )


def test_mock_bridge_returns_parseable_dom():
    """Mock-Bridge gibt DOM-Text zurück den TVClient parsen kann."""
    bridge = MCPBridge.mock(pf=1.24, wr=58.3, trades=142, max_dd=0.8)
    dom_result = bridge.ui_evaluate(expression="test")
    text = dom_result.get("result", "")
    result = TVClient._parse_dom_text(text)
    assert result is not None
    assert result.pf == pytest.approx(1.24, rel=0.05)
    assert result.trades == 142
