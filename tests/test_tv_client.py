# tests/test_tv_client.py
"""Tests für TVClient DOM-Parser."""

import pytest
pytest.importorskip("tv_kombinatorik")
from unittest.mock import MagicMock
from tv_kombinatorik.tv_client import TVClient


# Echter DOM-Text aus TradingView (deutsch)
SAMPLE_DOM_TEXT = (
    "G&V insgesamt−9.850,00USD−0,10%"
    "Max. Drawdown28.052,50USD0,28%"
    "Trades insgesamt96"
    "Gewinnbringende Trades59,38%57/96"
    "Profitfaktor0,897"
)

SAMPLE_DOM_TEXT_2 = (
    "G&V insgesamt19.440,00USD0,19%"
    "Max. Drawdown34.572,50USD0,35%"
    "Trades insgesamt87"
    "Gewinnbringende Trades51,72%45/87"
    "Profitfaktor0,834"
)


def test_parse_result_pf():
    result = TVClient._parse_dom_text(SAMPLE_DOM_TEXT)
    assert result is not None
    assert result.pf == pytest.approx(0.897, rel=0.01)


def test_parse_result_wr():
    result = TVClient._parse_dom_text(SAMPLE_DOM_TEXT)
    assert result is not None
    assert result.wr == pytest.approx(59.38, rel=0.01)


def test_parse_result_trades():
    result = TVClient._parse_dom_text(SAMPLE_DOM_TEXT)
    assert result is not None
    assert result.trades == 96


def test_parse_second_sample():
    result = TVClient._parse_dom_text(SAMPLE_DOM_TEXT_2)
    assert result is not None
    assert result.pf == pytest.approx(0.834, rel=0.01)
    assert result.trades == 87


def test_parse_not_found_returns_none():
    result = TVClient._parse_dom_text("nicht gefunden")
    assert result is None


def test_parse_empty_returns_none():
    result = TVClient._parse_dom_text("")
    assert result is None


def test_parse_missing_profitfaktor_returns_none():
    result = TVClient._parse_dom_text("Trades insgesamt 50 Gewinnbringende Trades 50%")
    assert result is None


def test_inject_and_compile_success():
    mock = MagicMock()
    mock.pine_set_source.return_value = {"success": True}
    mock.pine_smart_compile.return_value = {"has_errors": False, "errors": []}
    client = TVClient(mcp_tools=mock)
    assert client.inject_and_compile("pine code") is True


def test_inject_and_compile_compile_error():
    mock = MagicMock()
    mock.pine_set_source.return_value = {"success": True}
    mock.pine_smart_compile.return_value = {
        "has_errors": True,
        "errors": ["Syntax error"],
    }
    client = TVClient(mcp_tools=mock)
    assert client.inject_and_compile("bad pine code") is False


def test_read_result_parses_dom():
    mock = MagicMock()
    mock.ui_evaluate.return_value = {"result": SAMPLE_DOM_TEXT}
    client = TVClient(mcp_tools=mock)
    result = client.read_result(wait_secs=0)
    assert result is not None
    assert result.pf == pytest.approx(0.897, rel=0.01)


def test_parse_result_max_dd():
    result = TVClient._parse_dom_text(SAMPLE_DOM_TEXT)
    assert result is not None
    # SAMPLE_DOM_TEXT enthält "Max. Drawdown28.052,50USD0,28%"
    # max_dd soll der Prozentwert sein: 0.28
    assert result.max_dd == pytest.approx(0.28, rel=0.01)
