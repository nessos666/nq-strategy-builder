# tests/test_pine_template.py
"""Tests für pine_template."""

import pytest
from tv_kombinatorik.pine_template import render_pine, load_template


def test_render_substitutes_all_params():
    template = "tp_mult = {{tp_mult}}\nsl_mult = {{sl_mult}}"
    params = {"tp_mult": 2.5, "sl_mult": 1.5}
    result = render_pine(template, params)
    assert "tp_mult = 2.5000" in result
    assert "sl_mult = 1.5000" in result
    assert "{{" not in result


def test_render_int_param():
    template = "atr_per = {{atr_per}}"
    params = {"atr_per": 14}
    result = render_pine(template, params)
    assert "atr_per = 14" in result
    assert "{{" not in result


def test_render_raises_on_missing_param():
    template = "tp_mult = {{tp_mult}}\nsl_mult = {{sl_mult}}"
    params = {"tp_mult": 2.5}  # sl_mult fehlt
    with pytest.raises(KeyError, match="sl_mult"):
        render_pine(template, params)


def test_render_multiple_occurrences():
    template = "x = {{val}}\ny = {{val}}"
    params = {"val": 3.0}
    result = render_pine(template, params)
    assert result.count("3.0000") == 2


def test_load_template_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_template("nonexistent_strategy_xyz")


def test_load_template_s1_8_reborn():
    template = load_template("s1_8_reborn")
    assert "//@version=6" in template
    assert "{{tp_mult}}" in template
    assert "{{sl_mult}}" in template
    assert "{{atr_per}}" in template
