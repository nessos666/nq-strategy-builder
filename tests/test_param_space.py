# tests/test_param_space.py
"""Tests für param_space."""

import pytest
pytest.importorskip("tv_kombinatorik")
import optuna
from tv_kombinatorik.param_space import get_param_space, suggest_params


def test_s1_8_reborn_space_has_required_keys():
    space = get_param_space("s1_8_reborn")
    required = {
        "atr_per",
        "disp_mult",
        "min_body",
        "sw_lookback",
        "ob_max_age",
        "sl_mult",
        "tp_mult",
    }
    assert required == set(space.keys())


def test_get_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unbekannte Strategie"):
        get_param_space("nonexistent_strategy")


def test_suggest_params_returns_valid_dict():
    space = get_param_space("s1_8_reborn")
    study = optuna.create_study()
    trial = study.ask(fixed_distributions=space)
    params = suggest_params(trial, space)
    assert isinstance(params, dict)
    assert set(params.keys()) == set(space.keys())


def test_suggest_params_tp_mult_in_range():
    space = get_param_space("s1_8_reborn")
    study = optuna.create_study()
    for _ in range(10):
        trial = study.ask(fixed_distributions=space)
        params = suggest_params(trial, space)
        assert 1.5 <= params["tp_mult"] <= 4.0
        assert 0.75 <= params["sl_mult"] <= 2.5
        assert 7 <= params["atr_per"] <= 21


def test_suggest_params_int_types():
    space = get_param_space("s1_8_reborn")
    study = optuna.create_study()
    trial = study.ask(fixed_distributions=space)
    params = suggest_params(trial, space)
    assert isinstance(params["atr_per"], int)
    assert isinstance(params["sw_lookback"], int)
    assert isinstance(params["ob_max_age"], int)


def test_suggest_params_float_types():
    space = get_param_space("s1_8_reborn")
    study = optuna.create_study()
    trial = study.ask(fixed_distributions=space)
    params = suggest_params(trial, space)
    assert isinstance(params["tp_mult"], float)
    assert isinstance(params["sl_mult"], float)
    assert isinstance(params["disp_mult"], float)
