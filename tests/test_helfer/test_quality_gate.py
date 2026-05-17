"""Tests für quality_gate – Strategie-Integritäts-Wächter."""


def test_check_robust_passes_good_strategy():
    from helfer.quality_gate import check_robust

    result = check_robust(
        oos_pf=1.5,
        ho_pf=1.2,
        oos_trades=50,
        ho_trades=30,
        oos_degradation=0.10,
        ho_degradation=0.15,
    )
    assert result.passed is True


def test_check_robust_fails_low_ho_pf():
    from helfer.quality_gate import check_robust

    result = check_robust(
        oos_pf=1.5,
        ho_pf=0.8,
        oos_trades=50,
        ho_trades=30,
        oos_degradation=0.10,
        ho_degradation=0.45,
    )
    assert result.passed is False
    assert any("ho" in f.lower() for f in result.failures)


def test_check_robust_fails_few_trades():
    from helfer.quality_gate import check_robust

    result = check_robust(
        oos_pf=1.5,
        ho_pf=1.2,
        oos_trades=10,
        ho_trades=5,
        oos_degradation=0.10,
        ho_degradation=0.15,
    )
    assert result.passed is False
    assert any("trade" in f.lower() for f in result.failures)


def test_check_robust_fails_high_degradation():
    from helfer.quality_gate import check_robust

    result = check_robust(
        oos_pf=1.5,
        ho_pf=1.1,
        oos_trades=50,
        ho_trades=30,
        oos_degradation=0.35,
        ho_degradation=0.40,
    )
    assert result.passed is False
