from __future__ import annotations

import pytest


def _make_fake_registry(n_runs: int = 50) -> list[dict]:
    """Erstellt Fake-Registry mit n_runs Einträgen."""
    import random

    random.seed(42)
    concepts = [
        "FVG",
        "BOS",
        "OB",
        "JUDAS",
        "SWEEP",
        "CHOCH",
        "HURST",
        "GARCH",
        "KILLZONE",
        "REGIME",
        "DISPLACEMENT",
        "OTE",
        "MANIP",
    ]
    sessions = ["NY", "LONDON", "ASIA"]
    rows = []
    for i in range(n_runs):
        n_concepts = random.randint(2, 4)
        chosen = random.sample(concepts, n_concepts)
        session = random.choice(sessions)
        idea = " + ".join(chosen) + f" {session}"
        tier = random.choices(["A", "B", "C"], weights=[5, 35, 60])[0]
        rows.append(
            {
                "id": i + 1,
                "idea": idea,
                "tier": tier,
                "avg_oos_pf": 1.2 + random.random() * 1.5
                if tier in ("A", "B")
                else 0.8 + random.random() * 0.5,
            }
        )
    return rows


def test_meta_learner_trains_without_error():
    """MetaLearner.fit() läuft ohne Fehler durch."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    assert learner.is_fitted


def test_meta_learner_suggest_returns_suggestion_results_list():
    """suggest() gibt SuggestionResult-Objekte zurueck."""
    from sb.engine.meta_learner import MetaLearner, SuggestionResult

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    results = learner.suggest(n=10)
    assert isinstance(results, list)
    assert len(results) == 10
    assert all(isinstance(r, SuggestionResult) for r in results)
    assert all(isinstance(r.idea, str) and len(r.idea) > 0 for r in results)


def test_meta_learner_suggest_no_duplicates_with_existing():
    """Suggestions enthalten keine bereits getesteten Ideen."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    results = learner.suggest(n=10)
    existing_ideas = {r["idea"].lower() for r in registry}
    for s in results:
        assert s.idea.lower() not in existing_ideas, f"'{s.idea}' already in registry"


def test_meta_learner_needs_min_runs():
    """MetaLearner.fit() wirft ValueError wenn < 20 Runs vorhanden."""
    from sb.engine.meta_learner import MetaLearner

    learner = MetaLearner()
    with pytest.raises(ValueError, match="mindestens 20"):
        learner.fit(_make_fake_registry(10))


def test_meta_learner_feature_vector_covers_known_concepts():
    """_encode_idea() erzeugt Vektor der bekannte Konzepte abdeckt."""
    from sb.engine.meta_learner import MetaLearner

    learner = MetaLearner()
    vec = learner._encode_idea("FVG + BOS + HURST NY")
    assert isinstance(vec, list)
    assert len(vec) > 5
    assert sum(vec) >= 2  # mindestens FVG + BOS


def test_meta_learner_reproducible():
    """Gleiche Registry -> gleiche Vorschlaege."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    s1 = [r.idea for r in MetaLearner().fit(registry).suggest(n=5)]
    s2 = [r.idea for r in MetaLearner().fit(registry).suggest(n=5)]
    assert s1 == s2


def test_suggest_cli_command_exists():
    """suggest-Command ist im CLI verfügbar."""
    from typer.testing import CliRunner
    from sb.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["suggest", "--help"])
    assert result.exit_code == 0
    assert (
        "suggest" in result.output.lower()
        or "meta" in result.output.lower()
        or "vorschl" in result.output.lower()
    )


def test_meta_learner_rich_features_longer_vector():
    """Mit Registry-Stats hat der Feature-Vektor 64 Dimensionen."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    vec = learner._encode_idea_rich("FVG + JUDAS NY")
    # 57 Konzepte + 3 Sessions + 1 n_concepts + 3 Registry-Stats = 64
    assert len(vec) == 64


def test_meta_learner_calibrated_probs_between_0_and_1():
    """Kalibrierte Probs liegen immer zwischen 0 und 1."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    results = learner.suggest(n=5)
    for r in results:
        assert 0.0 <= r.prob_ab <= 1.0, f"Prob ausserhalb [0,1]: {r.prob_ab}"


def test_meta_learner_suggest_returns_suggestion_results():
    """suggest() gibt SuggestionResult mit korrekten Feldern zurueck."""
    from sb.engine.meta_learner import MetaLearner, SuggestionResult

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    results = learner.suggest(n=5)
    assert isinstance(results, list)
    assert all(isinstance(r, SuggestionResult) for r in results)
    for r in results:
        assert isinstance(r.idea, str)
        assert isinstance(r.band, str)
        assert r.band in {"auto_reject", "human_review", "auto_queue"}


def test_meta_learner_save_and_load(tmp_path):
    """MetaLearner kann gespeichert und geladen werden – gleiche Suggestions."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    suggestions_before = learner.suggest(n=5)

    path = tmp_path / "meta_learner.pkl"
    learner.save(path)

    loaded = MetaLearner.load(path)
    assert loaded.is_fitted
    suggestions_after = loaded.suggest(n=5)
    assert [r.idea for r in suggestions_before] == [r.idea for r in suggestions_after]


def test_meta_learner_feature_importance_returns_named_list():
    """feature_importance() gibt sortierte Liste von (name, float) zurueck."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    top = learner.feature_importance(top_n=5)
    assert len(top) == 5
    assert all(isinstance(name, str) and isinstance(imp, float) for name, imp in top)
    imps = [imp for _, imp in top]
    assert imps == sorted(imps, reverse=True)


def test_meta_learner_explore_ratio():
    """explore_ratio=0.5 liefert genau 10 Vorschlaege ohne Duplikate."""
    from sb.engine.meta_learner import MetaLearner

    registry = _make_fake_registry(60)
    learner = MetaLearner()
    learner.fit(registry)
    results = learner.suggest(n=10, explore_ratio=0.5)
    assert len(results) == 10
    ideas = [r.idea for r in results]
    assert len(set(ideas)) == 10  # keine Duplikate


def test_suggest_cli_saves_to_db(tmp_path):
    """suggest-Command speichert Vorschlaege in suggestions-Tabelle."""
    from typer.testing import CliRunner
    from sb.cli import app
    from sb.memory.db import BuilderDB

    # Fake-Registry: trials NOT NULL beachten
    db = BuilderDB(db_path=tmp_path / "builder.db")
    try:
        for i in range(25):
            db._execute_write(
                "INSERT INTO build_runs (idea, trials, tier, avg_oos_pf) VALUES (?, ?, ?, ?)",
                (
                    f"FVG + BOS_{i} NY",
                    50,
                    "B" if i % 3 == 0 else "C",
                    1.5 if i % 3 == 0 else 0.9,
                ),
            )
    finally:
        db.close()

    runner = CliRunner()
    result = runner.invoke(app, ["suggest", "--output", str(tmp_path), "--n", "5"])
    assert result.exit_code == 0, result.output

    db = BuilderDB(db_path=tmp_path / "builder.db")
    try:
        pending = db.get_pending_suggestions()
        assert len(pending) >= 1
        assert all(s["status"] == "pending" for s in pending)
    finally:
        db.close()
