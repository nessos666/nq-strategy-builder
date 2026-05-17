from __future__ import annotations


from sb.engine.knowledge import load_knowledge
from sb.models import KnowledgeCtx


def test_load_knowledge_returns_ctx(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(f"""
pda_library:
  path: {tmp_path / "pdas"}
  pattern: "algo_*.py"
backtest_data:
  path: {tmp_path / "data.parquet"}
se_knowledge:
  path: {tmp_path / "knowledge"}
  pattern: "*.md"
  max_files: 10
fehlerprotokoll:
  path: {tmp_path / "fehler.md"}
feedback_rules:
  path: {tmp_path / "feedback.md"}
ideen_bibliothek:
  path: {tmp_path / "ideen"}
  pattern: "*.md"
  max_files: 10
""")
    ctx = load_knowledge(sources_path=sources_yaml)
    assert isinstance(ctx, KnowledgeCtx)
    assert isinstance(ctx.pda_algos, list)
    assert isinstance(ctx.known_errors, list)
    assert isinstance(ctx.learnings, list)


def test_load_pda_algos(tmp_path):
    pdas = tmp_path / "pdas"
    pdas.mkdir()
    (pdas / "algo_01_order_block.py").write_text("# OB")
    (pdas / "algo_09_fair_value_gap.py").write_text("# FVG")
    (pdas / "other_file.py").write_text("# ignorieren")

    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(f"""
pda_library:
  path: {pdas}
  pattern: "algo_*.py"
backtest_data:
  path: {tmp_path / "x.parquet"}
se_knowledge:
  path: {tmp_path / "k"}
  pattern: "*.md"
  max_files: 10
fehlerprotokoll:
  path: {tmp_path / "f.md"}
feedback_rules:
  path: {tmp_path / "fb.md"}
ideen_bibliothek:
  path: {tmp_path / "i"}
  pattern: "*.md"
  max_files: 10
""")
    ctx = load_knowledge(sources_path=sources_yaml)
    assert "algo_01_order_block" in ctx.pda_algos
    assert "algo_09_fair_value_gap" in ctx.pda_algos
    assert "other_file" not in ctx.pda_algos


def test_load_pda_algos_multiple_paths(tmp_path):
    """Mehrere Bibliotheks-Pfade werden zusammengeführt (ohne Duplikate)."""
    pdas_a = tmp_path / "pdas_a"
    pdas_b = tmp_path / "pdas_b"
    pdas_a.mkdir()
    pdas_b.mkdir()
    (pdas_a / "algo_01_a.py").write_text("# A")
    (pdas_b / "algo_02_b.py").write_text("# B")
    (pdas_b / "algo_01_a.py").write_text("# Dup-Stem in zweitem Ordner")

    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(f"""
pda_library:
  paths:
    - {pdas_a}
    - {pdas_b}
  pattern: "algo_*.py"
backtest_data:
  path: {tmp_path / "x.parquet"}
se_knowledge:
  path: {tmp_path / "k"}
  pattern: "*.md"
  max_files: 10
fehlerprotokoll:
  path: {tmp_path / "f.md"}
feedback_rules:
  path: {tmp_path / "fb.md"}
ideen_bibliothek:
  path: {tmp_path / "i"}
  pattern: "*.md"
  max_files: 10
""")
    ctx = load_knowledge(sources_path=sources_yaml)
    assert "algo_01_a" in ctx.pda_algos
    assert "algo_02_b" in ctx.pda_algos
    assert ctx.pda_algos == sorted(ctx.pda_algos)
