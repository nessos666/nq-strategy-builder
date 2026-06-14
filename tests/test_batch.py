from __future__ import annotations

from pathlib import Path

from sb.commands.core import _read_ideas


def test_read_ideas_leere_zeilen(tmp_path: Path) -> None:
    f = tmp_path / "ideas.txt"
    f.write_text("Idee A\n\n\nIdee B\n", encoding="utf-8")
    assert _read_ideas(f) == ["Idee A", "Idee B"]


def test_read_ideas_kommentare(tmp_path: Path) -> None:
    f = tmp_path / "ideas.txt"
    f.write_text(
        "# Kommentar\nIdee A\n# Noch ein Kommentar\nIdee B\n", encoding="utf-8"
    )
    assert _read_ideas(f) == ["Idee A", "Idee B"]


def test_read_ideas_leer(tmp_path: Path) -> None:
    f = tmp_path / "ideas.txt"
    f.write_text("# Nur Kommentare\n\n", encoding="utf-8")
    assert _read_ideas(f) == []


def test_read_ideas_gemischt(tmp_path: Path) -> None:
    f = tmp_path / "ideas.txt"
    f.write_text(
        "# Header\nJudas Swing + FVG London\n\nSWEEP + OB NY\n",
        encoding="utf-8",
    )
    result = _read_ideas(f)
    assert len(result) == 2
    assert result[0] == "Judas Swing + FVG London"
    assert result[1] == "SWEEP + OB NY"
