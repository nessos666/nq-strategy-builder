from __future__ import annotations

from pathlib import Path

import yaml

from sb.models import KnowledgeCtx

_DEFAULT_SOURCES = (
    Path(__file__).parent.parent.parent / "knowledge_sources" / "sources.yaml"
)


def resolve_pda_library_dirs(cfg: dict, base_dir: Path) -> list[Path]:
    """Liest alle Signal-Generator-Bibliotheken (pda/smc/science) aus der YAML-Konfiguration."""
    path_list: list[Path] = []
    for lib_key in ("pda_library", "smc_library", "science_library"):
        lib_cfg = cfg.get(lib_key, {}) or {}
        paths_raw = lib_cfg.get("paths")
        if isinstance(paths_raw, list) and paths_raw:
            for p in paths_raw:
                if p:
                    path_list.append(_resolve_cfg_path(str(p), base_dir))
        else:
            single = lib_cfg.get("path", "")
            if single:
                path_list.append(_resolve_cfg_path(str(single), base_dir))
    return path_list


def load_knowledge(sources_path: Path | None = None) -> KnowledgeCtx:
    """Lädt alle Wissensquellen aus sources.yaml."""
    sources_path = sources_path or _DEFAULT_SOURCES
    base_dir = sources_path.parent

    cfg: dict = {}
    if sources_path.exists():
        with open(sources_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    return KnowledgeCtx(
        pda_algos=_load_pda_algos(cfg, base_dir=base_dir),
        known_errors=_load_text_file(
            _resolve_cfg_path(cfg.get("fehlerprotokoll", {}).get("path", ""), base_dir)
        ),
        feedback_rules=_load_text_file(
            _resolve_cfg_path(cfg.get("feedback_rules", {}).get("path", ""), base_dir)
        ),
        learnings=_load_markdown_files(cfg.get("se_knowledge", {}), base_dir=base_dir),
        ideas=_load_markdown_files(cfg.get("ideen_bibliothek", {}), base_dir=base_dir),
    )


def _resolve_cfg_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path_str:
        return path
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_pda_algos(cfg: dict, base_dir: Path) -> list[str]:
    stems: set[str] = set()
    for lib_key in ("pda_library", "smc_library", "science_library"):
        lib_cfg = cfg.get(lib_key, {}) or {}
        pattern = lib_cfg.get("pattern", "*_v2.py")
        paths_raw = lib_cfg.get("paths")
        if isinstance(paths_raw, list) and paths_raw:
            lib_roots = [_resolve_cfg_path(str(p), base_dir) for p in paths_raw if p]
        else:
            single = lib_cfg.get("path", "")
            lib_roots = [_resolve_cfg_path(single, base_dir)] if single else []
        for lib_path in lib_roots:
            if not lib_path.exists():
                continue
            for p in lib_path.glob(pattern):
                stems.add(p.stem)
    return sorted(stems)


def _load_text_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()][:200]
    except Exception:
        return []


def _load_markdown_files(src_cfg: dict, base_dir: Path) -> list[str]:
    src_path = _resolve_cfg_path(src_cfg.get("path", ""), base_dir)
    pattern = src_cfg.get("pattern", "*.md")
    max_files = max(int(src_cfg.get("max_files", 20)), 0)
    if not src_path.exists():
        return []
    results: list[str] = []
    for p in sorted(src_path.glob(pattern))[:max_files]:
        try:
            content = p.read_text(encoding="utf-8")
            results.append(f"=== {p.stem} ===\n{content[:500]}")
        except Exception:
            continue
    return results
