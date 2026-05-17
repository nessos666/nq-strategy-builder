from __future__ import annotations

# Basis-Entries (bereits getestet)
BASE_ENTRIES = [
    "Judas Swing",
    "SWEEP",
    "BOS",
    "CHoCH",
    "Silver Bullet",
]

# Erweiterte Entries (neu)
EXTENDED_ENTRIES = [
    "MMXM",
    "NDOG",
]

ALL_ENTRIES = BASE_ENTRIES + EXTENDED_ENTRIES

FILTERS = [
    "FVG",
    "OB",
    "OTE",
    "IFVG",
]

SESSIONS = [
    "London",
    "NY",
    "Asia",
]

# Science-Filter nur für Basis-Entries × FVG × alle Sessions
SCIENCE_FILTERS = [
    "Hurst",
    "GARCH",
]


def generate_ideas() -> list[str]:
    """Generiert ~100 Kombinationen: Entry × Filter × Session + Science-Varianten."""
    ideas: list[str] = []

    # Block 1: Alle Entries × FVG/OB/OTE × alle Sessions (7×3×3 = 63)
    for entry in ALL_ENTRIES:
        for f in ["FVG", "OB", "OTE"]:
            for session in SESSIONS:
                ideas.append(f"{entry} + {f} {session}")

    # Block 2: Alle Entries × IFVG × alle Sessions (7×3 = 21)
    for entry in ALL_ENTRIES:
        for session in SESSIONS:
            ideas.append(f"{entry} + IFVG {session}")

    # Block 3: Basis-Entries × Science-Filter × FVG × alle Sessions (5×2×3 = 30)
    for entry in BASE_ENTRIES:
        for science in SCIENCE_FILTERS:
            for session in SESSIONS:
                ideas.append(f"{entry} + FVG + {science} {session}")

    # Duplikate entfernen, Reihenfolge erhalten
    seen: set[str] = set()
    unique: list[str] = []
    for idea in ideas:
        if idea not in seen:
            seen.add(idea)
            unique.append(idea)

    return unique
