from sb.combinator import generate_ideas, BASE_ENTRIES, ALL_ENTRIES, FILTERS, SESSIONS


def test_generate_ideas_count():
    ideas = generate_ideas()
    assert len(ideas) >= 100


def test_generate_ideas_no_duplicates():
    ideas = generate_ideas()
    assert len(ideas) == len(set(ideas))


def test_generate_ideas_format():
    ideas = generate_ideas()
    assert "BOS + FVG Asia" in ideas
    assert "SWEEP + OB London" in ideas
    assert "Judas Swing + FVG NY" in ideas


def test_entry_signals_count():
    assert len(BASE_ENTRIES) == 5


def test_all_entries_includes_mmxm_ndog():
    assert "MMXM" in ALL_ENTRIES
    assert "NDOG" in ALL_ENTRIES


def test_filters_count():
    assert len(FILTERS) == 4


def test_sessions_count():
    assert len(SESSIONS) == 3
