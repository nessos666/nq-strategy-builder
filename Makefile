PYTHON := .venv/bin/python
PYTEST := $(PYTHON) -m pytest

.PHONY: test test-fast test-v install maintain

test:
	$(PYTEST) tests/ -q

test-fast:
	$(PYTEST) tests/ -q --ignore=tests/test_signal_cache.py --ignore=tests/test_integration.py

test-v:
	$(PYTEST) tests/ -v

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

maintain:
	$(PYTHON) sb.py maintain --output output
