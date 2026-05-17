#!/usr/bin/env python3
# sb.py – Entry Point für ./sb.py "Idee"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sb.cli import main

if __name__ == "__main__":
    main()
