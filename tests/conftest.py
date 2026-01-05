import sys
from pathlib import Path

# Ensure project root and examples are on sys.path for imports in tests
ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

for p in [ROOT, EXAMPLES]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

