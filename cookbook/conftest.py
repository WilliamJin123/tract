"""Make _providers importable when running cookbook examples via pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
