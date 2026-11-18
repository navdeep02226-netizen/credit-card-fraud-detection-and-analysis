"""Dashboard components. Importing this package also puts the project root on
sys.path so `from utils...` works no matter where Streamlit is launched from."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
