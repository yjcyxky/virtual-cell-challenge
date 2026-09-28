"""Local model plus frozen common providers, explicitly covered by DAG references."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([str(ROOT/"src"), str(ROOT/"experiments/init-linear/src")])
sys.path.insert(0, str(ROOT/"scripts"))
