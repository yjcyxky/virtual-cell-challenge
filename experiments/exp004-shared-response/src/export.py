"""Export an unchanged shared-response checkpoint using the common VCC writer."""
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT.parent / 'exp003-context-module-cvae/src'))

from shared_model import SharedResponseCVAE
from submission import argument_parser, main

if __name__ == '__main__':
    main(argument_parser().parse_args(), EXPERIMENT, SharedResponseCVAE)
