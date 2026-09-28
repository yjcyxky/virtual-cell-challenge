"""Resolve all configuration and bind before any run output or W&B creation."""
import runtime
import argparse
import json
import os
from pathlib import Path
from data import ROOT
from research import bind
from vcc_mechanism.runner import run_bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    config = json.loads(parser.parse_args().config.read_text())
    resume = os.environ.get('VCC_RESEARCH_RESUME') == '1'
    research = bind(ROOT, config['run_id'], config, resume=resume)
    run_bound(config, research, resume)


if __name__ == '__main__':
    main()
