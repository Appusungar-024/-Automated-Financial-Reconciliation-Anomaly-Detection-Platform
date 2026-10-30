"""Grade the pipeline against simulated ground truth.   python validate_ground_truth.py --help"""
import sys

from recon.cli import evaluate_main

if __name__ == "__main__":
    sys.exit(evaluate_main())
