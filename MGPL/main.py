"""
MGPL (Mixture of Gaussian Prototype Learning) for Open-Set Recognition.

This is the entry point for running MGPL-based open-set recognition
on communication signal (LLR) data.

Usage:
    # Train the model
    python MGPL/main.py --mode train

    # Evaluate (requires trained model)
    python MGPL/main.py --mode eval

    # Train + Evaluate
    python MGPL/main.py --mode both

The MGPL method uses:
  - VAE-style 1D ResNet encoder-decoder with lateral connections
  - Learnable Gaussian prototypes (n_classes × n_sub_prototypes)
  - Reconstruction + KLD + Entropy + Discriminative loss
  - Nearest-prototype classification
  - Minimum distance to prototypes as OSR outlier score
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from MGPL.train import main as train_main
from MGPL.evaluator import main as eval_main


def main():
    parser = argparse.ArgumentParser(
        description='MGPL: Mixture of Gaussian Prototype Learning for OSR'
    )
    parser.add_argument(
        '--mode', type=str, default='both',
        choices=['train', 'eval', 'both'],
        help='Run mode: train, eval, or both'
    )
    args = parser.parse_args()

    if args.mode in ('train', 'both'):
        print('=' * 60)
        print('MGPL Training Phase')
        print('=' * 60)
        train_main()

    if args.mode in ('eval', 'both'):
        print('=' * 60)
        print('MGPL Evaluation Phase')
        print('=' * 60)
        eval_main()

    if args.mode == 'both':
        print('=' * 60)
        print('MGPL Pipeline Complete!')
        print('=' * 60)


if __name__ == '__main__':
    main()
