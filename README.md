# DA-Mamba

DA-Mamba is a PyTorch research project for DA-Mamba: A Dual-Adaptive Mamba-based Mixture-of-Experts Network for Multi-Contrast MRI Super-Resolution.

## Overview

- Implements training and evaluation for DA-Mamba.
- Includes dataset handling, distributed training support, and logging with TensorBoard.
- Uses a custom `datasets` module and model definitions under `model/`.

## Key files

- `train.py` - main training script.
- `engine.py` - training and validation engine, dataset loading, and load balancing loss.
- `datasets/dataset.py` - multi degeneration types.
- `model/` - model implementations and MoE modules.

## Notes

- This repository is designed for research experiments and may require custom configuration for your environment.
- Make sure GPU and distributed training settings are configured properly if used.
