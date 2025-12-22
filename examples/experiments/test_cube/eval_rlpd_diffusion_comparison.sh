#!/bin/bash
# Wrapper script for convergence analysis evaluation

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run evaluation with sudo to access checkpoint files (saved with sudo during training)
sudo /home/daphne/miniconda3/envs/hilserl/bin/python ../../eval_rlpd_diffusion_comparison.py "$@" \
    --exp_name=test_cube
