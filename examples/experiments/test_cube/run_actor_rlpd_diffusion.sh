#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Expand demo path glob and pass each file separately
DEMO_ARGS=""
for demo_file in /home/daphne/Desktop/jax-hitl-hil-serl/demo_data/test_cube_20_demos_*.pkl; do
    if [ -f "$demo_file" ]; then
        DEMO_ARGS="$DEMO_ARGS --demo_path=$demo_file"
    fi
done

sudo /home/daphne/miniconda3/envs/hilserl/bin/python examples/train_rlpd_diffusion.py "$@" \
    --exp_name=test_cube \
    --checkpoint_path=test_cube_rlpd_diffusion \
    --diffusion_bc_checkpoint_path="${SCRIPT_DIR}/../../test_cube_bc_diffusion/checkpoint_19990" \
    $DEMO_ARGS \
    --actor
