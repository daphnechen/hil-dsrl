#!/usr/bin/env bash
# Q-OIL actor for shirt_unbutton.
# Needs sudo for SpaceMouse HID access; uses hilserl2 python directly so the
# conda env isn't stripped by sudo.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

sudo -E XLA_PYTHON_CLIENT_PREALLOCATE=false \
     XLA_PYTHON_CLIENT_MEM_FRACTION=.1 \
     DISPLAY="$DISPLAY" \
     XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
     /home/daphne/miniconda3/envs/hilserl2/bin/python ../../train_steer.py "$@" \
    --exp_name=shirt_unbutton \
    --method=qoil \
    --checkpoint_path=shirt_unbutton_qoil \
    --actor \
    --learnable_temperature=false \
    --temperature_init=0.005 \
    --bc_timestep_decay=0.0
