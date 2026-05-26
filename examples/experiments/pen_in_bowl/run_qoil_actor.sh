export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_steer.py "$@" \
    --exp_name=pen_in_bowl \
    --method=qoil \
    --checkpoint_path=pen_in_bowl_qoil \
    --actor \
    --learnable_temperature=false \
    --temperature_init=0.005 --bc_timestep_decay=0.0
