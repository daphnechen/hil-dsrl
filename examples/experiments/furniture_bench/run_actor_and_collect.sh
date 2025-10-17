export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
sudo /home/daphne/miniconda3/envs/hilserl/bin/python examples/train_rlpd_and_collect.py "$@" \
    --exp_name=furniture_bench \
    --checkpoint_path=rlpd_furniture_bench \
    --actor \
    --save_trajectories \
    --trajectory_save_period 1 \
