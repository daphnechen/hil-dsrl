export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
# python ../../train_bc.py "$@" \
sudo /home/daphne/miniconda3/envs/hilserl/bin/python examples/train_bc.py "$@" \
    --exp_name=test_cube \
    --bc_checkpoint_path=$(pwd)/test_cube_bc \
    --eval_n_trajs=10
