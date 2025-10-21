export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_bc.py "$@" \
    --exp_name=test_cube \
    --checkpoint_path=test_cube_bc \
    --actor
