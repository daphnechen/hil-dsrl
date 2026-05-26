export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
CUDA_VISIBLE_DEVICES=1 python ../../train_steer.py "$@" \
    --exp_name=banana \
    --method=steer \
    --checkpoint_path=banana_steer \
    --actor
