export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_steer.py "$@" \
    --exp_name=shirt_unbutton \
    --method=steer \
    --checkpoint_path=shirt_unbutton_steer \
    --actor
