export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_steer.py "$@" \
    --exp_name=pen_in_bowl \
    --method=hgdagger \
    --checkpoint_path=pen_in_bowl_hgdagger \
    --actor
