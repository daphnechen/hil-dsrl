export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
CUDA_VISIBLE_DEVICES="1" python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=hil_rico_14225 \
    --actor \
    --method=hil \
