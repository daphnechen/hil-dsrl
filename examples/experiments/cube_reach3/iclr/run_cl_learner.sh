export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
CUDA_VISIBLE_DEVICES="0" python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=hil_rico_14225 \
    --demo_path=/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr/demo_data/cube_reach3_20_demos_2025-11-14_12-01-28.pkl \
    --learner \
    --save_video \
    --method=hil \
