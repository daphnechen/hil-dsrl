# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
# python ../../../train_bc.py "$@" \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr/debug_bc \
#     --train_steps=9990 \
#     --eval_n_trajs=20 \


export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=hil_rico_14225 \
    --actor \
    --eval_checkpoint_step=10000 \
    --eval_n_trajs=20 \
