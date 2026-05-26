export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
sudo /home/qirico/miniconda3/envs/hilserl3/bin/python ../../../train_rlif.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=debug_bc \
    --actor \
    --eval_checkpoint_step=9900 \
    --eval_n_trajs=20 \
