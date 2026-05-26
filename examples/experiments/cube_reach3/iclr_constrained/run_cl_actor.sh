# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
# CUDA_VISIBLE_DEVICES="1" python ../../../train_rlif.py "$@" \
#     --exp_name=cube_reach3 \
#     --checkpoint_path=hgdagger_rico_1721 \
#     --actor \
#     --method=hgdagger \
#     --use_bc_loss \

# CUDA_VISIBLE_DEVICES="1" python ../../../train_steer.py "$@" \
#     --exp_name=cube_reach3 \
#     --checkpoint_path=steer_rico_decay00001_19155 \
#     --actor \
#     --method=steer --bc_timestep_decay=0.0001

CUDA_VISIBLE_DEVICES="1" python ../../../train_steer.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=steer_decay_3e_5_gripper0005_sriyash_1124 \
    --actor \
    --method=steer --bc_timestep_decay=0.00003
