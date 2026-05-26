# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
# CUDA_VISIBLE_DEVICES="0" python ../../../train_steer.py "$@" \
#     --exp_name=cube_reach3 \
#     --checkpoint_path=steer_rico_decay00001_19155 \
#     --demo_path=/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/demo_data/cube_reach3_20_demos_2025-11-19_15-15-07.pkl \
#     --learner \
#     --save_video \
#     --method=steer --bc_timestep_decay=0.0001

CUDA_VISIBLE_DEVICES="0" python ../../../train_steer.py "$@" \
    --exp_name=cube_reach3 \
    --checkpoint_path=steer_decay_3e_5_gripper0005_sriyash_1124 \
    --demo_path=/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/demo_data/cube_reach3_20_demos_2025-11-19_15-15-07.pkl \
    --learner \
    --save_video \
    --method=steer --bc_timestep_decay=0.00003
