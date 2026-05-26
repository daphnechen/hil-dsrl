# python ../../../record_success_fail.py --exp_name=cube_reach_iclr
# python ../../../train_reward_classifier.py --exp_name=cube_reach_iclr
python ../../../run_environment.py --exp_name=cube_reach_iclr
# python ../../../record_demos.py --exp_name=cube_reach_iclr --successes_needed=20

# python ../../../train_bc.py \
#     --exp_name=cube_reach_iclr \
#     --bc_checkpoint_path=debug_bc \
#     --train_steps=10000
