# python ../../../record_success_fail.py --exp_name=cube_reach3
# python ../../../train_reward_classifier.py --exp_name=cube_reach3
python ../../../run_environment.py --exp_name=cube_reach3
# python ../../../record_demos.py --exp_name=cube_reach3 --successes_needed=20
# python ../../../train_bc.py \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=debug_bc_pen2 \
#     --train_steps=100000

# python ../../../train_bc.py \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=debug_bc \
#     --train_steps=10000

# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_bc.py \
#     --exp_name=cube_reach3 \
#     --bc_checkpoint_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/cube_reach3/side_only/debug_bc/checkpoint_99990 \
#     --eval_n_trajs=20





# https://wandb.ai/social-rl/hil-serl-cubereach2/runs/cube_reach3_20251114_224537 hil_rico_14225