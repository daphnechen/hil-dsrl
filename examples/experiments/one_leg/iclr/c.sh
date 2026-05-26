# python ../../../record_success_fail.py --exp_name=one_leg
# python ../../../train_reward_classifier.py --exp_name=one_leg
# python ../../../run_environment.py --exp_name=one_leg
# python ../../../record_demos.py --exp_name=one_leg --successes_needed=20

python ../../../train_bc.py \
    --exp_name=cube_reach3 \
    --bc_checkpoint_path=debug_bc_yellow_cube \
    --train_steps=10000

# sudo /home/qirico/miniconda3/envs/hilserl/bin/python ../../../train_bc.py \
#     --exp_name=one_leg \
#     --bc_checkpoint_path=/home/qirico/Desktop/All-Weird/Human-Interventions/jax-hitl-hil-serl/examples/experiments/one_leg/side_only/debug_bc/checkpoint_99990 \
#     --eval_n_trajs=20

