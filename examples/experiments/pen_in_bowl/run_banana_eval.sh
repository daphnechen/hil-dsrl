#!/bin/bash
# Run eval sweep across all checkpoints, then plot results

CHECKPOINT_PATH="pen_in_bowl_steer-rico-may23"
# CHECKPOINT_PATH="pen_in_bowl_qoil"
EXP_NAME="pen_in_bowl"
METHOD="steer"
N_TRAJS=10

for step in 2000; do
    echo "========== Evaluating checkpoint $step =========="
    CUDA_VISIBLE_DEVICES=1 python ../../train_steer.py \
        --exp_name=$EXP_NAME \
        --method=$METHOD \
        --checkpoint_path=$CHECKPOINT_PATH \
        --actor \
        --eval_checkpoint_step=$step \
        --eval_n_trajs=$N_TRAJS
done

echo "========== All evals done, plotting... =========="
python plot_eval_results.py --checkpoint_path=$CHECKPOINT_PATH --n_trajs=$N_TRAJS

# steer no decay : 40%, 70%, 80%
# qoil: 60%, 85%, 95%