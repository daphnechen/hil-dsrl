export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_steer.py "$@" \
    --exp_name=pen_in_bowl \
    --method=hgdagger \
    --checkpoint_path=pen_in_bowl_hgdagger \
    --demo_path=../../demo_data/pen_in_bowl_hgdagger_10_demos.pkl \
    --learner \
    --save_video
