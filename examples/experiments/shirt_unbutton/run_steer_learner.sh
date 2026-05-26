export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_steer.py "$@" \
    --exp_name=shirt_unbutton \
    --method=steer \
    --checkpoint_path=shirt_unbutton_steer \
    --demo_path=../../demo_data/shirt_unbutton_10_demos.pkl \
    --learner \
    --save_video \
    --learnable_temperature=false \
    --temperature_init=0.005
