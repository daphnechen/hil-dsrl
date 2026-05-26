export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
CUDA_VISIBLE_DEVICES=0 python ../../train_steer.py "$@" \
    --exp_name=banana \
    --method=steer \
    --checkpoint_path=banana_steer \
    --demo_path=../../demo_data/banana_10_demos_2026-03-16_09-53-13.pkl \
    --learner \
    --save_video \
    --learnable_temperature=false \
    --temperature_init=0.005
