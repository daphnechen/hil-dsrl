export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
CUDA_VISIBLE_DEVICES=0 python ../../train_steer.py "$@" \
    --exp_name=pen_in_bowl \
    --method=qoil \
    --checkpoint_path=pen_in_bowl_qoil \
    --demo_path=./demo_data/pen_in_bowl_20_demos_2026-05-23_08-01-05.pkl \
    --learner \
    --save_video \
    --learnable_temperature=false \
    --temperature_init=0.005 --bc_timestep_decay=0.0

# ../../demo_data/pen_in_bowl_10_demos_2026-05-21_16-21-56.pkl