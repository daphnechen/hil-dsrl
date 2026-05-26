export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
CUDA_VISIBLE_DEVICES=0 python ../../train_steer.py "$@" \
    --exp_name=shirt_unbutton \
    --method=qoil \
    --checkpoint_path=shirt_unbutton_qoil \
    --demo_path=../../demo_data/shirt_unbutton_10_demos_2026-05-25_02-19-18.pkl \
    --learner \
    --save_video \
    --learnable_temperature=false \
    --temperature_init=0.005 \
    --bc_timestep_decay=0.0
