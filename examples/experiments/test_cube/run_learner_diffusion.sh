export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
sudo /home/daphne/miniconda3/envs/hilserl/bin/python examples/train_bc_diffusion.py "$@" \
    --exp_name=test_cube \
    --bc_checkpoint_path=test_cube_bc_diffusion \
    --train_steps=20000 \
    --action_horizon=1 \
    --num_train_timesteps=20 \
    --num_inference_timesteps=8
