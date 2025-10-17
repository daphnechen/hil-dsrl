export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
sudo /home/daphne/miniconda3/envs/hilserl/bin/python examples/train_rlpd.py "$@" \
    --exp_name=furniture_bench \
    --checkpoint_path=rlpd_furniture_bench \
    --demo_path=/home/daphne/Desktop/jax-hitl-hil-serl/demo_data/furniture_bench_20_demos_2025-06-06_14-30-19.pkl \
    --learner \
    --save_video \
