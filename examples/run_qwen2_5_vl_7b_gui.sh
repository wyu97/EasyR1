set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=/gy_1/share_302625455/models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
export RAY_DEBUG=legacy
python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_gui \
    trainer.n_gpus_per_node=8 
