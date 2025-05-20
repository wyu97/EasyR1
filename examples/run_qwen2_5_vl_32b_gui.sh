set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

MODEL_PATH=/gy_1/share_302625455/models/Qwen2.5-VL-32B-Instruct  # replace it with your local file path
export RAY_DEBUG=legacy
python3 -m verl.trainer.main \
    config=examples/grpo_32b.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=gui_rl_32b_$(TZ='America/Los_Angeles' date +%m_%d_%Y_%H%M) \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=/gy_1/share_302625455/user/yuchengshi/qwenvl_verl_32b_$(TZ='America/Los_Angeles' date +%m_%d_%Y_%H%M)