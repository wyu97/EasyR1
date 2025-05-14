# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager, GUIRewardManager
from .config import PPOConfig
#from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from .ray_gui_trainer import RayPPOGUITrainer, ResourcePoolManager, Role
import os
import pdb

def main():
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.config)
    del cli_args.config

    default_config = OmegaConf.structured(PPOConfig())
    ppo_config = OmegaConf.merge(default_config, file_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)
    auth_token_path = os.path.expanduser("~/.emulator_console_auth_token")
    with open(auth_token_path, "r") as f:
        auth_token = f.read().strip()
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "EMULATOR_AUTH_TOKEN": auth_token}})

    ray.get(main_task.remote(ppo_config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config: PPOConfig):
    config.deep_post_init()
    print(json.dumps(config.to_dict(), indent=2))
    # instantiate tokenizer
    tokenizer = get_tokenizer(config.worker.actor.model.model_path)
    processor = get_processor(config.worker.actor.model.model_path, use_fast=True)

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(FSDPWorker),
        Role.Critic: ray.remote(FSDPWorker),
        Role.RefPolicy: ray.remote(FSDPWorker),
    }
    #breakpoint()

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    reward_fn = GUIRewardManager(
        tokenizer=tokenizer, num_examine=1
    )
    val_reward_fn = GUIRewardManager(
        tokenizer=tokenizer, num_examine=1
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOGUITrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
