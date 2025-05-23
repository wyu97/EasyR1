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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.dataset import RLHFDataset, collate_fn, GUIDataset
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..utils.tracking import Tracking
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer
from ..environment.android_emulator import BatchedAndroidEnv, Qwen25VLEvaluator
from ..environment.qwen25vl_agent import Qwen25VLAgent
import random
import json
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_TRAJ = "grpo_traj"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma=1.0, lam=1.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_TRAJ:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        traj_index = data.non_tensor_batch["traj_id"]
        step_index = data.non_tensor_batch["step_id"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_traj_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index, traj_index=traj_index, step_index=step_index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError

    return data


def reduce_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens
    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, task_set + "_" + task_split + ".txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks

class RayPPOGUITrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if not any(estimator.value == config.algorithm.adv_estimator for estimator in AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        self.all_tasks = load_task_file('/cq/share_1603164/user/kaixinma/digirl/digirl/environment/android/assets/task_set/', 'general', 'train')
        self.test_tasks = load_task_file('/cq/share_1603164/user/kaixinma/digirl/digirl/environment/android/assets/task_set/', 'general', 'test')
        self._create_dataloader()

        base_port = 5556
        all_avd_names = [f"test{i}" for i in range(1,self.config.worker.rollout.env.bsize+1)]
        all_udids = [f"emulator-{base_port+2*i}" for i in range(self.config.worker.rollout.env.bsize)]
        evaluators = [Qwen25VLEvaluator() for _ in range(self.config.worker.rollout.env.bsize)]
        self.env = BatchedAndroidEnv(avd_name=self.config.worker.rollout.env.avd_name,
                                        cache_avd_names=all_avd_names,
                                        udids=all_udids,
                                        appium_base_port=base_port+1198,
                                        android_avd_home=self.config.worker.rollout.env.android_avd_home,
                                        emulator_path=self.config.worker.rollout.env.emulator_path,
                                        adb_path=self.config.worker.rollout.env.adb_path,
                                        max_steps=self.config.worker.rollout.env.max_steps,
                                        run_headless=True,
                                        evaluators=evaluators,
                                        temp_path=os.path.join(self.config.worker.rollout.env.save_path, "images"),
                                        save_images=True,
                                        all_tasks=None,
                                        record=False,
                                        image_size=self.config.worker.rollout.env.image_size
                                        )

    def _create_dataloader(self):
        total_train_samples = self.config.trainer.max_steps * self.config.data.rollout_batch_size
        total_val_samples = self.config.trainer.val_steps * self.config.data.rollout_batch_size * self.config.worker.rollout.env.n
        train_sample_ids = list(range(total_train_samples))
        val_sample_ids = list(range(total_val_samples))
        if self.config.data.shuffle:
            random.seed(self.config.data.seed)
            random.shuffle(train_sample_ids)
        self.train_dataloader = [train_sample_ids[i : i + self.config.data.rollout_batch_size] for i in range(0, total_train_samples, self.config.data.rollout_batch_size)]
        self.val_dataloader = [val_sample_ids[i : i + self.config.data.rollout_batch_size * self.config.worker.rollout.env.n] for i in range(0, total_val_samples, self.config.data.rollout_batch_size * self.config.worker.rollout.env.n)]
        self.training_steps = self.config.trainer.max_steps
        self.config.worker.actor.optim.training_steps = self.config.trainer.max_steps
        self.config.worker.critic.optim.training_steps = self.config.trainer.max_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples"""

        generations_to_log = self.config.trainer.val_generations_to_log

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and "wandb" not in self.config.trainer.logger:
            print("WARNING: `val_generations_to_log` is set, but no wandb logger is found.")
            return

        import wandb

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_lst = []
        # Lists to collect samples for the table
        sample_inputs_outputs = []
        for test_batch in self.val_dataloader:
            batch_tasks = [self.test_tasks[b] for b in test_batch]
            batch_dict = {'uid': torch.Tensor(test_batch), 'tasks': np.array(batch_tasks, dtype=object)}
           
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            _, batch_reward, input_output_for_log = self._rollout_batch_in_env(batch, {}, True)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test)
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print("validation generation end")

            # Store generated outputs
            # output_ids = test_output_gen_batch.batch["responses"]
            # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # # evaluate using reward_function
            # reward_tensor = self.val_reward_fn(test_batch)

            # # Store scores
            # scores = reward_tensor.sum(-1).cpu().tolist()
            # sample_scores.extend(scores)

            # reward_tensor_lst.append(reward_tensor)
            for r, exp in zip(batch_reward, input_output_for_log):
                if len(exp) > 0:
                    exp[-1]['reward'] = r
            reward_lst.extend(batch_reward)
            sample_inputs_outputs.extend(input_output_for_log)

        #self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        #reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        output_path = os.path.join(self.config.worker.rollout.env.save_path, f"step_{self.global_steps}_val_results.json")
        with open(output_path, 'w') as fout:
            json.dump(sample_inputs_outputs, fout, indent=4)
        reward_score = sum(reward_lst) / len(reward_lst)
        return {"val/test_score": reward_score}

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        print ('self resource_pool_to_cls', self.resource_pool_to_cls)
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
        print ('all_wg', all_wg)
        if self.use_critic:
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: {save_checkpoint_path}/global_step_{global_steps}/actor
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_steps}")
        actor_path = os.path.join(folder_path, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(
                critic_path,
                self.global_steps,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        #dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(self.train_dataloader, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, "latest_global_step.txt")
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should be in `global_step_xxx` format.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_steps = int(self.config.trainer.load_checkpoint_path.split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
        )
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(
                critic_path, remove_ckpt_after_load=self.config.trainer.remove_ckpt_after_load
            )

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader = dataloader_state
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _rollout_batch_in_env(self, batch, timing_raw, val=False):
        all_gen_batch_out = []
        traj_ids = [str(uuid.uuid4()) for _ in range(len(batch.batch))]
        agent = Qwen25VLAgent(policy_lm=self.config.worker.actor.model.model_path, max_prompt_len=6144)
        input_output_for_log = []
        for idx in range(0, len(batch), self.config.worker.rollout.env.bsize):
            for _ in range(self.config.worker.rollout.env.bsize):
                input_output_for_log.append([])
            #print (input_output_for_log, len(input_output_for_log))
            with _timer("env_setup", timing_raw):

                tasks = batch.non_tensor_batch['tasks'][idx:idx+self.config.worker.rollout.env.bsize]
                for _ in range(5):
                    try:
                        batch_obs = self.env.reset(tasks)
                        break
                    except Exception as e:
                        print(f"Error in environment reset")
                        print(e)
                        if hasattr(self.env, "reset_appium"):
                            print("Resetting appium")
                            self.env.reset_appium()
                        continue
                batch_reward = [0]*len(batch_obs)
                batch_done = [0]*len(batch_obs)
                print ('batch_obs', batch_obs)
            # generate a batch
            if any([ob is None for ob in batch_obs]):
                print ('Screenshot error during reset, try to copy the observation from other rollouts')
                for i in range(len(batch_obs)):
                    if batch_obs[i] is None:
                        start = i // self.config.worker.rollout.env.n
                        end = start + self.config.worker.rollout.env.n
                        for j in range(start, end):
                            if batch_obs[j] is not None:
                                batch_obs[i] = batch_obs[j].copy()
                                break
            for _ in range(self.config.worker.rollout.env.max_steps):
                prompts = agent.get_action_inputs(batch_obs)
                if val:
                    prompts.meta_info['do_sample'] = False
                #print ('prompts finished')
                with _timer("gen", timing_raw):  # wg: worker group
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts)
                print ('gen_batch_output', gen_batch_output)
                responses = gen_batch_output.batch['responses']

                #print ('responses', responses)
                actions = agent.processor.batch_decode(responses, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                #breakpoint()
                #print ('actions', actions, len(actions))
                this_gen_out = []
                with _timer("env_step", timing_raw):
                    batch_results = self.env.step(actions)
                    print ('batch_results', batch_results)
                    #breakpoint()
                    for i, res in enumerate(batch_results):
                        if res is not None and res[0] is not None:
                            print ('res', res)
                            input_output_for_log[idx+i].append({'history': batch_obs[i]['history'], 'task': batch_obs[i]['task'], 'image_path': batch_obs[i]['image_path'], 'action':actions[i]})
                            batch_obs[i] = res[0]
                            batch_reward[i] = res[1]
                            batch_done[i] = res[2]
                            single_instance = DataProto.from_dict(gen_batch_output[i].batch.unsqueeze(0), {k: np.expand_dims(v, 0) for k, v in gen_batch_output[i].non_tensor_batch.items()}, gen_batch_output[i].meta_info)
                            single_instance.non_tensor_batch['uid'] = np.array([batch.batch['uid'][idx+i]])
                            single_instance.non_tensor_batch['traj_id'] = np.array([traj_ids[idx+i]], dtype=object)
                            single_instance.non_tensor_batch['step_id'] = np.array([_])
                            single_instance.non_tensor_batch['reward'] = np.array([res[1]])
                            this_gen_out.append(single_instance)
                        else:
                            this_gen_out.append(None)
                            if res is not None and res[0] is None:
                                # env failure
                                batch_done[i] = 1
                #print ('all gen batch output', len(all_gen_batch_out), all_gen_batch_out)
                all_gen_batch_out.append(this_gen_out)  
                print ('after step', batch_reward, batch_done)
                #breakpoint()
                if all(batch_done):
                    break
        return all_gen_batch_out, batch_reward, input_output_for_log

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=self.config.to_dict(),
        )
        val_metrics: Optional[Dict[str, Any]] = None
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}.")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.val_only:
                return

        for _ in range(self.config.trainer.total_episodes):
            for batch in self.train_dataloader:
                self.global_steps += 1
                # if self.global_steps >= self.training_steps:
                #     break
                batch_tasks = [self.all_tasks[b] for b in batch]
                batch_dict = {'uid': torch.Tensor(batch), 'tasks': np.array(batch_tasks, dtype=object)}
                print ('task batch', batch)
                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.worker.rollout.env.n, interleave=True)
                print ('batch after repeat', batch)
                # # pop those keys for generation
                # if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                #     gen_batch = batch.pop(
                #         batch_keys=["input_ids", "attention_mask", "position_ids"],
                #         non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                #     )
                # else:
                #     gen_batch = batch.pop(
                #         batch_keys=["input_ids", "attention_mask", "position_ids"],
                #         non_tensor_batch_keys=["raw_prompt_ids"],
                #     )
                
                assert len(batch) % self.config.worker.rollout.env.bsize == 0
                with _timer("step", timing_raw):
                    try:
                        all_gen_batch_out, batch_reward, _ = self._rollout_batch_in_env(batch, timing_raw)
                    except:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    metrics.update({'critic/traj_reward/mean': sum(batch_reward)/len(batch_reward)})
                    # if self.config.algorithm.adv_estimator == "remax":
                    #     with _timer("gen_max", timing_raw):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info["do_sample"] = False
                    #         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    #         batch = batch.union(gen_baseline_output)
                    #         reward_baseline_tensor = self.reward_fn(batch)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                    #         batch.batch["reward_baselines"] = reward_baseline_tensor
                    #         del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array(
                    #     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    # )
                    # repeat to align with repeated responses in rollout
                    # 
                    # batch = batch.union(gen_batch_output)
                    print ('all gen batch out_fit', all_gen_batch_out)
                    real_batch = []
                    for step in all_gen_batch_out:
                        step = [s for s in step if s is not None]
                        print ('step', step)
                        if len(step) > 0:
                            real_batch.append(DataProto.concat(step))
                    batch = DataProto.concat(real_batch)
                    print ('batch after concat', batch)
                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_reward_model:
                            raise NotImplementedError

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # KM: Here we reorder the batch using the advantages, we basically put the negatives in the end for each chunk, which is probably safer to throw away, we do this sorting in the chunked way because we may need to throw away more data later after DP split
                    curr_idx = [(torch.max(batch.batch["advantages"][bid]), bid) for bid in range(len(batch))]
                    reorder_idx = []
                    chunk_size = len(batch) // 8
                    for _ in range(8):
                        if _ == 7:
                            chunk = curr_idx[_*chunk_size:]
                        else:
                            chunk = curr_idx[_*chunk_size:(_+1)*chunk_size]
                        reorder_idx.extend(sorted(chunk, key=lambda x: x[0], reverse=True))
                    print ('reorder_idx', reorder_idx)
                    reorder_idx = torch.tensor([r[1] for r in reorder_idx])
                    
                    batch.reorder(reorder_idx)

                    # KM: Here we throw away some data to make sure that data can be evenly distributed across DP ranks
                    batch.truncate(8)
                    print ('batch after truncate', batch)
                    for k, v in batch.non_tensor_batch.items():
                        print (k, v.shape)
                    #print ('after rollout', batch, len(batch))
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    #self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)


                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_steps % self.config.trainer.val_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

        # perform validation after training
        if self.val_reward_fn is not None:
            if val_metrics is None or self.global_steps % self.config.trainer.val_freq != 0:
                val_metrics = self._validate()
                logger.log(data=val_metrics, step=self.global_steps)

            print(f"Final validation metrics: {val_metrics}.")

        self._save_checkpoint()
