# Copyright 2022 The HuggingFace Team
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch

from ..utils import torch_functional as VF
import pdb

if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None: ...


class AdaptiveKLController(KLController):
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController(KLController):
    """Fixed KL controller."""

    def __init__(self, init_kl_coef: float):
        self.value = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_traj_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, traj_index: torch.Tensor, step_index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    traj_id2score = defaultdict(list)
    traj2id = {}
    id2mean, id2std = {}, {}
    print('scores', scores)
    #print ('index', index)
    #print ('traj_index', traj_index)

    bsz = scores.shape[0]
    for i in range(bsz):
        traj_id2score[traj_index[i]].append(scores[i].item())
        traj2id[traj_index[i]] = index[i]
        # print ('traj_id2score[traj_index[i]]', traj_id2score[traj_index[i]])
        # print ('traj2id[traj_index[i]]', traj2id[traj_index[i]])

    for k, v in traj_id2score.items():
        #print ('k', k, 'v', v)
        #print ('traj2id[k]', traj2id[k])
        id2score[traj2id[k]].append(max(v))
    
    print ('traj2id', traj2id)
    print ('id2score', id2score )

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
        else:
            raise ValueError(f"no score in prompt index: {idx}")
    #print ('id2mean', id2mean)
    #print ('id2std', id2std)
    #breakpoint()
    
    for i in range(bsz):
        scores[i] = (max(traj_id2score[traj_index[i]]) - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    print ('traj_id2score', traj_id2score)
    #print (i, traj_index[i], traj_id2score[traj_index[i]], max(traj_id2score[traj_index[i]]), id2mean[index[i]], (id2std[index[i]] + epsilon))
    #print (scores[-1])
    #print('scores', scores)
    #breakpoint()
    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores

@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        else:
            raise ValueError(f"no score in prompt index: {idx}.")

    for i in range(bsz):
        response_num = len(id2score[index[i]])
        if response_num > 1:
            scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                response_num - 1
            )

    scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return scores, scores


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * eos_mask[:, t]

    advantages = VF.masked_whiten(returns, eos_mask)
    advantages = advantages * eos_mask
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, eos_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    # scores = token_level_rewards.sum(dim=-1)
    returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return advantages, returns


def compute_rewards(
    token_level_scores: torch.Tensor, old_log_prob: torch.Tensor, ref_log_prob: torch.Tensor, kl_ratio: float
) -> torch.Tensor:
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
) -> Tuple[torch.Tensor, float, float]:
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = VF.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = VF.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = VF.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = VF.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = VF.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor, cliprange_value: float
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
    """
    vpredclipped = VF.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = VF.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob: torch.Tensor
        ref_logprob: torch.Tensor

    Returns:
        kl_div: torch.Tensor
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError

if __name__ == '__main__':
    traj_index = np.array(['33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       '4a68902b-8d18-4c6c-a09f-e79c726a8a58',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       '4a68902b-8d18-4c6c-a09f-e79c726a8a58',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       '4a68902b-8d18-4c6c-a09f-e79c726a8a58',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86',
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c' ,
       '6e54d83d-658d-4687-a72d-bd082128d341',
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86' ,
       '33de1738-697b-4176-8c9b-6bb29d68def1',
       'f6927de9-c212-4d01-ab8d-2c15f9def350',
       'd08bfd8e-5bfc-4dbc-b45b-ef7afbd9da8b',
       '73dd9da1-484c-433e-84d9-7e7e291ecc1d',
       'e050fe70-8783-4eee-87bb-be4b3d1f6b36',
       '3fd9b5a0-0d30-4f06-9db4-720ee566593d',
       '4d053e3e-26a8-4622-a6a0-096a2719954c',
       '6e54d83d-658d-4687-a72d-bd082128d341' ,
       'c2049444-fe5e-4768-aa4c-4a7698859f4f',
       'a56b22a9-9eff-4607-adca-bdec51463903',
       '0c02d4d6-bc2d-4b90-ba9c-3c70c459fbc2',
       'da5ea29e-b182-413c-8d2d-79b581f8278e',
       '40e94dda-640b-4af6-9a83-20cc8a43cb01',
       'a95607be-57b8-4af5-8623-62905982dc27',
       'ca84d4f5-3ceb-46ce-b481-f2d4e5d1dd86' ], dtype=object)

    scores = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 1.])

    index = np.array([11., 11., 11., 11., 11., 11., 11., 11.,  5.,  5.,  5.,  5.,  5.,
        5.,  5.,  5., 11., 11., 11., 11., 11., 11., 11., 11.,  5.,  5.,
        5.,  5.,  5.,  5.,  5.,  5., 11., 11., 11., 11., 11., 11., 11.,
       11.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5., 11., 11., 11., 11.,
       11., 11., 11., 11.,  5.,  5.,  5.,  5.,  5.,  5.,  5., 11., 11.,
       11., 11., 11., 11., 11., 11.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
       11., 11., 11., 11., 11., 11., 11., 11.,  5.,  5.,  5.,  5.,  5.,
        5.,  5., 11., 11., 11., 11., 11., 11., 11., 11.,  5.,  5.,  5.,
        5.,  5.,  5.,  5., 11., 11., 11., 11., 11., 11., 11., 11.,  5.,
        5.,  5.,  5.,  5.,  5.,  5., 11., 11., 11., 11., 11., 11., 11.,
       11.,  5.,  5.,  5.,  5.,  5.,  5.,  5., 11., 11., 11., 11., 11.,
       11., 11., 11.,  5.,  5.,  5.,  5.,  5.,  5.,  5.], dtype=np.float32)

    compute_grpo_traj_outcome_advantage(None, scores, None, index, traj_index, None)