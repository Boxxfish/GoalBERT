import copy
from typing import *

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from goalbert.training.goalbert import GoalBERT
from goalbert.training.rollout_buffer import RolloutBuffer


def train_ppo(
    p_net: GoalBERT,
    v_net: nn.Module,
    p_opt: torch.optim.Optimizer,
    v_opt: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    train_iters: int,
    train_batch_size: int,
    discount: float,
    lambda_: float,
    epsilon: float,
    gradient_steps: int = 1,
    train_p_net: bool = True,
) -> Tuple[float, float]:
    """
    Performs the PPO training loop. Returns a tuple of total policy loss and
    total value loss.

    Args:
        gradient_steps: Number of batches to step through before before
        adjusting weights.
    """
    p_net.train()
    v_net_frozen = copy.deepcopy(v_net)
    v_net.train()

    total_v_loss = 0.0
    total_p_loss = 0.0

    p_opt.zero_grad()
    v_opt.zero_grad()

    for _ in tqdm(range(train_iters), position=1):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net_frozen)
        for (
            i,
            (
                prev_input_ids,
                prev_attn_masks,
                actions,
                action_probs,
                returns,
                advantages,
                action_masks,
            ),
        ) in enumerate(batches):
            # Move batch to device if applicable
            prev_input_ids = prev_input_ids.to(device=device)
            prev_attn_masks = prev_attn_masks.to(device=device)
            actions = actions.to(device=device).long()
            action_probs = action_probs.to(device=device)
            returns = returns.to(device=device)
            advantages = advantages.to(device=device)
            action_masks = action_masks.to(device=device)
            
            # Train policy network
            if train_p_net:
                with torch.no_grad():
                    old_act_probs = torch.gather(action_probs, 2, actions[..., None]).squeeze(-1).log()
                    old_act_probs[action_masks[:, :, 0]] = 0
                    old_act_probs = old_act_probs.sum(-1)
                new_act_probs, _, _ = p_net.compute_probs(
                    prev_input_ids, prev_attn_masks
                )
                new_act_probs = torch.gather(new_act_probs, 2, actions[..., None]).squeeze(-1).log()
                new_act_probs[action_masks[:, :, 0]] = 0
                new_act_probs = new_act_probs.sum(-1)
                term1 = (new_act_probs - old_act_probs).exp() * advantages.squeeze()
                term2 = (1.0 + epsilon * advantages.squeeze().sign()) * advantages.squeeze()
                p_loss = -term1.min(term2).mean() / gradient_steps
                p_loss.backward()
                total_p_loss += p_loss.item()

            # Train value network
            diff = v_net(prev_input_ids.to("cuda:1"), prev_attn_masks.to("cuda:1")) - returns.to("cuda:1")
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            total_v_loss += v_loss.item()

            if (i + 1) % gradient_steps == 0:
                if train_p_net:
                    p_opt.step()
                    p_opt.zero_grad()
                v_opt.step()
                v_opt.zero_grad()

    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss)
