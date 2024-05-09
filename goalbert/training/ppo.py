import copy
from typing import *

import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from goalbert.training.goalbert import GoalBERT, logits_act_masks_to_masked_probs
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
    p_grad_clip: float = 99999.0,
    distill_coeff: float = 0.0,
    entropy_coeff: float = 0.0,
) -> Tuple[float, float, float, float, float]:
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
    total_v_norm = 0.0
    total_p_norm = 0.0
    total_entropy = 0.0

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
                non_masks,
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
            non_masks = non_masks.to(device=device)
            action_masks = action_masks.to(device=device)
            
            # Train policy network
            if train_p_net:
                with torch.no_grad():
                    selected_masks = torch.gather(action_masks, 2, actions[..., None]).squeeze(-1)
                    old_act_probs = action_probs.clone()
                    old_act_probs = torch.gather(old_act_probs, 2, actions[..., None]).squeeze(-1)
                    old_act_probs[selected_masks] = 1
                    old_act_probs = old_act_probs.log().sum(-1)
                    old_act_probs = old_act_probs.detach()
                new_act_logits, _, new_non_masks = p_net.compute_logits(
                    prev_input_ids, prev_attn_masks
                )

                new_act_probs = logits_act_masks_to_masked_probs(new_act_logits, action_masks)
                new_act_probs_orig = new_act_probs.clone()
                new_act_probs_orig = torch.gather(new_act_probs_orig, 2, actions[..., None]).squeeze(-1)
                new_act_probs_orig[selected_masks] = 0

                new_act_probs = torch.gather(new_act_probs, 2, actions[..., None]).squeeze(-1)
                new_act_probs[selected_masks] = 1
                new_act_probs = new_act_probs.log()

                # Compute entropy of new probs
                entropy = -(new_act_probs_orig * new_act_probs).sum(-1).mean()
                entropy_loss = entropy * entropy_coeff
                total_entropy += entropy.item()

                new_act_probs = new_act_probs.sum(-1)
                term1 = (new_act_probs - old_act_probs).exp() * advantages.squeeze()
                term2 = (1.0 + epsilon * advantages.squeeze().sign()) * advantages.squeeze()
                
                # Compute distillation loss
                non_masks_flat = non_masks.flatten(0, 1)
                new_non_masks_flat = new_non_masks.flatten(0, 1).to(device=device)
                distill_loss = (torch.diag(new_non_masks_flat @ non_masks_flat.T).sum() / (~action_masks[:, 0, :]).sum()) * distill_coeff

                # Compute diversity of [MASK]s
                div_loss = ((new_non_masks_flat @ new_non_masks_flat.T).sum() / (~action_masks[:, 0, :]).sum()) * 0.003

                p_loss = (-term1.min(term2).mean() + -distill_loss + -entropy_loss + div_loss) / gradient_steps
                
                p_loss.backward()
                total_p_loss += p_loss.item()

            # Train value network
            diff = v_net(prev_input_ids.to("cuda:1"), prev_attn_masks.to("cuda:1")) - returns.to("cuda:1")
            v_loss = (diff * diff).mean() / gradient_steps
            v_loss.backward()
            
            total_v_loss += v_loss.item()

            if (i + 1) % gradient_steps == 0:
                if train_p_net:
                    total_p_norm += grad_norm(p_net)
                    torch.nn.utils.clip_grad.clip_grad_norm_(p_net.parameters(), p_grad_clip)
                    p_opt.step()
                    p_opt.zero_grad()
                total_v_norm += grad_norm(v_net)
                v_opt.step()
                v_opt.zero_grad()

    p_net.eval()
    v_net.eval()

    return (total_p_loss, total_v_loss, total_p_norm, total_v_norm, total_entropy)

def grad_norm(net: nn.Module) -> float:
    total_norm = 0.0
    for p in net.parameters():
        if isinstance(p.grad, torch.Tensor):
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    norm = total_norm ** 0.5
    return norm