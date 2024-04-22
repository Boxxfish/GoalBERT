"""
Performs the training for GoalBERT.
"""

from colbert.infra.config.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
from colbert.searcher import Searcher
from goalbert.config import GoalBERTConfig

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
import wandb
import random
from string import ascii_lowercase, digits
from pathlib import Path
import json
from safetensors.torch import save_model, load_file
from transformers import DistilBertModel

from goalbert.training.checkpoint import GCheckpoint
from goalbert.training.env import (
    FactIndex,
    GoalBERTEnv,
    QuestionIndex,
    SharedResources,
    fmt_context,
)
from goalbert.training.goalbert import MAX_MASKS, probs_act_masks_to_distrs
from goalbert.training.ppo import train_ppo
from goalbert.training.rollout_buffer import RolloutBuffer


class ValueNet(nn.Module):
    def __init__(self, from_pretrained: str = "distilbert/distilbert-base-uncased"):
        nn.Module.__init__(self)
        self.bert = DistilBertModel.from_pretrained(from_pretrained)
        self.linear = nn.Linear(768, 1)

    def forward(
        self, input_ids: torch.Tensor, attn_masks: torch.Tensor
    ) -> torch.Tensor:
        # Similar to ColBERT forward pass
        x = self.bert(input_ids, attention_mask=attn_masks)[0][:, 0]  # Use only [CLS]
        x = self.linear(x)
        return x


def main():
    config = GoalBERTConfig.parse_args()

    fact_index = FactIndex(
        "../sid_to_pid_sid.json", "../wiki.abstracts.2017/collection.json"
    )
    q_index = QuestionIndex("../hover/train/qas.json")
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config_ = ColBERTConfig(
            root="./index",
            query_maxlen=64,
        )
        searcher = Searcher(index="wiki2017.nbits=2", config=config_)
        colbert = searcher.checkpoint
        goalbert = GCheckpoint(colbert.name, colbert_config=config_)
        goalbert.load_state_dict(colbert.state_dict())
        searcher.checkpoint = goalbert
        del colbert
    shared = SharedResources(searcher, fact_index, q_index)
    env = gym.vector.SyncVectorEnv(
        [
            lambda: GoalBERTEnv(goalbert, shared=shared)
            for _ in range(config.training.num_envs)
        ]
    )
    test_env = GoalBERTEnv(goalbert, shared=shared)

    v_net = ValueNet()
    if config.v_net_finetune:
        v_net.load_state_dict(load_file(config.v_net_finetune))
    v_net.to("cuda:1")
    v_opt = torch.optim.Adam(v_net.parameters(), lr=config.training.v_lr)
    p_opt = torch.optim.Adam(goalbert.parameters(), lr=config.training.p_lr)

    buffer = RolloutBuffer(
        MAX_MASKS,
        config.training.max_input_ids,
        config.training.num_envs,
        config.training.train_steps,
    )

    # Set up experiment directory
    exp_name_full = (
        config.exp_name
        + "-"
        + "".join([random.choice(ascii_lowercase + digits) for _ in range(8)])
    )
    wandb.init(project="goalbert", name=exp_name_full, config=config.flat_dict())
    exp_dir = Path(config.exp_root) / exp_name_full
    assert not exp_dir.exists(), "Experiment directory exists, aborting"
    exp_dir.mkdir()

    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.flat_dict(), f)

    chkpt_path = exp_dir / "checkpoints"
    chkpt_path.mkdir()

    obs = env.reset()[0]
    for iter_idx in tqdm(range(config.training.iterations), position=0):
        # Collect experience
        total_train_reward = 0.0
        with torch.no_grad():
            for _ in tqdm(
                range(config.training.train_steps), position=1, desc=f"Iter {iter_idx}"
            ):
                probs_all, act_masks_all, _ = searcher.compute_probs(
                    list(obs[0]),
                    context=[fmt_context(ctx) if ctx else "" for ctx in obs[1]],
                )
                action_distrs = probs_act_masks_to_distrs(probs_all, act_masks_all)
                input_ids, attention_mask = goalbert.query_tokenizer.tensorize(
                    list(obs[0]),
                    context=[fmt_context(ctx) if ctx else "" for ctx in obs[1]],
                )
                actions = [
                    action_distr.sample().cpu().tolist()
                    for action_distr in action_distrs
                ]
                obs_, rewards, dones, truncs, _ = env.step(actions)

                input_ids_padded = torch.zeros((config.training.num_envs, config.training.max_input_ids))
                input_ids_padded[:, : input_ids.shape[1]] = input_ids

                attn_masks_padded = torch.zeros((config.training.num_envs, config.training.max_input_ids))
                attn_masks_padded[:, : attention_mask.shape[1]] = attention_mask

                buffer.insert_step(
                    input_ids_padded,
                    attn_masks_padded,
                    torch.tensor(
                        [q_acts + [0] * (MAX_MASKS - len(q_acts)) for q_acts in actions]
                    ),
                    probs_all,
                    rewards,
                    dones,
                    truncs,
                    act_masks_all.to(torch.bool)
                )
                obs = obs_
                total_train_reward += float(rewards.sum())
            input_ids, attention_mask = goalbert.query_tokenizer.tensorize(
                list(obs[0]),
                context=[fmt_context(ctx) if ctx else "" for ctx in obs[1]],
            )
            input_ids_padded = torch.zeros((config.training.num_envs, config.training.max_input_ids))
            input_ids_padded[:, : input_ids.shape[1]] = input_ids

            attn_masks_padded = torch.zeros((config.training.num_envs, config.training.max_input_ids))
            attn_masks_padded[:, : attention_mask.shape[1]] = attention_mask
            buffer.insert_final_step(input_ids_padded, attn_masks_padded)

        # Train
        total_p_loss, total_v_loss, total_p_norm, total_v_norm = train_ppo(
            goalbert,
            v_net,
            p_opt,
            v_opt,
            buffer,
            torch.device(config.device),
            config.training.train_iters,
            config.training.train_batch_size,
            config.training.discount,
            config.training.lambda_,
            config.training.epsilon,
            gradient_steps=config.training.gradient_steps,
            train_p_net=iter_idx >= config.training.value_warmup
        )
        buffer.clear()

        # Save checkpoint
        if iter_idx % config.save_every == 0:
            save_model(v_net, chkpt_path / f"v_net-{iter_idx}.safetensors")
            save_model(goalbert, chkpt_path / f"goalbert-{iter_idx}.safetensors")

        log_dict = {
            "avg_train_reward": total_train_reward / config.training.train_steps,
            "avg_v_loss": total_v_loss / config.training.train_iters,
            "avg_p_loss": total_p_loss / config.training.train_iters,
            "avg_v_norm": total_v_norm / config.training.train_iters,
            "avg_p_norm": total_p_norm / config.training.train_iters,
        }

        # Evaluate performance
        if iter_idx % config.eval_every == 0:
            with torch.no_grad():
                em_total = 0.0
                f1_total = 0.0
                p_total = 0.0
                r_total = 0.0
                for _ in range(config.eval_runs):
                    eval_obs = test_env.reset()[0]
                    while True:
                        probs_all, act_masks_all, _ = searcher.compute_probs(
                            [eval_obs[0]],
                            context=[fmt_context(eval_obs[1])] if eval_obs[1] else None,
                        )
                        action_distrs = probs_act_masks_to_distrs(probs_all, act_masks_all)
                        input_ids, attention_mask = goalbert.query_tokenizer.tensorize(
                            [eval_obs[0]],
                            context=[fmt_context(eval_obs[1])] if eval_obs[1] else None,
                        )
                        actions = [
                            action_distr.sample().cpu().tolist()
                            for action_distr in action_distrs
                        ]
                        eval_obs, _, done, _, _ = test_env.step(actions[0])

                        if done:
                            em_total += test_env.compute_em()
                            f1_total += test_env.compute_f1()
                            p_total += test_env.compute_p()
                            r_total += test_env.compute_r()
                            break

                log_dict.update(
                    {
                        "avg_eval_em": em_total / config.eval_runs,
                        "avg_eval_f1": f1_total / config.eval_runs,
                        "avg_eval_p": p_total / config.eval_runs,
                        "avg_eval_r": r_total / config.eval_runs,
                    }
                )

        wandb.log(log_dict)


if __name__ == "__main__":
    main()
