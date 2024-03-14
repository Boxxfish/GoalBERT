"""
Performs the training for GoalBERT.
"""

from goalbert.config import GoalBERTConfig

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import wandb
import random
from string import ascii_lowercase, digits
from pathlib import Path
import json
from safetensors.torch import save_model

from goalbert.training.ppo import train_ppo
from goalbert.training.rollout_buffer import RolloutBuffer


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        self.v_layer1 = nn.Linear(obs_shape[0], 64)
        self.v_layer2 = nn.Linear(64, 64)
        self.v_layer3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.v_layer1(x)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        self.a_layer1 = nn.Linear(obs_shape[0], 64)
        self.a_layer2 = nn.Linear(64, 64)
        self.a_layer3 = nn.Linear(64, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)

    def forward(self, x: torch.Tensor):
        x = self.a_layer1(x)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


def main():
    config = GoalBERTConfig.parse_args()

    env = gym.make_vec(
        "CartPole-v1", num_envs=config.training.num_envs, vectorization_mode="sync"
    )
    test_env = gym.make("CartPole-v1")

    obs_space = env.single_observation_space
    act_space = env.single_action_space
    assert isinstance(obs_space, Box), obs_space
    assert isinstance(act_space, Discrete), act_space

    v_net = ValueNet(obs_space.shape)
    p_net = PolicyNet(obs_space.shape, act_space.n)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=config.training.v_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=config.training.p_lr)

    buffer = RolloutBuffer(
        obs_space.shape,
        torch.Size((1,)),
        torch.Size((act_space.n,)),
        torch.int,
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

    obs = torch.Tensor(env.reset()[0])
    for iter_idx in tqdm(range(config.training.iterations), position=0):
        # Collect experience
        total_train_reward = 0.0
        with torch.no_grad():
            for _ in tqdm(
                range(config.training.train_steps), position=1, desc=f"Iter {iter_idx}"
            ):
                action_probs = p_net(obs)
                actions = Categorical(logits=action_probs).sample().numpy()
                obs_, rewards, dones, truncs, _ = env.step(actions)
                buffer.insert_step(
                    obs,
                    torch.from_numpy(actions).unsqueeze(-1),
                    action_probs,
                    rewards,
                    dones,
                    truncs,
                )
                obs = torch.from_numpy(obs_)
                total_train_reward += float(rewards.sum())
            buffer.insert_final_step(obs)

        # Train
        total_p_loss, total_v_loss = train_ppo(
            p_net,
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
        )
        buffer.clear()

        # Save checkpoint
        if iter_idx % config.save_every == 0:
            save_model(v_net, chkpt_path / f"v_net-{iter_idx}.safetensors")
            save_model(p_net, chkpt_path / f"p_net-{iter_idx}.safetensors")

        log_dict = {
            "avg_train_reward": total_train_reward / config.training.train_steps,
            "avg_v_loss": total_v_loss / config.training.train_iters,
            "avg_p_loss": total_p_loss / config.training.train_iters,
        }

        # Evaluate performance
        if iter_idx % config.eval_every == 0:
            with torch.no_grad():
                reward_total = 0
                entropy_total = 0.0
                total_steps = 0
                for _ in range(config.eval_runs):
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    for _ in range(config.max_eval_steps):
                        distr = Categorical(
                            logits=p_net(eval_obs.unsqueeze(0)).squeeze(0)
                        )
                        action = distr.sample().numpy()
                        obs_, reward, done, trunc, _ = test_env.step(action)
                        eval_obs = torch.Tensor(obs_)
                        reward_total += reward
                        entropy_total += distr.entropy().item()
                        if done or trunc:
                            break
                        total_steps += 1

                log_dict.update(
                    {
                        "avg_eval_episode_reward": reward_total / config.eval_runs,
                        "avg_eval_entropy": entropy_total / total_steps,
                    }
                )

        wandb.log(log_dict)


if __name__ == "__main__":
    main()
