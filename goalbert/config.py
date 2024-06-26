"""
Configuration classes.
"""

from argparse import ArgumentParser
from typing_extensions import Self
import torch
from typing import *


class BaseConfig:
    def __init__(self, **kwargs):
        for [key, val] in self.__class__.__dict__.items():
            if (
                not (isinstance(val, Callable) or isinstance(val, classmethod))
                and key[:2] != "__"
            ):
                self.__setattr__(key, val)
        for key in kwargs:
            self.set(key, kwargs[key])

    def set(self, key: str, val: Any):
        """
        Sets the value of this config or a child config's option.
        """
        assert self.exists(key)
        keys = self.__dict__.keys()
        if key in keys:
            self.__setattr__(key, val)
        else:
            for attr_name in keys:
                attr = self.__getattribute__(attr_name)
                if isinstance(attr, BaseConfig):
                    if attr.exists(key):
                        attr.set(key, val)

    def exists(self, key: str) -> bool:
        """
        Returns true if this config or a child config contains this option.
        """
        keys = self.__dict__.keys()
        if key in keys:
            return True
        for attr_name in keys:
            attr = self.__getattribute__(attr_name)
            if isinstance(attr, BaseConfig):
                if attr.exists(key):
                    return True
        return False

    def flat_dict(self) -> Dict[str, Any]:
        """
        Returns a flat dict of all config options.
        """
        d = {}
        for attr_name in self.__dict__.keys():
            attr = self.__getattribute__(attr_name)
            if isinstance(attr, BaseConfig):
                d.update(attr.flat_dict())
            else:
                d[attr_name] = attr
        return d

    @classmethod
    def parse_args(cls) -> Self:
        """
        Parses arguments and returns the filled config.
        Underscores are replaced with dashes.
        """
        parser = ArgumentParser()
        c = cls()
        for [key, val] in c.flat_dict().items():
            parser.add_argument(f"--{key}", type=type(val), default=val)
        d = cls(**parser.parse_args().__dict__)
        return d


class TrainingConfig(BaseConfig):
    num_envs: int = (
        32  # Number of environments to step through at once during sampling.
    )
    train_steps: int = (
        16  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
    )
    iterations: int = 1000  # Number of sample/train iterations.
    train_iters: int = 1  # Number of passes over the samples collected.
    train_batch_size: int = 8  # Minibatch size while training models.
    discount: float = 1.0  # Discount factor applied to rewards.
    lambda_: float = 0.99  # Lambda for GAE.
    epsilon: float = 0.2  # Epsilon for importance sample clipping.
    v_lr: float = 0.0001  # Learning rate of the value net.
    p_lr: float = 0.000005  # Learning rate of the policy net.
    distill_coeff: float = 3.0  # Cosine maximization applied to non-MASKs.
    gradient_steps: int = 8  # Number of gradient steps before optimizing.
    max_input_ids: int = 512  # Maxmimum # of input IDs.
    value_warmup: int = 0 # Number of iterations to train just the value network at the beginning, to give the baseline a head start.
    p_grad_clip: float = 1000.0 # The gradient clipping applied to the policy network.
    entropy_coeff: float = 0.003

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GoalBERTConfig(BaseConfig):
    training: TrainingConfig = TrainingConfig()
    device: str = "cuda"
    exp_name: str = "exp"
    exp_root: str = "./experiments"
    save_every: int = 10
    eval_every: int = 10
    eval_runs: int = 100
    v_net_finetune: str = ""
    max_hops: int = 4
    num_masks: int = 0 # Number of [MASK]s to use. If 0, we use all [MASK]s.
    query_maxlen: int = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
