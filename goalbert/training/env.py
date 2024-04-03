import json
from typing import *
from typing import Any
import gymnasium as gym
import random

from colbert.searcher import Searcher
from query_colbert import load_collectionX


class FactIndex:
    def __init__(self, fact_to_pid_sid_path: str, collection_path: str) -> None:
        with open(fact_to_pid_sid_path, "r") as f:
            self.sid_to_pid_sid = json.load(f)
        self.collectionX = load_collectionX(collection_path)

    def get_fact_str(self, fact_id: int) -> str:
        pid_sid = tuple(self.sid_to_pid_sid[fact_id])
        return self.collectionX.get(pid_sid)


class QuestionIndex:
    def __init__(self, qas_path: str) -> None:
        qas = []
        with open(qas_path, "r") as f:
            for line in f:
                qas.append(json.load(line))

    def random(self) -> Dict[str, Any]:
        random.choice(self.qas)


class SharedResources:
    def __init__(
        self, searcher: Searcher, fact_index: FactIndex, q_index: QuestionIndex
    ):
        self.searcher = searcher
        self.fact_index = fact_index
        self.q_index = q_index


class GoalBERTEnv(gym.Env):
    """
    Gym environment for the multi-hop retrieval task.
    On each timestep, the agent selects a number of query and context tokens, forming a new query.
    """

    def __init__(self, shared: SharedResources, n_hops: int = 4):
        super().__init__()
        self.n_hops = n_hops
        self.query = None
        self.context = []
        self.shared = shared

    def step(self, action: List[int]) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        assert isinstance(self.query, str)
        return super().step(action)

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)
