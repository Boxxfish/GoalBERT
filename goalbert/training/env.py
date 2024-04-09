from dataclasses import dataclass
import json
from typing import *
from typing import Any
import gymnasium as gym
import random

import torch

from colbert.searcher import Searcher
from goalbert.training.goalbert import GoalBERT
from query_colbert import load_collectionX


@dataclass
class QA:
    qid: int
    question: str
    support_pids: List[str]
    support_facts: List[List[int]]  # List of [pid, sid]s
    support_titles: List[str]
    num_hops: int
    label: int  # 1 if supported, 0 if unsupported
    uid: str
    hpqa_id: str


class FactIndex:
    def __init__(self, fact_to_pid_sid_path: str, collection_path: str):
        with open(fact_to_pid_sid_path, "r") as f:
            self.sid_to_pid_sid = json.load(f)
        self.pid_sid_to_sid = {}
        for k, v in self.sid_to_pid_sid.items():
            self.pid_sid_to_sid[tuple(v)] = k
        self.collectionX = load_collectionX(collection_path)

    def get_fact_str(self, fact_id: int) -> str:
        pid_sid = tuple(self.sid_to_pid_sid[fact_id])
        return self.collectionX.get(pid_sid)

    def from_pid_sid_to_fact_id(self, pid_sid: Tuple[int, int]) -> int:
        return self.pid_sid_to_sid[pid_sid]


class QuestionIndex:
    def __init__(self, qas_path: str):
        self.qas = []
        with open(qas_path, "r") as f:
            for line in f:
                self.qas.append(QA(**json.loads(line)))

    def random(self) -> QA:
        return random.choice(self.qas)


class SharedResources:
    def __init__(
        self, searcher: Searcher, fact_index: FactIndex, q_index: QuestionIndex
    ):
        self.searcher = searcher
        self.fact_index = fact_index
        self.q_index = q_index


State = Tuple[str, List[str]]


class GoalBERTEnv(gym.Env):
    """
    Gym environment for the multi-hop retrieval task.
    On each timestep, the agent selects a number of query and context tokens, forming a new query.
    """

    def __init__(
        self,
        goalbert: GoalBERT,
        shared: SharedResources,
        n_hops: int = 4,
        reward_depth: int = 100,  # Cutoff for reward
    ):
        super().__init__()
        self.n_hops = n_hops
        self.hops = 0
        self.context = []
        self.seen_facts = set()
        self.shared = shared
        self.goalbert = goalbert
        self.qa = None
        self.support_facts = []
        self.reward_depth = reward_depth

    @torch.no_grad()
    def step(
        self, action: List[int]
    ) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        assert self.qa, "Need QA"

        # Retrieve facts
        query = self.qa.question
        ranking = self.shared.searcher.search(
            query,
            context=(" [SEP] ".join(self.context)) if self.context else None,
            indices=action,
            k=self.reward_depth,
        )
        doc_ids: List[int] = ranking[0]

        # Compute reward
        ranks = []
        for fact_idx in self.support_facts:
            if fact_idx in doc_ids and fact_idx not in self.seen_facts:
                ranks.append(doc_ids.index(fact_idx))
        closest_rank = min(ranks, default=None)
        reward = (
            0.0
            if closest_rank is None
            else ((self.reward_depth - closest_rank) / self.reward_depth)
        )

        # Greedily select fact
        fact_idx = doc_ids[0]
        self.context.append(self.shared.fact_index.get_fact_str(fact_idx))
        self.seen_facts.add(fact_idx)

        self.hops += 1
        done = self.hops == self.n_hops

        return (query, self.context), reward, done, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[State, Dict[str, Any]]:
        if seed is None:
            random.seed(seed)
        self.hops = 0
        self.qa = self.shared.q_index.random()
        self.context = []
        self.seen_facts = set()
        self.support_facts = [
            self.shared.fact_index.from_pid_sid_to_fact_id(tuple(pid_qid))
            for pid_qid in self.qa.support_facts
        ]
        return (self.qa.question, None), {}
