from dataclasses import dataclass
import json
from typing import *
from typing import Any
import gymnasium as gym
import random

import torch

from colbert.infra.config.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
from colbert.searcher import Searcher
from goalbert.training.checkpoint import GCheckpoint
from goalbert.training.goalbert import (
    MAX_ACTIONS,
    MAX_MASKS,
    GoalBERT,
    logits_act_masks_to_distrs,
)
from goalbert.eval import metrics
from query_colbert import load_collectionX


@dataclass
class QA:
    qid: int
    question: str
    support_pids: List[str]
    support_facts: List[List[str]]  # List of [pid, sid]s
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
        pid_sid = tuple(self.sid_to_pid_sid[str(fact_id)])
        return self.collectionX.get(pid_sid)

    def from_pid_sid_to_fact_id(self, pid_sid: Tuple[int, int]) -> int:
        return self.pid_sid_to_sid[pid_sid]

    def valid_pid_sid(self, pid_sid: Tuple[int, int]) -> bool:
        """
        Checks if this exists, needed since the dataset sometimes requests nonexistent ones.
        """
        return pid_sid in self.pid_sid_to_sid


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
        reward_depth: int = 100,  # Cutoff for reward
        max_hops: int = 4,  # Maximum number of hops to perform (impacts data)
    ):
        super().__init__()
        self.hops = 0
        self.context = []
        self.context_facts = []
        self.seen_facts = set()
        self.shared = shared
        self.goalbert = goalbert
        self.qa = None
        self.max_hops = max_hops
        self.support_facts = []
        self.reward_depth = reward_depth
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Text(MAX_MASKS), gym.spaces.Sequence(gym.spaces.Text(MAX_ACTIONS))]
        )
        self.action_space = gym.spaces.MultiDiscrete([MAX_MASKS, MAX_ACTIONS])

    @torch.no_grad()
    def step(
        self, action: List[int]
    ) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        assert self.qa, "Need QA"

        # Retrieve facts
        query = self.qa.question
        ranking = self.shared.searcher.search(
            query,
            context=fmt_context(self.context),
            idxs=[action],
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

        # Greedily select fact, skipping ones we already chose.
        # If the closest fact was very close to being selected, add it to the seen facts so we don't select it again.
        fact_idx = doc_ids.pop(0)
        while fact_idx in self.seen_facts:
            fact_idx = doc_ids.pop(0)
        self.context.append(self.shared.fact_index.get_fact_str(fact_idx))
        self.context_facts.append(fact_idx)
        if closest_rank is not None and closest_rank < 10:
            self.seen_facts.add(fact_idx)

        self.hops += 1
        done = self.hops == self.qa.num_hops

        return (query, self.context), reward, done, False, {}

    def reset(self, *, seed=None, options=None) -> Tuple[State, Dict[str, Any]]:
        if seed is None:
            random.seed(seed)
        self.hops = 0
        self.qa = self.shared.q_index.random()
        _, attention_mask = self.shared.searcher.checkpoint.query_tokenizer.tensorize(
            [self.qa.question]
        )

        # Ensure we can retrieve all of our facts and that the query < (query maxlen - 4)
        while not all(
            [
                self.shared.fact_index.valid_pid_sid(tuple(pid_qid))
                for pid_qid in self.qa.support_facts
            ]
        ) or (attention_mask.sum().item() >= MAX_MASKS - 4) or self.qa.num_hops > self.max_hops:
            self.qa = self.shared.q_index.random()
            _, attention_mask = (
                self.shared.searcher.checkpoint.query_tokenizer.tensorize(
                    [self.qa.question]
                )
            )

        self.context = []
        self.context_facts = []
        self.seen_facts = set()
        self.support_facts = [
            int(self.shared.fact_index.from_pid_sid_to_fact_id(tuple(pid_qid)))
            for pid_qid in self.qa.support_facts
        ]
        return (self.qa.question, None), {}
    
    def compute_em(self) -> float:
        return metrics.em(set(self.support_facts), set(self.context_facts))
    
    def compute_f1(self) -> float:
        return metrics.f1(set(self.support_facts), set(self.context_facts))
    
    def compute_p(self) -> float:
        return metrics.precision(set(self.support_facts), set(self.context_facts))
    
    def compute_r(self) -> float:
        return metrics.recall(set(self.support_facts), set(self.context_facts))


def fmt_context(ctx: List[str]) -> Optional[str]:
    """
    Formats a list of facts into the expected format.
    """
    return (" [SEP] ".join(ctx)) if ctx else None


# Testing the environment
def test():
    # Initialize environment
    fact_index = FactIndex(
        "../sid_to_pid_sid.json", "../wiki.abstracts.2017/collection.json"
    )
    q_index = QuestionIndex("../hover/train/qas.json")
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config = ColBERTConfig(
            root="./index",
            query_maxlen=64,
        )
        searcher = Searcher(index="wiki2017.nbits=2", config=config)
        colbert = searcher.checkpoint
        goalbert = GCheckpoint(colbert.name, colbert_config=config)
        goalbert.load_state_dict(colbert.state_dict())
        searcher.checkpoint = goalbert
        del colbert
    shared = SharedResources(searcher, fact_index, q_index)
    env = GoalBERTEnv(goalbert, shared, reward_depth=100)

    obs, _ = env.reset()
    for i in range(10):
        logits_all, act_masks_all, _ = searcher.compute_logits(
            obs[0], context=fmt_context(obs[1])
        )
        action_distr = logits_act_masks_to_distrs(logits_all, act_masks_all)[0]
        actions = action_distr.sample().tolist()
        obs, reward, done, trunc, _ = env.step(actions)
        print("Step:", i)
        print("Query:", obs[0])
        print("Context:", obs[1])
        print("Reward:", reward)
        print("Done:", reward)
        if done or trunc:
            obs, _ = env.reset()
