"""
Evaluates the trained model on HoVer.
With this file, we measure R@25, to evaluate its ability to perform initial retrieval.
"""
from argparse import ArgumentParser
import copy
from distutils import config
import json
from colbert.data import Queries
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from typing import *
import torch
from safetensors.torch import load_file
from scipy import stats

from colbert.infra.config.config import ColBERTConfig, RunConfig
from colbert.infra.run import Run
from colbert.searcher import Searcher
from goalbert.config import GoalBERTConfig
from goalbert.training.checkpoint import GCheckpoint
from goalbert.training.env import FactIndex, QuestionIndex, fmt_context
from goalbert.training.goalbert import logits_act_masks_to_distrs
from goalbert.eval import metrics

def goalbert_rr(query: str, n_hops: int, fact_index: FactIndex, searcher: Searcher, gold_facts: Set[Tuple[int, int]]) -> List[float]:
    """
    Performs GoalBERT search, reporting the RR@25 across hops.
    """
    context_facts = []
    context = []
    gold_facts = copy.deepcopy(gold_facts)
    rr = []
    for _ in range(0, n_hops):
        # Compute action indices
        logits_all, act_masks_all, _ = searcher.compute_logits(
            query,
            context=fmt_context(context),
        )
        action_distrs = logits_act_masks_to_distrs(logits_all, act_masks_all)
        idxs = [distr.sample().long() for distr in action_distrs]

        # Perform ranking
        ranking = searcher.search(
            query,
            context=fmt_context(context),
            idxs=idxs,
            k=25,
        )
        doc_ids: List[int] = ranking[0]

        # Check RR@25 against our gold PIDs
        rank = 1
        seen_pids = set()
        fact_idx = None
        for idx in doc_ids:
            pid, _ = fact_index.sid_to_pid_sid[str(idx)]
            if pid in seen_pids:
                continue
            broke = False
            for gold_pid, gold_sid in gold_facts:
                if pid == gold_pid:
                    gold_facts.remove((gold_pid, gold_sid))
                    fact_idx = idx
                    rr.append(1 / rank)
                    broke = True
                    break
            if broke:
                break
            rank += 1

        # Use closest gold fact, if none were present in top 25, use a gold fact
        if fact_idx is None:
            fact_idx = gold_facts.pop()[0]
            rr.append(0)
        context.append(fact_index.get_fact_str(fact_idx))
        context_facts.append(fact_idx)
    return rr

labels = [
    "T MRR@25",
    "2-hop MRR@25",
    "3-hop MRR@25",
    "4-hop MRR@25",
]

def main():
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="dev")
    parser.add_argument("--compare-base", default=False, action="store_true")
    args = parser.parse_args()
    if args.compare_base:
        print("Changed:")
        changed = list(perform_test(args, False))
        json.dump(changed, open("changed_rr.json", "w"))
        # changed = json.load(open("changed.json", "r"))
        print("Base:")
        base = list(perform_test(args, True))
        json.dump(base, open("base_rr.json", "w"))
        
        for label, stat_c, stat_b in zip(labels, changed, base):
            pval = stats.ttest_ind(stat_c, stat_b).pvalue
            print(label, "P-Value:", pval)
    else:
        perform_test(args, False)

def perform_test(args, use_base: bool):
    data_dir = Path(args.datadir)
    qas_path = data_dir / "hover" / args.train_split / "qas.json"

    # Load GoalBERT
    fact_index = FactIndex(
        "../sid_to_pid_sid.json", "../wiki.abstracts.2017/collection.json"
    )
    q_index = QuestionIndex(qas_path)
    config = GoalBERTConfig(**json.load(open(Path(args.checkpoint).parent.parent / "config.json", "r")))
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config_ = ColBERTConfig(
            root="./index",
            query_maxlen=config.query_maxlen,
        )
        searcher = Searcher(index="wiki2017.nbits=2", config=config_)
        colbert = searcher.checkpoint
        goalbert = GCheckpoint(colbert.name, colbert_config=config_, goalbert_config=config)
        if use_base:
            goalbert.load_state_dict(colbert.state_dict())
        else:
            goalbert.load_state_dict(load_file(args.checkpoint))
        searcher.checkpoint = goalbert
        del colbert

    torch.manual_seed(100)
    rr_total_hops = [[] for _ in range(5)]
    total_hops = [0] * 5
    rr_total = []
    total_qas = 0
    for qa in tqdm(q_index.qas):
        try:
            gold_facts = set([(int(pid), int(sid)) for [pid, sid] in qa.support_facts])
            gold_psgs = set([pid for (pid, _) in gold_facts])
            rrs = goalbert_rr(qa.question, qa.num_hops, fact_index, searcher, gold_facts)
            rr_total_hops[qa.num_hops] += rrs
            rr_total += rrs
        except Exception as e:
            print("Exception (likely not enough masks):", e)
        total_qas += qa.num_hops
        total_hops[qa.num_hops] += qa.num_hops

    print("Total MRR@25:", sum(rr_total, 0.0) / total_qas)
    print(f"Total Count:", total_qas)
    for i in range(2, 5):
        print(f"{i}-hop Sentence F1:", sum(rr_total_hops[i], 0.0) / total_hops[i])
        print(f"{i}-hop Count:", total_hops[i])
    return (
        rr_total,

        rr_total_hops[2],
        
        rr_total_hops[3],
        
        rr_total_hops[4],
    )

if __name__ == "__main__":
    main()