"""
Evaluates the trained model on HoVer.
"""
from argparse import ArgumentParser
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

def goalbert_search(query: str, n_hops: int, fact_index: FactIndex, searcher: Searcher) -> Set[Tuple[int, int]]:
    """
    Given a query and number of hops, returns a set of (pid, sid) pairs.
    """
    seen_facts: Set[int] = set()
    context_facts = []
    context = []
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
            k=10,
        )
        doc_ids: List[int] = ranking[0]

        # Greedily select fact, skipping ones we already chose.
        fact_idx = doc_ids.pop(0)
        while fact_idx in seen_facts:
            fact_idx = doc_ids.pop(0)
        context.append(fact_index.get_fact_str(fact_idx))
        context_facts.append(fact_idx)
    return set([tuple(fact_index.sid_to_pid_sid[str(sid)]) for sid in context_facts])

labels = [
    "T Sentence F1",
    "T Passage F1",
    "T Sentence EM",
    "T Passage EM",
    "2 Sentence F1",
    "2 Passage F1",
    "2 Sentence EM",
    "2 Passage EM",
    "3 Sentence F1",
    "3 Passage F1",
    "3 Sentence EM",
    "3 Passage EM",
    "4 Sentence F1",
    "4 Passage F1",
    "4 Sentence EM",
    "4 Passage EM",
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
        changed = list(perform_test(args))
        print("Base:")
        base = list(perform_test(args))
        
        for label, stat_c, stat_b in zip(labels, changed, base):
            pval = stats.ttest_ind(stat_c, stat_b).pvalue
            print(label, "P-Value:", pval)
    else:
        perform_test(args)

def perform_test(args):
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
        if args.checkpoint:
            goalbert.load_state_dict(load_file(args.checkpoint))
        else:
            goalbert.load_state_dict(colbert.state_dict())
        searcher.checkpoint = goalbert
        del colbert

    torch.manual_seed(100)
    s_f1_total_hops = [[] for _ in range(5)]
    p_f1_total_hops = [[] for _ in range(5)]
    s_em_total_hops = [[] for _ in range(5)]
    p_em_total_hops = [[] for _ in range(5)]
    total_hops = [0] * 5
    s_f1_total = []
    p_f1_total = []
    s_em_total = []
    p_em_total = []
    total_qas = 0
    for qa in tqdm(q_index.qas):
        try:
            gold_facts = set([(int(pid), int(sid)) for [pid, sid] in qa.support_facts])
            gold_psgs = set([pid for (pid, _) in gold_facts])
            pred_facts = goalbert_search(qa.question, qa.num_hops, fact_index, searcher)
            pred_psgs = set([pid for (pid, _) in pred_facts])
            s_f1 = metrics.f1(pred_facts, gold_facts)
            p_f1 = metrics.f1(pred_psgs, gold_psgs)
            s_em = metrics.em(pred_facts, gold_facts)
            p_em = metrics.em(pred_psgs, gold_psgs)
            s_f1_total_hops[qa.num_hops] += [s_f1]
            p_f1_total_hops[qa.num_hops] += [p_f1]
            s_em_total_hops[qa.num_hops] += [s_em]
            p_em_total_hops[qa.num_hops] += [p_em]
            s_f1_total += [s_f1]
            p_f1_total += [p_f1]
            s_em_total += [s_em]
            p_em_total += [p_em]
        except Exception as e:
            print("Exception (likely not enough masks):", e)
        total_qas += 1
        total_hops[qa.num_hops] += 1

    print("Total Sentence F1:", sum(s_f1_total, 0.0) / total_qas)
    print("Total Passage F1:", sum(p_f1_total, 0.0) / total_qas)
    print("Total Sentence EM:", sum(s_em_total, 0.0) / total_qas)
    print("Total Passage EM:", sum(p_em_total, 0.0) / total_qas)
    print(f"Total Count:", total_qas)
    for i in range(2, 5):
        print(f"{i}-hop Sentence F1:", sum(s_f1_total_hops[i], 0.0) / total_hops[i])
        print(f"{i}-hop Passage F1:", sum(p_f1_total_hops[i], 0.0) / total_hops[i])
        print(f"{i}-hop Sentence EM:", sum(s_em_total_hops[i], 0.0) / total_hops[i])
        print(f"{i}-hop Passage EM:", sum(p_em_total_hops[i], 0.0) / total_hops[i])
        print(f"{i}-hop Count:", total_hops[i])
    return (
        s_f1_total,
        p_f1_total,
        s_em_total,
        p_em_total,

        s_f1_total_hops[2],
        p_f1_total_hops[2],
        s_em_total_hops[2],
        p_em_total_hops[2],
        
        s_f1_total_hops[3],
        p_f1_total_hops[3],
        s_em_total_hops[3],
        p_em_total_hops[3],
        
        s_f1_total_hops[4],
        p_f1_total_hops[4],
        s_em_total_hops[4],
        p_em_total_hops[4],
        )

if __name__ == "__main__":
    main()