"""
Directly queries the HoVer index with the query.
This is more meant for debugging and analysis than for end users.
"""

from argparse import ArgumentParser

import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from tqdm import tqdm
import ujson  # type: ignore
from matplotlib import pyplot as plt # type: ignore
import matplotlib

from goalbert.config import GoalBERTConfig
from transformers import AutoTokenizer # type: ignore

from goalbert.training.checkpoint import GCheckpoint
from goalbert.training.goalbert import logits_act_masks_to_distrs
from safetensors.torch import load_file


def load_collectionX(collection_path, dict_in_dict=False):
    """
    Same exact function as in Baleen, only reimported since Baleen fns don't seem to work.
    """
    print("Loading collection...")
    collectionX = {}

    with open(collection_path) as f:
        for line_idx, line in tqdm(enumerate(f)):
            line = ujson.loads(line)

            assert type(line["text"]) is list
            assert line["pid"] == line_idx, (line_idx, line)

            passage = [line["title"] + " | " + sentence for sentence in line["text"]]

            if dict_in_dict:
                collectionX[line_idx] = {}

            for idx, sentence in enumerate(passage):
                if dict_in_dict:
                    collectionX[line_idx][idx] = sentence
                else:
                    collectionX[(line_idx, idx)] = sentence

    return collectionX


def main():
    parser = ArgumentParser()
    parser.add_argument("--goalbert", action="store_true")
    parser.add_argument("--visualize-distrs", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num_masks", default=None, type=int)
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    with open("../sid_to_pid_sid.json", "r") as f:
        sid_to_pid_sid = ujson.load(f)
    collectionX = load_collectionX("../wiki.abstracts.2017/collection.json")
    with Run().context(RunConfig(nranks=1, experiment="wiki2017")):
        config = ColBERTConfig(
            root="./index",
            query_maxlen=64 if args.hops > 1 else 32,
        )
        searcher = Searcher(index="wiki2017.nbits=2", config=config)

        # Replace the checkpoint
        if args.goalbert:
            if args.checkpoint:
                colbert = searcher.checkpoint
                goalbert_cfg = GoalBERTConfig()
                goalbert_cfg.num_masks = args.num_masks or 0
                goalbert = GCheckpoint(colbert.name, colbert_config=config, goalbert_config=goalbert_cfg)
                goalbert.load_state_dict(load_file(args.checkpoint))
                searcher.checkpoint = goalbert
                del colbert
            else:
                colbert = searcher.checkpoint
                goalbert_cfg = GoalBERTConfig()
                goalbert_cfg.num_masks = args.num_masks or 0
                goalbert = GCheckpoint(colbert.name, colbert_config=config, goalbert_config=goalbert_cfg)
                goalbert.load_state_dict(colbert.state_dict())
                searcher.checkpoint = goalbert
                del colbert

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        matplotlib.use("TkAgg")
        while True:
            query = input("> ")
            context = []
            seen_facts = []
            print("Searching...")
            for i in range(args.hops):
                print("Hop", i + 1)
                print("Context:", context)
                context_processed = (" [SEP] ".join(context)) if context else None
                logits_all, act_masks_all, non_masks_all = searcher.compute_logits(
                    query,
                    context=context_processed,
                )
                action_distrs = logits_act_masks_to_distrs(logits_all, act_masks_all)
                idxs = [distr.sample().long() for distr in action_distrs]
                ranking = searcher.search(
                    query, k=10, context=context_processed, idxs=idxs
                )
                doc_ids = ranking[0]
                print("Indices:", idxs)

                if args.visualize_distrs:
                    input_ids, _ = goalbert.query_tokenizer.tensorize(
                        [query],
                        context=[context_processed] if context else None
                    )
                    toks = tokenizer.convert_ids_to_tokens(input_ids[0])
                    distrs = action_distrs[0]
                    n_plots = goalbert_cfg.num_masks or 1
                    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), tight_layout=True)
                    first_mask = toks.index("[MASK]")
                    if i > 0:
                        toks = toks[:first_mask] + toks[64:]
                    else:
                        toks = toks[:first_mask]
                    toks_len = len(toks) 
                    toks = toks + [""] * (512 - toks_len)
                    toks = [f"{t}:{i}" for i, t in enumerate(toks)]
                    for j, distr in zip(range(0, n_plots), distrs.probs):
                        masked_distr = distr.clone()
                        axs[j].bar(toks[:toks_len], masked_distr.tolist()[:toks_len])
                    plt.show()

                for doc_id in doc_ids:
                    pid_sid = tuple(sid_to_pid_sid[str(doc_id)])
                    sent = collectionX.get(pid_sid)
                    print(sent)

                fact_idx = doc_ids.pop(0)
                while fact_idx in seen_facts:
                    fact_idx = doc_ids.pop(0)
                seen_facts.append(fact_idx)
                pid_sid = tuple(sid_to_pid_sid[str(fact_idx)])
                sent = collectionX.get(pid_sid)
                context.append(sent)


if __name__ == "__main__":
    main()
