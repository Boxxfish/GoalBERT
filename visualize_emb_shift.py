"""
Visualizes embedding shifts
"""
import shutil
from matplotlib import pyplot as plt # type: ignore
import matplotlib
from argparse import ArgumentParser
from sklearn.decomposition import PCA # type: ignore
from pathlib import Path
import json
import torch
import numpy as np
from safetensors.torch import load_file
from typing import *
import os
from textwrap import wrap

from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from goalbert.config import GoalBERTConfig
from goalbert.training.checkpoint import GCheckpoint

def query_to_embs(query: str, checkpoint: GCheckpoint, context) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = checkpoint.query_tokenizer.tensorize(
        [query], context=None if context is None else [context],
    )
    embs = checkpoint.query(input_ids, attention_mask)
    return embs[0].cpu().numpy(), input_ids[0].cpu().numpy()

def load_checkpoint(exp_path: Path, idx: int) -> Checkpoint:
    cfg_path = exp_path / "config.json"
    g_config = GoalBERTConfig(**json.load(open(cfg_path, "r")))
    c_config = ColBERTConfig(
        root="./index",
        query_maxlen=g_config.query_maxlen,
    )
    checkpoint = Checkpoint("../colbertv2.0", colbert_config=c_config)
    chkpt_path = exp_path / "checkpoints" / f"goalbert-{idx}.safetensors"
    checkpoint.load_state_dict(load_file(chkpt_path))
    return checkpoint

def main():
    matplotlib.use("TkAgg")
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./experiments")
    parser.add_argument("--out", type=str, default="./emb_shift")
    parser.add_argument("--name", type=str)
    parser.add_argument("--num-masks", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--query", type=str, default="Before I Go to Sleep stars an Australian actress, producer and occasional singer.")
    parser.add_argument("--context", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Load first model
    exp_path = Path(args.root) / args.name
    checkpoint = load_checkpoint(exp_path, 0)

    # Train PCA on first query
    embs, input_ids = query_to_embs(args.query, checkpoint, args.context) # Shape: (num_embs, emb_dim)
    pca = PCA(2)
    pca.fit(embs)

    # Create plots of [MASK] positions
    MASK = 103
    chkpt_idx = 0
    img_idxs = []
    while (exp_path / "checkpoints" / f"goalbert-{chkpt_idx}.safetensors").exists():
        checkpoint = load_checkpoint(exp_path, chkpt_idx)
        embs, input_ids = query_to_embs(args.query, checkpoint, args.context) # Shape: (num_embs, emb_dim)
        first_mask = (input_ids == MASK).argmax(0)
        first_ctx = 64
        xformed_embs = pca.transform(embs)
        nline = "\n"
        plt.title(f"\"{nline.join(wrap(args.query, 40))}\"\nIteration {chkpt_idx}")
        plt.scatter(xformed_embs[:first_mask, 0], xformed_embs[:first_mask, 1], c="black")
        if args.context:
            plt.scatter(xformed_embs[first_ctx:, 0], xformed_embs[first_ctx:, 1], c="black")
        plt.scatter(xformed_embs[first_mask:first_mask + args.num_masks, 0], xformed_embs[first_mask:first_mask + args.num_masks, 1], c="red")
        
        ax = plt.gca()
        ax.set_xlim([-0.9, 0.9])
        ax.set_ylim([-0.9, 0.9])

        toks = checkpoint.query_tokenizer.tok.convert_ids_to_tokens(input_ids)
        for i in range(0, first_mask):
            plt.annotate(toks[i], xformed_embs[i])
        if args.context:
            for i in range(first_ctx, len(toks)):
                plt.annotate(toks[i], xformed_embs[i])
        
        plt.savefig(out_dir / f"{chkpt_idx}.png")
        plt.cla()
        img_idxs.append(chkpt_idx)
        chkpt_idx += args.step_size

    # Create GIF
    os.chdir(out_dir)
    os.system(f"convert -delay 20 -loop 0 {' '.join([str(i) + '.png' for i in img_idxs])} animation.gif")

if __name__ == "__main__":
    main()