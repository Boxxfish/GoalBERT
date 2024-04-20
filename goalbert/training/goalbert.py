from colbert.modeling.colbert import ColBERT, colbert_score
from typing import *

import torch
import torch.nn.functional as F

MAX_MASKS = 64
MAX_ACTIONS = 512


def probs_act_masks_to_distrs(
    probs: torch.Tensor,  # Shape: (num_queries, MAX_MASKS, MAX_ACTIONS)
    act_masks: torch.Tensor,  # Shape: (num_queries, MAX_MASKS, MAX_ACTIONS)
) -> List[torch.distributions.Categorical]:
    """
    Converts probs and action masks into a list of distributions.
    """
    # Slice down the action probablities
    all_distrs = []
    for p, m in zip(probs, act_masks):
        max_mask_idx = m.argmax(0).cpu()[0].item()
        max_action_idx = m.argmax(1).cpu()[0].item()
        distr = torch.distributions.Categorical(probs=p[:max_mask_idx, :max_action_idx])
        all_distrs.append(distr)
    return all_distrs


class GoalBERT(ColBERT):
    """
    A modified version of ColBERT, that uses only [MASK]s to generate queries.
    """

    def __init__(self, name="bert-base-uncased", colbert_config=None):
        super().__init__(name, colbert_config)

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims="return_mask")

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        assert False, "Don't use this method."

    def compute_probs(
        self,
        input_ids: torch.Tensor,  # Shape: (num_queries, query_maxlen)
        attention_mask: torch.Tensor,  # Shape: (num_queries, query_maxlen)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes probability distributions over the query and context for each [MASK] token present in each query. All
        distributions have the same # of options per action, and the same # of actions.

        Returns a tensor of probabilites (num_queries, MAX_MASKS, MAX_ACTIONS), a tensor of action masks (num_queries, MAX_MASKS, MAX_ACTIONS),
        and a tensor of non-MASKs (num_queries, MAX_ACTIONS, EMB_DIM).
        """
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
            self.device
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)  # Shape: (num_queries, query_maxlen, emb_dim)

        # `mask` shouldn't do anything to the query; [MASK] tokens count as part of the input
        mask = (
            torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device)
            .unsqueeze(2)
            .float()
        )  # Shape: (num_queries, query_maxlen, 1)
        Q = Q * mask

        # For each [MASK], generate a distribution over all non-[MASK] tokens and greedy select.
        # Non-[MASK]s include context.
        MASK = 103
        mask_idxs = (
            (input_ids == MASK).int().argmax(1).cpu().tolist()
        )  # List of the first [MASK] in each query
        probs_all = []
        action_masks_all = []
        non_masks_all = []
        for i, mask_idx in enumerate(mask_idxs):
            # Compute probabilities
            non_masks = torch.concat(
                [Q[i, :mask_idx, :], Q[i, self.colbert_config.query_maxlen :, :]], dim=0
            )  # Shape: (num_non_masks, q_dim)
            masks = Q[
                i, mask_idx : self.colbert_config.query_maxlen, :
            ]  # Shape: (num_masks, q_dim)
            interaction = masks @ non_masks.T  # Shape: (num_masks, num_non_masks)
            probs = F.pad(
                torch.softmax(interaction, dim=1),
                (
                    0,
                    MAX_ACTIONS - interaction.shape[1],
                    0,
                    MAX_MASKS - interaction.shape[0],
                ),
                "constant",
                0,
            )  # Shape: (MAX_MASKS, MAX_ACTIONS)
            non_masks_padded = torch.zeros((MAX_ACTIONS, non_masks.shape[1]))
            non_masks_padded[: non_masks.shape[0], :] = non_masks

            # Compute action masks
            action_masks = torch.ones((MAX_MASKS, MAX_ACTIONS), dtype=torch.int)
            action_masks[: interaction.shape[0], : interaction.shape[1]] = 0

            probs_all.append(probs)
            action_masks_all.append(action_masks)
            non_masks_all.append(non_masks_padded)
        probs_arr = torch.stack(probs_all)
        action_masks_arr = torch.stack(action_masks_all)
        non_masks_arr = torch.stack(non_masks_all)

        return probs_arr, action_masks_arr, non_masks_arr

    def query(
        self,
        input_ids,  # Shape: (num_queries, query_maxlen)
        attention_mask,  # Shape: (num_queries, query_maxlen)
        idxs=None,  # Shape: (num_queries, num_masks)
    ):
        new_Q = []
        probs_all, act_masks_all, non_masks_all = self.compute_probs(
            input_ids, attention_mask
        )
        for i, (probs, act_masks, non_masks) in enumerate(
            zip(probs_all, act_masks_all, non_masks_all)
        ):
            if idxs is not None:
                selected = torch.tensor(idxs[i]).long()
            else:
                distr = probs_act_masks_to_distrs(
                    probs.unsqueeze(0), act_masks.unsqueeze(0)
                )[0]
                selected = distr.sample().long()  # Shape: (num_masks,)
            selected = selected.to(non_masks.device)
            new_q = non_masks[selected]  # Shape: (MAX_MASKS, emb_dim)
            new_Q.append(new_q)
        Q = torch.stack(new_Q, 0)  # Shape: (num_queries, num_masks, q_dim)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def queryFromText(self, queries, context=None, idxs=None):
        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context
        )
        return self.query(input_ids, attention_mask, idxs)

    def score(self, Q, D_padded, D_mask):
        assert self.colbert_config.similarity == "cosine"
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]  # Shape: (num_queries, query_maxlen)
        return mask
