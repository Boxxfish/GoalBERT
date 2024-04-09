from colbert.modeling.colbert import ColBERT, colbert_score

import torch


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

    def act_distrs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.distributions.Categorical]:
        """
        Returns distributions over the query and context for each [MASK] token present in each query.
        """
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
            self.device
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

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
        distrs = []
        for i, mask_idx in enumerate(mask_idxs):
            non_masks = torch.concat(
                [Q[i, :mask_idx, :], Q[i, self.colbert_config.query_maxlen :, :]], dim=0
            )  # Shape: (num_non_masks, q_dim)
            masks = Q[
                i, mask_idx : self.colbert_config.query_maxlen, :
            ]  # Shape: (num_masks, q_dim)
            interaction = masks @ non_masks.T  # Shape: (num_masks, num_non_masks)
            distr = torch.distributions.Categorical(
                probs=torch.softmax(interaction, dim=1)
            )
            distrs.append(distr)
        return distrs


    def query(self, input_ids, attention_mask, idxs=None):  # Shape: (num_queries, query_maxlen)
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(
            self.device
        )
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

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
        new_Q = []
        for i, mask_idx in enumerate(mask_idxs):
            non_masks = torch.concat(
                [Q[i, :mask_idx, :], Q[i, self.colbert_config.query_maxlen :, :]], dim=0
            )  # Shape: (num_non_masks, q_dim)
            masks = Q[
                i, mask_idx : self.colbert_config.query_maxlen, :
            ]  # Shape: (num_masks, q_dim)
            interaction = masks @ non_masks.T  # Shape: (num_masks, num_non_masks)
            distr = torch.distributions.Categorical(
                probs=torch.softmax(interaction, dim=1)
            )
            # selected = interaction.argmax(dim=1) # Shape: (num_masks,)
            selected = distr.sample().long()  # Shape: (num_masks,)
            new_q = non_masks[selected]  # Shape: (num_masks, q_dim)
            new_Q.append(new_q)
        Q = torch.stack(new_Q, 0)  # Shape: (num_queries, num_masks, q_dim)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def queryFromText(self, queries, context=None):
        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context
        )
        return self.query(input_ids, attention_mask)

    def score(self, Q, D_padded, D_mask):
        assert self.colbert_config.similarity == "cosine"
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]  # Shape: (num_queries, query_maxlen)
        return mask
