from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.colbert import ColBERT

import torch
class GoalBERT(ColBERT):
    """
        A modified version of ColBERT, that uses only [MASK]s to generate queries.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        assert False, "Don't use this method."

    def query(
        self,
        input_ids, # Shape: (num_queries, query_maxlen)
        attention_mask
    ):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        # `mask` shouldn't do anything to the query; [MASK] tokens count as part of the input
        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float() # Shape: (num_queries, query_maxlen, 1)
        Q = Q * mask

        # For each [MASK], generate a distribution over all non-[MASK] tokens and greedy select
        MASK = 103
        mask_idxs = (input_ids == MASK).int().argmax(1).cpu().tolist() # List of the first [MASK] in each query
        new_Q = []
        for i, mask_idx in enumerate(mask_idxs):
            non_masks = Q[i, :mask_idx, :] # Shape: (num_non_masks, q_dim)
            masks = Q[i, mask_idx:, :] # Shape: (num_masks, q_dim)
            num_masks, q_dim = masks.shape
            interaction = masks @ non_masks.T # Shape: (num_masks, num_non_masks)
            max_select = interaction.argmax(dim=1) # Shape: (num_masks,)
            new_q = non_masks[max_select] # Shape: (num_masks, q_dim)
            new_Q.append(new_q)
        Q = torch.stack(new_Q, 0) # Shape: (num_queries, num_masks, q_dim)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def score(self, Q, D_padded, D_mask):
        assert self.colbert_config.similarity == 'cosine'
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()] # Shape: (num_queries, query_maxlen)
        return mask


def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction == "colbert", config.interaction

    return scores.sum(-1)


def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu:
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
