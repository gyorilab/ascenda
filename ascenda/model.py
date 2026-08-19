"""Self-attention model for joint candidate scoring."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_TYPES = (
    "Small Molecule", "Biological Function", "Disease", "Tissue/Organ", "Taxon",
    "Human Gene", "Nonhuman Gene", "Cellular Component", "Cell types/Cell lines",
    "miRNA",
)
AUX_TYPE_IGNORE = -100

BIAS_COLS = [0, 1, 2, 3, 6]


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, attn_mask):
        a, _ = self.attn(seq, seq, seq, attn_mask=attn_mask, need_weights=False)
        seq = self.norm1(seq + self.dropout(a))
        return self.norm2(seq + self.ff(seq))


class JointDisambiguator(nn.Module):
    """Disambiguates a mention's Gilda candidates by attending to the Gilda candidates
    of other mentions from the same source (document, experimental dataset, etc.). Each
    candidate is represented as a numerical vector of [LLM_embedding; gilda_score], which
    is projected into a lower-dimensional latent space, then passed through a stack of
    Transformer blocks to let candidates from different mentions inform each other as
    they're updated. A linear head then produces a scalar score per candidate, and
    sorting each mention's candidates by that score gives the re-ranked list.

     Params:
     -------
     embed_dim :
        Dimensionality of LLM embeddings (keep at 768 for PubMedBERT)
     hidden_dim :
        Dimensionality of hidden layers
     n_heads :
        Number of attention heads
     dropout :
        Dropout rate for training
    n_cand_features :
        Number of lexical features used from Gilda for a candidate (6 sub-scores that
        go into  the Gilda score calculation, and 4 one-hot flags for string match
        status)
    context_dim :
        Dimensionality of the CTX token's embedding (also keep at 768 for PubMedBERT)
    num_layers :
        Number of TransformerBlock blocks layered on top of each other
    feature_skip :
        If True, re-attach (concatenate) the n_cand_features lexical features to the
        updated vector representations produced by the num_layers TransformerBlock
        blocks.
    aux_type:
        If True, turn on per-mention entity type prediction as an auxiliary learning
        task.
    n_types:
        Number of entity type classes that the auxiliary prediction head has to choose
        from.
    """

    wants_context = True
    wants_candidate_features = True

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_cand_features=10,
        context_dim=768,
        num_layers=3,
        feature_skip=True,
        aux_type=True,
        n_types=len(AUX_TYPES)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_cand_features = n_cand_features
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.feature_skip = feature_skip
        self.aux_type = aux_type
        self.n_types = n_types
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(embed_dim + n_cand_features, hidden_dim)
        self.feat_attn_bias = nn.Linear(len(BIAS_COLS), 1)
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(num_layers)])
        self.score_head = nn.Linear(
            hidden_dim + (n_cand_features if feature_skip else 0), 1)
        if aux_type:
            self.type_head = nn.Linear(hidden_dim, n_types)

    def _trunk(self, embeddings, context_emb, mention_ids=None):
        """Handles the sending of the CTX and candidate embeddings through
        TransformerBlock blocks. Returns updated candidate representations and
        the updated CTX representation when applicable.
        """
        N = embeddings.shape[0]
        feats_b = embeddings[:, self.embed_dim:][:, BIAS_COLS]
        x = self.dropout(F.relu(self.input_proj(embeddings)))
        key_bias = self.feat_attn_bias(feats_b).squeeze(-1)
        cross_block = (mention_ids.unsqueeze(0) != mention_ids.unsqueeze(1)
                       if mention_ids is not None else None)

        if context_emb is None:
            attn_mask = key_bias.unsqueeze(0).expand(N, N).clone()
            if cross_block is not None:
                attn_mask = attn_mask.masked_fill(cross_block, float("-inf"))
            seq = x.unsqueeze(0)
            for blk in self.blocks:
                seq = blk(seq, attn_mask)
            return seq.squeeze(0), None

        ctx = F.relu(self.ctx_proj(context_emb.view(1, -1)))
        seq = torch.cat([ctx, x], dim=0).unsqueeze(0)
        full_bias = torch.cat([torch.zeros(1, device=key_bias.device), key_bias])
        attn_mask = full_bias.unsqueeze(0).expand(1 + N, 1 + N).clone()
        if cross_block is not None:
            blocked = torch.zeros(1 + N, 1 + N, dtype=torch.bool,
                                  device=cross_block.device)
            blocked[1:, 1:] = cross_block
            attn_mask = attn_mask.masked_fill(blocked, float("-inf"))
        for blk in self.blocks:
            seq = blk(seq, attn_mask)
        out = seq.squeeze(0)
        return out[1:], out[0]

    def _score(self, x, embeddings):
        """Handles the scoring of candidates based on their updated representations (and
        their lexical features if self.feature_skip is True).
        """
        if self.feature_skip:
            x = torch.cat([x, embeddings[:, self.embed_dim:]], dim=-1)
        return self.score_head(x).squeeze(-1)

    def forward(self, embeddings, context_emb=None, return_hidden=False,
                mention_ids=None):
        """Calls self._trunk and self._score to facilitate a forward pass.
        """
        x, ctx_hidden = self._trunk(embeddings, context_emb, mention_ids)
        scores = self._score(x, embeddings)
        return (scores, x, ctx_hidden) if return_hidden else scores

    def compute_aux_loss(self, x, mention_ids, type_targets, w_type=0.2):
        """Per-mention cross-entropy loss for entity type prediction. Best `w_type` was
        found to be 0.2 based on validation loss.
        """
        if not (self.aux_type and w_type > 0 and type_targets is not None
                and (type_targets != AUX_TYPE_IGNORE).any()):
            return x.new_zeros(())
        M = int(mention_ids.max().item()) + 1
        pooled = x.new_zeros(M, x.size(1)).index_add_(0, mention_ids, x)
        counts = x.new_zeros(M).index_add_(
            0, mention_ids, x.new_ones(mention_ids.shape[0]))
        pooled = pooled / counts.clamp(min=1.0).unsqueeze(1)
        return w_type * F.cross_entropy(self.type_head(pooled), type_targets,
                                        ignore_index=AUX_TYPE_IGNORE)


def compute_loss(
    scores: torch.Tensor,
    mention_ids: torch.Tensor,
    gold_indices: torch.Tensor,
    temperature: float = 1.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes per-mention cross-entropy loss averaged over mentions whose gold
     is reachable.

    Params:
    -------
    scores : (N,)
        Raw model scores for all candidates.
    mention_ids : (N,)
        Specifies which mention each candidate belongs to.
    gold_indices : (M,)
        The index of the correct candidate within each mention's candidates. A
        value of -1 means the true grounding isn't in the candidate list and that
        mention's candidates are excluded from the loss.
    temperature : float
        Denominator that divides model scores before the softmax to sharpen the
        distribution (when < 1), which restores gradient magnitude if the scores have
        a narrow range. Ranking within a mention is invariant to this, so evaluation is
        unaffected and only the training gradients change. Default is 1.0.
    weights : (M,) or None
        Per-mention loss weights for class rebalancing
        
    Returns:
    --------
    loss : scalar torch.Tensor
    """
    losses, ws = [], []
    for m_id in range(gold_indices.shape[0]):
        gold_idx = gold_indices[m_id].item()
        if gold_idx < 0:
            continue
        mention_scores = scores[mention_ids == m_id]
        if mention_scores.shape[0] == 0:
            continue
        log_probs = F.log_softmax(mention_scores / temperature, dim=0)
        losses.append(-log_probs[gold_idx])
        if weights is not None:
            ws.append(weights[m_id])
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)
    stacked = torch.stack(losses)
    if weights is None:
        return stacked.mean()
    w = torch.stack(ws).to(stacked.dtype)
    return (stacked * w).sum() / w.sum().clamp_min(1e-8)
