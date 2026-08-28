"""Self-attention model for joint candidate scoring.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageBlock(nn.Module):
    """One mean-field round of masked self-attention over mention tokens
    followed by an FFN.
    """
    def __init__(self, hidden_dim, n_heads, dropout, msg_gate=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_msg = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.msg_gate = nn.Parameter(torch.tensor(float(msg_gate)))
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, u):
        """Aggregate neighbors into a message.
        """
        mask = torch.eye(u.shape[0], dtype=torch.bool, device=u.device)
        a, _ = self.attn(u.unsqueeze(0), u.unsqueeze(0), u.unsqueeze(0),
                         attn_mask=mask, need_weights=False)
        return self.msg_gate * self.norm_ff(self.ff(self.norm_msg(a.squeeze(0))))


class JointDisambiguator(nn.Module):
    """Disambiguates a mention's Gilda candidates by attending to the semantics of other
    mentions from the same source (document, experimental dataset, etc.). Each
    mention and candidate is represented as a numerical vector, which is projected into a
    lower-dimensional latent space, and passed through a stack of attention blocks that
    allows mentions to update their belief distributions over their candidates using other
    mentions as context. A learned scoring function then produces a probability
    distribution over a mention's candidates, from which candidates are re-ranked in
    order of likelihood, and the argmax is taken as the predicted grounding.

    Params:
    -------
    embed_dim:
        Dimensionality of LLM embeddings (keep at 768 for PubMedBERT)
    hidden_dim:
        Dimensionality of hidden layers
    n_heads:
        Number of attention heads
    dropout:
        Dropout rate for training
    num_rounds:
        Number of mean-field belief-update rounds after round 0
    temperature:
        Initial value of learned temperature param for smoothing or sharpening per-mention
        beliefs
    message_temperature:
        Fixed param for smoothing beliefs passed as messages between mentions
    msg_gate:
        Initial value of learned param for how much to weight messages from other
        mentions into the score.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        num_rounds: int = 3,
        temperature: float = 0.07,
        message_temperature: float = 0.3,
        msg_gate: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_rounds = num_rounds
        self.init_temperature = temperature
        self.msg_gate_init = msg_gate

        self.mention_proj = nn.Linear(embed_dim, hidden_dim)
        self.mention_norm = nn.LayerNorm(hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.belief_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))

        self.message_temperature = message_temperature
        self.belief_norm = nn.LayerNorm(hidden_dim)
        self.belief_gate = nn.Parameter(torch.tensor(1.0))

        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ctx_gate = nn.Linear(2 * hidden_dim, 1)

        self.rounds = nn.ModuleList([
            MessageBlock(hidden_dim, n_heads, dropout, msg_gate)
            for _ in range(num_rounds)])

    def get_pairwise_sim(self, h, keys, msg=None, return_parts=False):
        """Returns match scores for candidates based on vector similarity. Returns
        all parts of the score when return_parts=True.
        """
        q = F.normalize(self.query_proj(h), dim=-1)
        local = torch.einsum("md,mcd->mc", q, keys)
        if msg is None:
            return (local, local, None, None) if return_parts else local
        qc = F.normalize(self.ctx_proj(msg), dim=-1)
        ctx = torch.einsum("md,mcd->mc", qc, keys)
        gate = torch.sigmoid(self.ctx_gate(torch.cat([h, msg], dim=-1)))
        total = local + gate * ctx
        return (total, local, ctx, gate) if return_parts else total

    def get_ctx_log_probs(self, msg, keys, cand_mask):
        """Returns per-mention log-beliefs (log probs over candidates) from the context
        term alone (msg).
        """
        qc = F.normalize(self.ctx_proj(msg), dim=-1)
        logits = (torch.einsum("md,mcd->mc", qc, keys)
                  / self.log_temp.exp().clamp(min=1e-2))
        return F.log_softmax(logits.masked_fill(~cand_mask, float("-inf")), dim=1)

    def get_score(self, h, keys, cand_mask, msg=None):
        """Returns log-beliefs tensor of size (M, C) at the learned scoring
        temperature.
        """
        logits = self.get_pairwise_sim(h, keys, msg) / self.log_temp.exp().clamp(min=1e-2)
        return F.log_softmax(logits.masked_fill(~cand_mask, float("-inf")), dim=1)

    def get_belief(self, h, keys, cand_mask, msg=None):
        """Returns log-beliefs tensor of size (M, C) at the fixed scoring
        temperature (message_temperature). Builds the distribution that's passed
        between mentions.
        """
        sim = self.get_pairwise_sim(h, keys, msg) / self.message_temperature
        return F.softmax(sim.masked_fill(~cand_mask, float("-inf")), dim=1)

    def get_keys(self, cand_emb, cand_mask):
        """Returns key vectors for candidate embeddings.
        """
        keys = F.normalize(self.key_proj(cand_emb), dim=-1)
        return keys.masked_fill(~cand_mask.unsqueeze(-1), 0.0)

    def run_rounds(self, h0, keys, cand_mask):
        """Runs the message rounds. Returns (msg, per-round log-beliefs) for each
        mention.
        """
        log_probs = [self.get_score(h0, keys, cand_mask)]
        msg = None
        if h0.shape[0] >= 2:
            for i in range(self.num_rounds):
                belief = self.get_belief(h0, keys, cand_mask, msg)
                e = (belief.unsqueeze(-1) * keys).sum(dim=1)
                u = h0 + self.belief_gate * self.belief_norm(self.belief_proj(e))
                msg = self.rounds[i](u)
                log_probs.append(self.get_score(h0, keys, cand_mask, msg))
        return msg, log_probs

    def forward(self, mention_emb, cand_emb, cand_mask, return_all=False,
                return_aux=False):
        """Calls self-contained methods to facilitate a forward pass.

        Params:
        -------
        mention_emb:
            Mention embeddings
        cand_emb:
            Candidate embeddings
        cand_mask:
            True where a candidate is real
        return_all:
            If True, return the per-round output (for deep supervision). Otherwise, return
            the final round's output.
        return_aux:
            If True, also return components that go into the `ctx` and `orth` objective fn terms.

        Returns:
        --------
        torch.Tensor (M, C) of final log-beliefs if return_all and return_aux are False.
        """
        keys = self.get_keys(cand_emb, cand_mask)
        h0 = self.mention_norm(self.dropout(self.mention_proj(mention_emb)))
        msg, log_probs = self.run_rounds(h0, keys, cand_mask)

        result = log_probs if return_all else log_probs[-1]
        if not return_aux:
            return result
        aux = {"msg": msg, "q_loc": F.normalize(self.query_proj(h0), dim=-1),
               "q_ctx": None, "ctx_log_prob": None}
        if msg is not None:
            aux["q_ctx"] = F.normalize(self.ctx_proj(msg), dim=-1)
            aux["ctx_log_prob"] = self.get_ctx_log_probs(msg, keys, cand_mask)
        return result, aux

    @torch.no_grad()
    def score_parts(self, mention_emb, cand_emb, cand_mask):
        """Returns the different parts of the candidate score as the tuple (total,
        local, ctx, gate):
            total: the overall score
            local: what the mention's own text wanted (vector similarity with mention)
            ctx: what the other mentions' wanted (vector similarity with source context)
            gate: how much the other mentions' preferences were weighted
        """
        keys = self.get_keys(cand_emb, cand_mask)
        h0 = self.mention_norm(self.mention_proj(mention_emb))
        msg, _ = self.run_rounds(h0, keys, cand_mask)
        return self.get_pairwise_sim(h0, keys, msg, return_parts=True)


def compute_loss(log_probs, pos_mask, round_weights=None):
    """Per-mention negative log-likelihood (NLL) averaged over mentions whose gold is
    reachable.

    Params
    ------
    log_probs: list[(M, C)]
        Log-beliefs from each round.
    pos_mask: (M, C) bool
        True at every correct candidate. A mention with no reachable correct candidate
        is excluded from the loss.
    round_weights: list[float] or None
        How to weight log-beliefs from each round. Defaults to uniform over rounds.

    Returns
    -------
    loss: scalar torch.Tensor loss, and the number of mentions that contributed.
    """
    trainable = pos_mask.any(dim=1)
    n = int(trainable.sum())
    if n == 0:
        return log_probs[-1].exp().sum() * 0.0, 0
    if round_weights is None:
        round_weights = [1.0 / len(log_probs)] * len(log_probs)

    total = log_probs[-1].new_zeros(())
    for w, lp in zip(round_weights, log_probs):
        masked = lp.masked_fill(~pos_mask, float("-inf"))[trainable]
        total = total + w * (-torch.logsumexp(masked, dim=1)).mean()
    return total, n
