import numpy as np
import torch
from .embedder import CandidateEmbedder, _build_embedding_text


_STATUS_IDX = {"curated": 0, "name": 1, "synonym": 2, "former_name": 3}

def match_feature_vector(c):
    """Return a 10-d feature vector for a ScoredMatch with 4 dims dedicated for
    status one-hot encoding and 6 dims for normalized string sub-scores from c.match.
    """
    m = c.match
    status = [0.0, 0.0, 0.0, 0.0]
    si = _STATUS_IDX.get(getattr(c.term, "status", None))
    if si is not None:
        status[si] = 1.0
    return status + [
        m.score_short_abbr() / 1.0,
        m.score_mixed() / 2.0,
        m.score_exact() / 1.0,
        m.score_acic() / 2.0,
        m.score_combo() / 6.0,
        m.score_dash() / 2.0,
    ]


class JointReranker(object):
    """Jointly re-ranks ScoredMatch lists from multiple ground() calls.
    """
    def __init__(self, model, grounder=None, device="cpu", cache=None,
                 embedder=None):
        self.model = model.to(device)
        self.grounder = grounder
        self._embedder = embedder
        self.device = device
        self.cache = dict(cache or {})

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = CandidateEmbedder(device=self.device,
                                               grounder=self.grounder)
        return self._embedder

    @classmethod
    def from_checkpoint(cls, path, grounder, device="cpu", cache=None):
        # For when you want to load a model from disk
        from .train import load_model
        return cls(load_model(path, device=device), grounder, device, cache)

    def _embed(self, terms):
        """Embed a batch of terms that haven't been embedded yet (aren't in the cache). Takes
        a list of terms as input.
        """
        new = [t for t in terms if (t.db, t.id) not in self.cache]
        if new:
            # Get embedding text for each new term
            texts = [_build_embedding_text(t, self.embedder.grounder) for t in new]

            # Embed each new term and add to the cache
            for t, v in zip(new, self.embedder.embed_texts(texts, batch_size=64)):
                self.cache[(t.db, t.id)] = v

    def build_model_tensors(self, candidate_lists):
        """Builds tensors needed by the model from list of ScoredMatch lists.
        """
        embs, scores, m_ids, gate_feats = [], [], [], []
        m_id = 0

        # Get everything the model needs
        for cands in candidate_lists:
            if not cands:
                continue
            # 1. Embed any new candidates not already in the cache
            self._embed([c.term for c in cands])

            # 2. Build lists
            use_feats = getattr(self.model, "wants_candidate_features", False)
            for c in cands:
                emb = self.cache[(c.term.db, c.term.id)]  # embedding from cache
                if use_feats:  # append Gilda score ingredients
                    emb = np.concatenate(
                        [emb, np.asarray(match_feature_vector(c), dtype=emb.dtype)])
                embs.append(emb)
                scores.append(c.score)  # Gilda score
                m_ids.append(m_id)  # unique id

            # Get top candidate features for input to confidence gate MLP
            top_score = cands[0].score
            gate_feats.append([top_score - cands[1].score if len(cands) > 1 else top_score,
                               len(cands), top_score])
            m_id += 1

        if not embs:
            return None

        # tensorize since that's what the model expects
        embs_tensor = torch.tensor(np.stack(embs), dtype=torch.float32, device=self.device)
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device).unsqueeze(-1)  # add a dim
        m_ids_tensor = torch.tensor(m_ids, dtype=torch.long, device=self.device)
        gate_feats_tensor = torch.tensor(gate_feats, dtype=torch.float32, device=self.device)

        return embs_tensor, scores_tensor, m_ids_tensor, gate_feats_tensor

    def rerank(self, candidate_lists, context_emb=None):
        """Re-ranks a list[list[ScoredMatch]] object with a provided model. Returns another
        list[list[ScoredMatch]] object. context_emb is a 1d document-context vector.
        """
        # Build the tensors the model needs from candidate_lists
        model_tensors = self.build_model_tensors(candidate_lists)
        if model_tensors is None:
            return candidate_lists

        # Run model and unpack output
        embs, _, _, _ = model_tensors
        ctx = None
        if context_emb is not None:
            ctx = torch.as_tensor(context_emb, dtype=torch.float32,
                                  device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(embs, context_emb=ctx).cpu().numpy()

        ranked = []
        i = 0  # need i since "out" is a flat list (array) but candidate_lists isn't
        for cands in candidate_lists:
            if not cands:
                ranked.append(cands)
                continue
            n = len(cands)
            order = np.argsort(-out[i:i + n])
            i += n
            ranked.append([cands[k] for k in order])

        return ranked
