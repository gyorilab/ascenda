import numpy as np
import torch

from .embedder import EntityEmbedder, build_embedding_text
from .train import (ENTITY_MAX_LENGTH, MENTION_MAX_LENGTH,
                              embed_strings_meanpool)


class JointReranker(object):
    """Jointly re-ranks ScoredMatch lists from multiple ground() calls. All lists
    should come from the same source.
    """
    def __init__(self, model, grounder=None, device="cpu", cache=None,
                 embedder=None):
        self.model = model.to(device)
        self.grounder = grounder
        self._embedder = embedder
        self.device = device
        self.cache = dict(cache or {})
        self.mention_cache = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EntityEmbedder(device=self.device,
                                               grounder=self.grounder)
        return self._embedder

    @classmethod
    def from_checkpoint(cls, path, grounder, device="cpu", cache=None):
        # For when you want to load a model from disk
        from .train import load_model
        return cls(load_model(path, device=device), grounder, device, cache)

    def embed_candidates(self, candidates):
        """Embed a batch of candidates that aren't in the entity cache yet. Takes a list of
        candidates as input. Uses [CLS] pooling.
        """
        new, seen = [], set()
        for c in candidates:
            key = (c.db, c.id)
            if key not in self.cache and key not in seen:
                seen.add(key)
                new.append(c)
        if new:
            # Get embedding text for each new candidate
            texts = [build_embedding_text(c, self.embedder.grounder) for c in new]
            vecs = self.embedder.embed_texts(texts, batch_size=64,
                                             max_length=ENTITY_MAX_LENGTH)
            # Embed each new candidate and add to the cache
            for c, v in zip(new, vecs):
                self.cache[(c.db, c.id)] = np.asarray(v, dtype="float32")

    def embed_mentions(self, texts):
        """Embed a batch of mentions that aren't in the entity cache yet. Takes a list of
        terms as input. Uses mean pooling instead of [CLS] pooling.
        """
        new = list(dict.fromkeys(t for t in texts if t not in self.mention_cache))
        if new:
            vecs = embed_strings_meanpool(self.embedder, new,
                                           max_length=MENTION_MAX_LENGTH)
            for t, v in zip(new, vecs):
                self.mention_cache[t] = np.asarray(v, dtype="float32")

    def build_model_tensors(self, mention_texts, candidate_lists):
        """Build the three tensors the model expects from lists of mention surface
        strings and their ScoredMatch lists.

        Returns the tuple (mention_emb (M, 768), cand_emb (M, C, 768), cand_mask
        (M, C) bool), or None when no mention has any candidate.
        """
        rows = [(t, c) for t, c in zip(mention_texts, candidate_lists) if c]
        if not rows:
            return None

        # 1. Embed new mentions and candidates not already in the cache
        self.embed_mentions([t for t, _ in rows])
        for _, cands in rows:
            self.embed_candidates([c.term for c in cands])

        # 2. Build tensors
        M = len(rows)
        C = max(len(c) for _, c in rows)
        dim = self.model.embed_dim

        mention_emb = np.stack([self.mention_cache[t] for t, _ in rows])
        cand_emb = np.zeros((M, C, dim), dtype="float32")
        cand_mask = torch.zeros(M, C, dtype=torch.bool)
        for i, (_, cands) in enumerate(rows):
            for j, c in enumerate(cands):
                cand_emb[i, j] = self.cache[(c.term.db, c.term.id)]  # embedding from cache
            cand_mask[i, :len(cands)] = True

        return (
            torch.as_tensor(mention_emb, dtype=torch.float32, device=self.device),
            torch.as_tensor(cand_emb, dtype=torch.float32, device=self.device),
            cand_mask.to(self.device),
        )

    def rerank(self, candidate_lists, mention_texts):
        """Re-rank a list[list[ScoredMatch]] object jointly. Returns another
        list[list[ScoredMatch]] object in the same order.

        `mention_texts` is parallel to `candidate_lists` and their lengths should
        match. They should also come from the same source.
        """
        # Build the tensors the model needs from candidate_lists
        model_tensors = self.build_model_tensors(mention_texts, candidate_lists)
        if model_tensors is None:
            return list(candidate_lists)

        # Run model and unpack output
        mention_emb, cand_emb, cand_mask = model_tensors
        self.model.eval()
        with torch.no_grad():
            log_prob = self.model(mention_emb, cand_emb, cand_mask).cpu().numpy()

        ranked = []
        i = 0  # need i since "log_prob" is a flat list (array) but candidate_lists isn't
        for cands in candidate_lists:
            if not cands:
                ranked.append(cands)
                continue
            order = np.argsort(-log_prob[i, :len(cands)])
            i += 1
            ranked.append([cands[k] for k in order])

        return ranked
