"""Joint disambiguation module for Gilda that uses PubMedBERT embeddings and
self-attention.

Candidates are embedded using their full name and namespace label via a frozen
PubMedBERT model, concatenated with a 10-d status/string-match vector, and
re-ranked jointly based on their representations after being passed through a
stack of transformer blocks.

One forward pass takes all mentions from a single document/source, lets them
attend to each others vector representations (as well as a per-document CTX
token built from embedding all mention spans as one string), and infers
per-mention which candidate should be ranked highest.
"""
from typing import Optional

_model = None
_embedder = None
_cache = {}
_jr = None


def disambiguate(
        mention_candidates: dict[str, list],
        model_path: Optional[str] = None,
        device: str = "cpu",
) -> dict[str, list]:
    """Return re-ranked Gilda candidates using joint disambiguation.

    Params:
    -------
    mention_candidates :
        Mapping of text mentions to their Gilda ScoredMatch lists from
        gilda.ground(). For joint disambiguation, all mentions should
        come from the same source document/dataset.
    model_path :
        Path to a trained disambiguation model checkpoint.
    device :
        Device to use for inference ('cpu' or 'cuda').

    Returns:
    --------
    dict[str, list[ScoredMatch]]
        Same structure as mention_candidates but with re-ranked candidates.
    """
    global _model, _embedder

    if model_path is None:
        raise ValueError("model_path is required (no bundled model yet)")

    if _model is None:
        import torch
        from .embedder import CandidateEmbedder
        from .train import load_model

        _model = load_model(model_path, device=device)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)

        # Use rich embeddings if the checkpoint was trained with them
        embedding_mode = ckpt.get("embedding_mode", "plain")
        if embedding_mode == "rich":
            from gilda.grounder import Grounder
            _embedder = CandidateEmbedder(device=device, grounder=Grounder())
        else:
            _embedder = CandidateEmbedder(device=device)

    # Re-rank with a shared JointReranker
    from .rerank import JointReranker
    global _jr
    if _jr is None:
        _jr = JointReranker(_model, grounder=None, device=device, cache=_cache, embedder=_embedder)

    texts = list(mention_candidates.keys())
    ctx = None
    if getattr(_model, "wants_context", False) and texts:
        ctx = _embedder.embed_text(", ".join(dict.fromkeys(texts)), max_length=512)
    ranked = _jr.rerank([mention_candidates[t] for t in texts], context_emb=ctx)
    return dict(zip(texts, ranked))