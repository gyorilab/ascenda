"""Joint disambiguation module for Gilda that uses PubMedBERT embeddings and
self-attention.

Candidates and mentions are embedded using a frozen PubMedBERT model, then
rounds of self-attention allows mentions to update their beliefs over their
candidates using other mentions from the same source.

One forward pass takes all mentions from a single source, lets them attend
to each others vector representations, and infers per-mention which
candidate should be ranked highest.
"""
import os
from typing import Optional

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(_MODELS, "final_model_best_checkpoint_seed0.pt")

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
    global _model, _embedder, _jr

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise ValueError(f"No checkpoint at {model_path!r}")

    if _model is None:
        from gilda.grounder import Grounder
        from .embedder import EntityEmbedder
        from .train import load_model

        _model = load_model(model_path, device=device)
        _embedder = EntityEmbedder(device=device, grounder=Grounder())

    # Re-rank with a shared JointReranker
    from .rerank import JointReranker
    if _jr is None:
        _jr = JointReranker(_model, grounder=None, device=device, cache=_cache,
                            embedder=_embedder)

    texts = list(mention_candidates.keys())
    ranked = _jr.rerank([mention_candidates[t] for t in texts], texts)
    return dict(zip(texts, ranked))