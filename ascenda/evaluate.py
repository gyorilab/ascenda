"""Evaluation for joint disambiguation model results.

Usage for best current model...

python -m ascenda.evaluate \
  --model-path ascenda/model_L3_ga8_auxtype_us_lr5e5_vrex_ns_seed0.pt \
  --datasets bioid bc5cdr nlmchem ncbi_disease gnormplus medmentions_st21pv \
  --corpus-cache ascenda/corpus_cache/tier2_merged.pkl \
  --embedding-cache ascenda/embedding_cache_tier2.pkl \
  --context-cache ascenda/context_cache_tier2.pkl
"""
from collections import defaultdict
import pandas as pd

from .data import DocumentExample
from .embedder import CandidateEmbedder


def evaluate_predictions(
    docs: list[DocumentExample],
    predictions: dict[str, dict[str, list]],
) -> pd.DataFrame:
    """Evaluate a model's predicted re-ranking of candidates by computing
    precision, recall, and F1, grouped by entity types found in the corpus.

    Params:
    -------
    docs :
        Held-out set of documents
    predictions :
        Dictionary that maps doc_id: {mention_text: list[ScoredMatch]}

    Returns:
    --------
    pd.DataFrame :
        Evaluation results grouped by entity type.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0,
                                  "total": 0})

    for doc in docs:
        doc_preds = predictions.get(doc.doc_id, {})
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            preds = doc_preds.get(mention.text, mention.candidates)
            if not preds:
                continue
            counts[etype]["has_grounding"] += 1
            top_curie = f"{preds[0].term.db}:{preds[0].term.id}"
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def filter_ambiguous_mentions(
    docs: list[DocumentExample],
    max_score_gap: float = 0.05,
    min_candidates: int = 2,
    min_top_score: float = 0.3,
) -> list[DocumentExample]:
    """Filter documents to only include mentions where Gilda's candidate
    list is genuinely ambiguous. A mention is considered ambiguous when it
    has at least `min_candidates` candidates, the gap between the 1st and 2nd
    candidate scores is at most `max_score_gap`, and the top candidate score
    is at least `min_top_score` (to exclude poor-quality matches).

    Params:
    ------
    docs :
        List of DocumentExample instances.
    max_score_gap :
        Maximum allowed difference between the top two candidate scores.
    min_candidates :
        Minimum number of candidates required.
    min_top_score :
        Minimum score for the top candidate.

    Returns:
    --------
    list[DocumentExample]
        Filtered documents containing only ambiguous mentions. Documents
        with no remaining mentions are dropped.
    """
    total_mentions = 0
    kept_mentions = 0
    filtered = []
    for doc in docs:
        amb_mentions = []
        for m in doc.mentions:
            total_mentions += 1
            if len(m.candidates) < min_candidates:
                continue
            top = m.candidates[0].score
            second = m.candidates[1].score
            gap = top - second
            if gap <= max_score_gap and top >= min_top_score:
                amb_mentions.append(m)
                kept_mentions += 1
        if amb_mentions:
            filtered.append(DocumentExample(
                doc_id=doc.doc_id,
                mentions=amb_mentions,
            ))
    print(f"Ambiguity filter: kept {kept_mentions}/{total_mentions} mentions "
          f"({kept_mentions/total_mentions:.1%}) across {len(filtered)} docs "
          f"(gap<={max_score_gap}, candidates>={min_candidates}, "
          f"top>={min_top_score})")
    return filtered


def evaluate_baseline(docs: list[DocumentExample]) -> pd.DataFrame:
    """"Evaluate Gilda's default pre-disambiguation candidate ranking.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0, "total": 0})

    for doc in docs:
        for mention in doc.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            if not mention.candidates:
                continue
            counts[etype]["has_grounding"] += 1
            top_curie = f"{mention.candidates[0].term.db}:{mention.candidates[0].term.id}"
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def _counts_to_df(counts: dict) -> pd.DataFrame:
    """Converts counts dictionary to a nice DataFrame.
    """
    rows = []
    for etype in sorted(counts.keys()):
        c = counts[etype]
        precision = c["correct"] / c["has_grounding"] if c["has_grounding"] \
            else 0
        recall = c["correct"] / c["total"] if c["total"] else 0
        f1 = (2 * precision * recall / (precision + recall)) if \
            (precision + recall) else 0
        rows.append({
            "Entity Type": etype,
            "Correct": c["correct"],
            "Has Grounding": c["has_grounding"],
            "Total": c["total"],
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
        })
    # Total row
    total = {k: sum(r[k] for r in rows) for k in ("Correct",
                                                  "Has Grounding", "Total")}
    p = total["Correct"] / total["Has Grounding"] if (
        total)["Has Grounding"] else 0
    r = total["Correct"] / total["Total"] if total["Total"] else 0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    rows.append({
        "Entity Type": "Total",
        **total,
        "Precision": round(p, 3),
        "Recall": round(r, 3),
        "F1": round(f1, 3),
    })
    return pd.DataFrame(rows)


def comparison_table(docs, predictions, model_col="Attn F1",
                     total_label="Total", add_average_row=True):
    """Generates a comparison table with per entity_type Total, Gilda F1,
    <model_col>, Gains, Losses, Net, G/L, and Net %.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "g_has": 0, "g_corr": 0,
                               "a_has": 0, "a_corr": 0, "gains": 0, "losses": 0})

    def curie(c):
        return f"{c.term.db}:{c.term.id}"

    for doc in docs:
        dp = predictions.get(doc.doc_id, {})
        for m in doc.mentions:
            r = agg[m.entity_type]
            r["total"] += 1
            base = m.candidates
            attn = dp.get(m.text, m.candidates)
            b_ok = bool(base) and curie(base[0]) in m.gold_synonyms
            a_ok = bool(attn) and curie(attn[0]) in m.gold_synonyms
            r["g_has"] += bool(base)
            r["a_has"] += bool(attn)
            r["g_corr"] += b_ok
            r["a_corr"] += a_ok
            r["gains"] += (a_ok and not b_ok)
            r["losses"] += (b_ok and not a_ok)

    def f1(corr, has, total):
        p = corr / has if has else 0
        rec = corr / total if total else 0
        return round(2 * p * rec / (p + rec), 3) if (p + rec) else 0

    def gl_ratio(gains, losses):
        if losses == 0:
            return float("inf") if gains else 0.0
        return round(gains / losses, 2)

    def net_pct(net, total):
        return round(100 * net / total, 1) if total else 0.0

    rows, tot = [], defaultdict(int)
    for et in sorted(agg):
        r = agg[et]
        for k in r:
            tot[k] += r[k]
        rows.append({
            "Entity Type": et, "Total": r["total"],
            "Gilda F1": f1(r["g_corr"], r["g_has"], r["total"]),
            model_col: f1(r["a_corr"], r["a_has"], r["total"]),
            "Gains": r["gains"],
            "Losses": r["losses"],
            "Net": r["gains"] - r["losses"],
            "G/L": gl_ratio(r["gains"], r["losses"]),
            "Net %": net_pct(r["gains"] - r["losses"], r["total"]),
        })
    # Macro (per-type) averages over the per-type rows
    gl_finite = [row["G/L"] for row in rows if row["Losses"] > 0]
    avg_gl = round(sum(gl_finite) / len(gl_finite), 2) if gl_finite else 0.0
    avg_gf1 = round(sum(row["Gilda F1"] for row in rows) / len(rows), 3) if rows else 0
    avg_af1 = round(sum(row[model_col] for row in rows) / len(rows), 3) if rows else 0
    rows.append({
        "Entity Type": total_label, "Total": tot["total"],
        "Gilda F1": f1(tot["g_corr"], tot["g_has"], tot["total"]),
        model_col: f1(tot["a_corr"], tot["a_has"], tot["total"]),
        "Gains": tot["gains"],
        "Losses": tot["losses"],
        "Net": tot["gains"] - tot["losses"],
        "G/L": gl_ratio(tot["gains"], tot["losses"]),
        "Net %": net_pct(tot["gains"] - tot["losses"], tot["total"]),
    })
    if add_average_row:
        rows.append({
            "Entity Type": "Average (per-type, excl inf)", "Total": "",
            "Gilda F1": avg_gf1, model_col: avg_af1,
            "Gains": "", "Losses": "", "Net": "", "G/L": avg_gl, "Net %": "",
        })
    import pandas as pd
    return pd.DataFrame(rows)


def filter_docs_by_source(docs, src):
    """Handles merged documents whose source is a merged string
    (e.g. 'bc5cdr+ncbi_disease').
    """
    out = []
    for d in docs:
        ms = [m for m in d.mentions if src in m.source_datasets]
        if ms:
            out.append(DocumentExample(doc_id=d.doc_id, mentions=ms,
                                       source=d.source, split=d.split))
    return out




if __name__ == "__main__":
    import argparse
    import os

    _CACHES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caches")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Path to trained attention model")
    parser.add_argument("--embedding-cache",
                        default=os.path.join(_CACHES, "embedding_cache_final.pkl"))
    parser.add_argument("--equivalences", default=None)
    parser.add_argument("--ambiguous-only", action="store_true",
                        help="Evaluate only on ambiguous mentions")
    parser.add_argument("--max-score-gap", type=float, default=0.05,
                        help="Max gap between top two candidate scores "
                             "(if --ambiguous-only)")
    parser.add_argument("--min-candidates", type=int, default=2,
                        help="Min number of candidates for a mention "
                             "(if --ambiguous-only)")
    parser.add_argument("--min-top-score", type=float, default=0.3,
                        help="Min score for the top candidate "
                             "(if --ambiguous-only)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", default=["bioid"])
    parser.add_argument("--corpus-cache", default=None)
    parser.add_argument("--context-cache", default=None,
                        help="per-document context embedding cache (for "
                             "wants_context=True models)")
    args = parser.parse_args()

    import json
    from .data import load_corpus, make_splits
    from .train import (precompute_embeddings, precompute_context_embeddings,
                        predict_document, load_model)

    equivalences = {}
    if args.equivalences:
        with open(args.equivalences) as f:
            equivalences = json.load(f)

    docs = load_corpus(args.datasets, equivalences=equivalences,
                       merged_cache=args.corpus_cache)
    _, _, test_docs = make_splits(docs)
    full_test_docs = test_docs

    if args.ambiguous_only:
        test_docs = filter_ambiguous_mentions(
            test_docs,
            max_score_gap=args.max_score_gap,
            min_candidates=args.min_candidates,
            min_top_score=args.min_top_score,
        )

    # Gilda baseline
    print("\n=== Gilda Baseline ===")
    print(evaluate_baseline(test_docs).to_markdown(index=False))

    # Attention-based model
    if args.model_path:
        infer_docs = full_test_docs if args.headline else test_docs
        embedder = CandidateEmbedder(device=args.device)
        cache = precompute_embeddings(infer_docs, embedder,
                                      args.embedding_cache)
        model = load_model(args.model_path, device=args.device)

        context_cache = None
        if getattr(model, "wants_context", False):
            context_cache = precompute_context_embeddings(
                infer_docs, embedder, args.context_cache)

        predictions = {}
        for doc in infer_docs:
            predictions[doc.doc_id] = predict_document(
                doc, cache, model, device=args.device,
                context_cache=context_cache,
            )

        model_df = evaluate_predictions(test_docs, predictions)
        print("\n=== Attention Model ===")
        print(model_df.to_markdown(index=False))

        print("\n=== Combined (all sources) ===")
        print(comparison_table(test_docs, predictions).to_markdown(index=False))

        print("\n=== Per source ===")
        srcs = sorted({s for d in test_docs for m in d.mentions
                       for s in m.source_datasets})
        for src in srcs:
            sub = filter_docs_by_source(test_docs, src)
            print(f"\n[{src}]")
            print(comparison_table(sub, predictions).to_markdown(index=False))
