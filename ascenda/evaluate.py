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

from .data import SourceExample


def evaluate_predictions(
    sources: list[SourceExample],
    predictions: dict[str, dict[str, list]],
) -> pd.DataFrame:
    """Evaluate a model's predicted re-ranking of candidates by computing
    precision, recall, and F1, grouped by entity types found in the corpus.

    Params:
    -------
    sources :
        Held-out set of sources
    predictions :
        Dictionary that maps source_id: {mention_text: list[ScoredMatch]}

    Returns:
    --------
    pd.DataFrame :
        Evaluation results grouped by entity type.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0,
                                  "total": 0})

    for source in sources:
        source_preds = predictions.get(source.source_id, {})
        for mention in source.mentions:
            etype = mention.entity_type
            counts[etype]["total"] += 1
            preds = source_preds.get(mention.text, mention.candidates)
            if not preds:
                continue
            counts[etype]["has_grounding"] += 1
            top_curie = f"{preds[0].term.db}:{preds[0].term.id}"
            if top_curie in mention.gold_synonyms:
                counts[etype]["correct"] += 1

    return _counts_to_df(counts)


def filter_ambiguous_mentions(
    sources: list[SourceExample],
    max_score_gap: float = 0.05,
    min_candidates: int = 2,
    min_top_score: float = 0.3,
) -> list[SourceExample]:
    """Filter sources to only include mentions where Gilda's candidate
    list is genuinely ambiguous. A mention is considered ambiguous when it
    has at least `min_candidates` candidates, the gap between the 1st and 2nd
    candidate scores is at most `max_score_gap`, and the top candidate score
    is at least `min_top_score` (to exclude poor-quality matches).

    Params:
    ------
    sources :
        List of SourceExample instances.
    max_score_gap :
        Maximum allowed difference between the top two candidate scores.
    min_candidates :
        Minimum number of candidates required.
    min_top_score :
        Minimum score for the top candidate.

    Returns:
    --------
    list[SourceExample]
        Filtered sources containing only ambiguous mentions. Sources
        with no remaining mentions are dropped.
    """
    total_mentions = 0
    kept_mentions = 0
    filtered = []
    for source in sources:
        amb_mentions = []
        for m in source.mentions:
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
            filtered.append(SourceExample(
                source_id=source.source_id,
                mentions=amb_mentions,
            ))
    print(f"Ambiguity filter: kept {kept_mentions}/{total_mentions} mentions "
          f"({kept_mentions/total_mentions:.1%}) across {len(filtered)} sources "
          f"(gap<={max_score_gap}, candidates>={min_candidates}, "
          f"top>={min_top_score})")
    return filtered


def evaluate_baseline(sources: list[SourceExample]) -> pd.DataFrame:
    """"Evaluate Gilda's default pre-disambiguation candidate ranking.
    """
    counts = defaultdict(lambda: {"correct": 0, "has_grounding": 0, "total": 0})

    for source in sources:
        for mention in source.mentions:
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


def comparison_table(sources, predictions, model_col="Attn F1",
                     total_label="Total", add_average_row=True):
    """Generates a comparison table with per entity_type Total, Gilda F1,
    <model_col>, Gains, Losses, Net, G/L, and Net %.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "g_has": 0, "g_corr": 0,
                               "a_has": 0, "a_corr": 0, "gains": 0, "losses": 0})

    def curie(c):
        return f"{c.term.db}:{c.term.id}"

    for source in sources:
        dp = predictions.get(source.source_id, {})
        for m in source.mentions:
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


def filter_sources_by_origin(sources, src):
    """Handles merged source whose origin is a merged string
    (e.g. 'bc5cdr+ncbi_disease').
    """
    out = []
    for s in sources:
        ms = [m for m in s.mentions if src in m.source_datasets]
        if ms:
            out.append(SourceExample(source_id=s.source_id, mentions=ms,
                                       source=s.source, split=s.split))
    return out




if __name__ == "__main__":
    import argparse
    import json
    import pickle

    from .data import load_corpus, make_splits
    from .train import (DEFAULT_CORPUS_CACHE, DEFAULT_DATASETS,
                        DEFAULT_ENTITY_CACHE, load_model, predict_source)
    from .__init__ import DEFAULT_MODEL_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="mean-field model checkpoint")
    parser.add_argument("--embedding-cache", default=DEFAULT_ENTITY_CACHE)
    parser.add_argument("--corpus-cache", default=DEFAULT_CORPUS_CACHE)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
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
    parser.add_argument("--headline", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    equivalences = {}
    if args.equivalences:
        with open(args.equivalences) as f:
            equivalences = json.load(f)

    sources = load_corpus(args.datasets, equivalences=equivalences,
                       merged_cache=args.corpus_cache)
    _, _, test_sources = make_splits(sources)
    full_test_sources = test_sources

    if args.ambiguous_only:
        test_sources = filter_ambiguous_mentions(
            test_sources,
            max_score_gap=args.max_score_gap,
            min_candidates=args.min_candidates,
            min_top_score=args.min_top_score,
        )

    # Gilda baseline
    print("\n=== Gilda Baseline ===")
    print(evaluate_baseline(test_sources).to_markdown(index=False))

    infer_sources = full_test_sources if args.headline else test_sources
    with open(args.embedding_cache, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} entity embeddings from {args.embedding_cache}")
    model = load_model(args.model_path, device=args.device)

    predictions = {}
    for source in infer_sources:
        predictions[source.source_id] = predict_source(
            source, cache, model, device=args.device)

    model_df = evaluate_predictions(test_sources, predictions)
    print("\n=== Attention Model ===")
    print(model_df.to_markdown(index=False))

    print("\n=== Combined (all sources) ===")
    print(comparison_table(test_sources, predictions).to_markdown(index=False))

    print("\n=== Per source ===")
    srcs = sorted({s for d in test_sources for m in d.mentions
                   for s in m.source_datasets})
    for src in srcs:
        sub = filter_sources_by_origin(test_sources, src)
        print(f"\n[{src}]")
        print(comparison_table(sub, predictions).to_markdown(index=False))
