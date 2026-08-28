"""Training loop, embedding precomputation, variance penalty, and inference for joint
disambiguation.
"""
import copy
import os
import pickle
import numpy as np
import torch
import collections
import random

from .data import NS_TYPE, load_corpus, make_splits
from .embedder import (EntityEmbedder, DEFAULT_MODEL, ORGANISM_NAMES,
                       build_embedding_text)
from .model import JointDisambiguator, compute_loss
from .tensors import build_banks, build_source_tensors, gather

DEFAULT_DATASETS = ["bioid", "bc5cdr", "nlmchem", "ncbi_disease", "gnormplus",
                    "medmentions_st21pv"]

# Make sure the caches resolve regardless of the working directory
_PKG = os.path.dirname(os.path.abspath(__file__))
_CACHES = os.path.join(_PKG, "caches")
DEFAULT_CORPUS_CACHE = os.path.join(_CACHES, "corpus_cache", "final_merged.pkl")
DEFAULT_ENTITY_CACHE = os.path.join(_CACHES, "entity_embeddings_final.pkl")
DEFAULT_MENTION_CACHE = os.path.join(_CACHES, "mention_embeddings_final.pkl")
DEFAULT_OUTPUT = os.path.join(_PKG, "models", "model.pt")

# Imported by rerank.py
ENTITY_MAX_LENGTH = 64
MENTION_MAX_LENGTH = 64


# === For building embeddings ===

@torch.no_grad()
def embed_strings_meanpool(enc: EntityEmbedder, texts, batch_size=64,
                            max_length=MENTION_MAX_LENGTH) -> np.ndarray:
    """Returns vector embeddings for mention strings, mean-pooled over all non-padding
    tokens, by passing them through a frozen pre-trained PubMedBERT encoder.
    """
    out = []
    for i in range(0, len(texts), batch_size):
        tok = enc.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(enc.device)
        hid = enc.model(**tok).last_hidden_state
        m = tok["attention_mask"].unsqueeze(-1).to(hid.dtype)
        out.append(((hid * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def get_canonical_term(sources):
    """Create dict so that there's only one term per (db, id).
    """
    best = {}
    for d in sources:
        for m in d.mentions:
            for c in m.candidates or ():
                t = c.term
                key = (t.db, t.id)
                rank = (t.organism in ORGANISM_NAMES, t.entry_name or "",
                        t.text or "", t.status or "")
                if key not in best or rank > best[key][0]:
                    best[key] = (rank, t)
    return {k: v[1] for k, v in best.items()}


def precompute_candidate_vectors(sources, cache_path, device="cpu") -> dict:
    """Returns the dict {(db, id): float32[768]} of PubMedBERT embeddings for candidates.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} entity embeddings from {cache_path}")
        return cache

    from gilda.grounder import Grounder
    enc = EntityEmbedder(model_name=DEFAULT_MODEL, device=device, grounder=Grounder())
    terms = get_canonical_term(sources)
    ids = sorted(terms)
    texts = [build_embedding_text(terms[k], enc.grounder) for k in ids]
    print(f"Embedding {len(texts)} unique candidates...")
    vecs = enc.embed_texts(texts, batch_size=64, max_length=ENTITY_MAX_LENGTH)
    cache = {k: np.asarray(v, dtype="float32") for k, v in zip(ids, vecs)}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Saved embeddings to {cache_path}")
    return cache


def precompute_mention_vectors(sources, device="cpu") -> dict:
    """Returns the dict {(source_id, mention_idx): float32[768]} of PubMedBERT embeddings
    for entity mentions.
    """
    if os.path.exists(DEFAULT_MENTION_CACHE):
        with open(DEFAULT_MENTION_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded {len(cache)} mention embeddings from {DEFAULT_MENTION_CACHE}")
        return cache

    need = {}
    for d in sources:
        for i, m in enumerate(d.mentions):
            if m.candidates:
                need.setdefault(m.text, []).append((d.source_id, i))

    out = {}
    if need:
        texts = list(need)
        n = sum(len(v) for v in need.values())
        print(f"Embedding {len(texts)} unique surface strings for {n} mentions...")
        enc = EntityEmbedder(model_name=DEFAULT_MODEL, device=device)
        for t, v in zip(texts, embed_strings_meanpool(enc, texts)):
            for k in need[t]:
                out[k] = v
    print(f"Mention embeddings: {len(out)}")

    os.makedirs(os.path.dirname(DEFAULT_MENTION_CACHE) or ".", exist_ok=True)
    with open(DEFAULT_MENTION_CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved mention embeddings to {DEFAULT_MENTION_CACHE}")
    return out


def model_forward(model, dt, mention_bank, entity_bank, device, return_aux=False):
    """Pass one source through the model with deep supervision on always.
    """
    return model(*gather(dt, mention_bank, entity_bank, device),
                 return_all=True, return_aux=return_aux)


# === For training stats ===

def accumulate_stats(stats, log_probs, dt, aux=None):
    """Update stats on mentions that are reachable.
    """
    sel = dt.pos_mask.any(dim=1) & (dt.n_cand > 1)
    if not bool(sel.any()):
        return
    pos = dt.pos_mask[sel]
    first, last = log_probs[0].detach().cpu()[sel], log_probs[-1].detach().cpu()[sel]
    a0, aL = first.argmax(1), last.argmax(1)
    ok0 = pos.gather(1, a0.unsqueeze(1)).squeeze(1)
    okL = pos.gather(1, aL.unsqueeze(1)).squeeze(1)
    stats["n"] += int(sel.sum())
    stats["c0"] += int(ok0.sum())
    stats["cL"] += int(okL.sum())
    stats["flip"] += int((a0 != aL).sum())
    stats["gain"] += int((~ok0 & okL).sum())
    stats["loss"] += int((ok0 & ~okL).sum())
    stats["nw"] += int((~ok0).sum())
    if aux is not None and aux.get("ctx_log_prob") is not None:
        ca = aux["ctx_log_prob"].detach().cpu()[sel].argmax(1)
        stats["cctx"] += int(pos.gather(1, ca.unsqueeze(1)).sum())
        stats["nctx"] += int(sel.sum())
    for tag, lp in (("h0", first), ("hL", last)):
        p = lp.exp()
        stats[tag] += float(-(p * lp.clamp(min=-30)).sum(1).sum())


def format_stats(stats):
    n = max(stats["n"], 1)
    gl = stats["gain"] + stats["loss"]
    ctxacc = (f"ctxacc={stats['cctx'] / max(stats['nctx'], 1):.4f} "
              if stats["nctx"] else "")
    return (f"acc0={stats['c0'] / n:.4f} accL={stats['cL'] / n:.4f} "
            f"flip={stats['flip'] / n:.4f} "
            f"fprec={stats['gain'] / max(gl, 1):.4f} "
            f"addr={stats['gain'] / max(stats['nw'], 1):.4f} "
            f"gain/loss={stats['gain']}/{stats['loss']} "
            f"{ctxacc}"
            f"H0={stats['h0'] / n:.3f} HL={stats['hL'] / n:.3f} n={stats['n']}")


def get_new_stats():
    return {"n": 0, "c0": 0, "cL": 0, "flip": 0, "h0": 0.0, "hL": 0.0,
            "gain": 0, "loss": 0, "nw": 0, "cctx": 0, "nctx": 0}


# === For model training ===

def get_env_labels(sources, ns_purity=0.5):
    """Give each source an environment label based on the composition of its gold
    namespaces. For implementing the Variance Risk Extrapolation (V-REx) method.

    For example, if the proportion of gold namespaces in a source that are HGNC
    exceeds `ns_purity`, then that source is given an environment label of 'ns:gene'.
    If no namespace proportion exceeds `ns_purity`, a source is labeled as 'ns:mixed'.
    Sources with no gold labels are labeled 'ns:none'.
    """
    out = {}
    for s in sources:
        hist = collections.Counter()
        for m in s.mentions:
            if m.gold_index is not None and m.candidates:
                hist[NS_TYPE.get(m.candidates[m.gold_index].term.db, "other")] += 1
        n = sum(hist.values())
        if not n:
            out[s.source_id] = "ns:none"
            continue
        top, c = hist.most_common(1)[0]
        out[s.source_id] = f"ns:{top}" if c / n >= ns_purity else "ns:mixed"
    return out


def get_round_weights(n_out, round0_weight):
    """Get weights for each message-passing belief-update round.
    """
    if n_out == 1:
        return [1.0]
    return [round0_weight] + [(1.0 - round0_weight) / (n_out - 1)] * (n_out - 1)


def train(model, train_dt, val_dt, mention_bank, entity_bank, *, epochs=300, lr=3e-4,
          weight_decay=0.01, grad_accum=8, patience=20, seed=0, device="cpu",
          deep_supervision=True, round0_weight=0.2, w_ctx=0.0,
          w_orth=0.0, env_of=None, vrex_beta=0.0, vrex_warmup=5, vrex_min_sources=50):
    device = torch.device(device)
    model = model.to(device)
    mention_bank = mention_bank.to(device)
    entity_bank = entity_bank.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_accl = -1.0
    best_state = None
    patience_counter = 0
    rng = random.Random(seed)

    # Establish V-REx environments. Keep those with >= `vrex_min_sources`, drop 'ns:none'.
    # Dropped environments still contribute to the mean risk, but not variance risk.
    vrex_envs = set()
    if vrex_beta > 0:
        if env_of is None:
            raise ValueError("vrex_beta > 0 requires env_of")
        env_n = collections.Counter(env_of[dt.source_id] for dt in train_dt)
        env_n.pop("ns:none", None)
        vrex_envs = {k for k, c in env_n.items() if c >= vrex_min_sources}
        print(f"[v-rex] beta={vrex_beta} warmup={vrex_warmup} "
              f"min_sources={vrex_min_sources}")
        print(f"[v-rex] {len(vrex_envs)} environments kept: "
              + ", ".join(f"{k}({env_n[k]})" for k, _ in env_n.most_common()
                          if k in vrex_envs))
        dropped = [k for k in env_n if k not in vrex_envs]
        if dropped:
            print(f"[v-rex] dropped (<{vrex_min_sources} sources): "
                  + ", ".join(f"{k}({env_n[k]})" for k in dropped))

    def penalize_vrex(win, beta_now):
        """Applies the V-REx penalty to the objective for one 'batch' of sources.

        Returns the tuple (objective, variance) for one gradient-accumulation window
        (batch) where objective = mean risk over all losses + beta_now * variance over
        per-environment mean risks (computed only for `vrex_envs` members).
        """
        risk = torch.stack([l for _, l in win]).mean()
        by_env = collections.defaultdict(list)
        for e, l in win:
            if e in vrex_envs:
                by_env[e].append(l)
        if len(by_env) < 2:
            return risk, None
        env_risk = torch.stack([torch.stack(ls).mean() for ls in by_env.values()])
        v = env_risk.var(unbiased=True)
        return risk + beta_now * v, v

    for epoch in range(epochs):
        # --- train ---
        model.train()
        optimizer.zero_grad()

        train_loss = 0.0
        n_train = 0
        accum = 0
        window = []
        vrex_var_sum = 0.0
        vrex_var_n = 0
        vrex_erm_windows = 0
        beta_now = vrex_beta * min(1.0, (epoch + 1) / max(vrex_warmup, 1))

        # Shuffle to decorrelate batches
        order = list(train_dt)
        rng.shuffle(order)

        for dt in order:
            need_aux = w_ctx > 0 or w_orth > 0
            if need_aux:
                log_probs, aux = model_forward(model, dt, mention_bank, entity_bank, device,
                                      return_aux=True)
            else:
                log_probs, aux = model_forward(model, dt, mention_bank, entity_bank, device), None
            if not deep_supervision:
                log_probs = [log_probs[-1]]
            loss, n = compute_loss(log_probs, dt.pos_mask.to(device),
                                   get_round_weights(len(log_probs), round0_weight))
            if n == 0:
                continue
            if aux is not None and aux.get("ctx_log_prob") is not None:
                # ctx_loss is like the contribution of other mentions to disambiguation
                if w_ctx > 0:
                    ctx_loss, ctx_n = compute_loss([aux["ctx_log_prob"]], dt.pos_mask.to(device))
                    if ctx_n:
                        loss = loss + w_ctx * ctx_loss
                # w_orth * cosine_sim is a penalty that encourages the contribution of
                # other mentions to be different from the contribution of the focal mention.
                # --> prevents them from converging on the same information.
                if w_orth > 0:
                    cosine_sim = (aux["q_loc"] * aux["q_ctx"]).sum(-1).abs().mean()
                    loss = loss + w_orth * cosine_sim
            train_loss += loss.detach().item()
            n_train += 1
            accum += 1

            # --- apply variance penalty ---
            if vrex_beta > 0:  # i.e., if vrex is ON
                window.append((env_of[dt.source_id], loss))
                if accum < grad_accum:
                    continue
                total, v = penalize_vrex(window, beta_now)
                if v is None:
                    vrex_erm_windows += 1
                else:
                    vrex_var_sum += float(v.detach())
                    vrex_var_n += 1
                total.backward()
            else:
                (loss / grad_accum).backward()
                if accum < grad_accum:
                    continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum = 0
            window = []

        if accum > 0:
            if vrex_beta > 0 and window:
                torch.stack([l for _, l in window]).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            window = []

        # --- eval ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        stats = get_new_stats()
        with torch.no_grad():
            for dt in val_dt:
                log_probs, aux = model_forward(model, dt, mention_bank, entity_bank, device,
                                      return_aux=True)
                loss, n = compute_loss([log_probs[-1]], dt.pos_mask.to(device))
                if n == 0:
                    continue
                val_loss += loss.item()
                n_val += 1
                accumulate_stats(stats, log_probs, dt, aux)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        vx = (f"vrex_var={vrex_var_sum / vrex_var_n:.2e}  beta={beta_now:.1f}"
              if vrex_var_n else "")
        if vrex_beta > 0 and vrex_erm_windows:
            vx += f"  erm_windows={vrex_erm_windows}"
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f} \n\tv-rex: {vx} \n\tstats: "
              f"{format_stats(stats)}", flush=True)

        # Select on accL (final round accuracy) instead of val_loss
        accl = stats["cL"] / max(stats["n"], 1)
        if accl > best_accl:
            best_accl = accl
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, source_tensors, mention_bank, entity_bank, device="cpu"):
    device = torch.device(device)
    model = model.to(device).eval()
    stats = get_new_stats()
    for dt in source_tensors:
        log_probs, aux = model_forward(model, dt, mention_bank, entity_bank, device,
                              return_aux=True)
        accumulate_stats(stats, log_probs, dt, aux)
    return stats


def save_model(model, path, train_config=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_type": type(model).__name__,
        "config": {
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "n_heads": model.n_heads,
            "dropout": model.dropout.p,
            "num_rounds": model.num_rounds,
            "temperature": model.init_temperature,
            "message_temperature": model.message_temperature,
            "msg_gate": model.msg_gate_init,
        },
        **({"train_config": train_config} if train_config else {}),
    }, path)
    print(f"Model saved to {path}")


def load_model(path, device="cpu") -> JointDisambiguator:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = JointDisambiguator(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model



rerankers = {}

def get_reranker(model, entity_cache, device):
    """Build once and reuse a JointReranker per (model, device) so the cache doesn't
    get duplicated and PubMedBERT is only loaded once.
    """
    from .rerank import JointReranker
    key = (id(model), str(device))
    jr = rerankers.get(key)
    if jr is None:
        jr = JointReranker(model, grounder=None, device=device,
                               cache=entity_cache)
        rerankers[key] = jr
    return jr


def predict_source(source, entity_cache: dict, model, device: str = "cpu") -> dict:
    """Run inference on a single source and return {mention_text: re-ranked
    ScoredMatch list}.
    """
    jr = get_reranker(model, entity_cache, device)
    with_cands = [m for m in source.mentions if m.candidates]
    ranked = jr.rerank([m.candidates for m in with_cands],
                       [m.text for m in with_cands]) if with_cands else []
    results = {m.text: r for m, r in zip(with_cands, ranked)}
    for m in source.mentions:
        results.setdefault(m.text, m.candidates)
    return results




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train joint disambiguation model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=3,
                   help="mean-field rounds after round 0")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--no-deep-supervision", action="store_true")
    parser.add_argument("--round0-weight", type=float, default=0.2)
    parser.add_argument("--message-temperature", type=float, default=0.3,
                   help="Fixed temperature for the belief that feeds the messages")
    parser.add_argument("--msg-gate", type=float, default=0.5,
                   help="initial mix of message into the state; reported each epoch")
    parser.add_argument("--vrex-beta", type=float, default=10.0,
                        help="coefficient for V-REx penalty in loss")
    parser.add_argument("--vrex-warmup", type=int, default=5,
                        help="epochs over which beta ramps linearly from 0")
    parser.add_argument("--vrex-min-sources", type=int, default=50,
                        help="drop environments with fewer training sources than this from "
                             "the variance term to avoid adding noise")
    parser.add_argument("--vrex-ns-purity", type=float, default=0.5,
                        help="share of a source's labeled mentions the argmax gold "
                             "type must hold for the source to be assigned to that "
                             "environment instead of ns:mixed")
    parser.add_argument("--w-ctx", type=float, default=0.5,
                   help="weight on the context-only ranking loss: require the trunk to "
                        "disambiguate with no mention-text query (additive mode only)")
    parser.add_argument("--w-orth", type=float, default=0.1,
                   help="weight on |cos(q_ctx, q_loc)|: require what the trunk adds to "
                        "be orthogonal to what the mention's own text already says")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--corpus-cache", default=DEFAULT_CORPUS_CACHE,
                   help="v2 gold. The v1 set in caches/corpus_cache/ belongs to the "
                        "older model and is not comparable.")
    parser.add_argument("--embedding-cache", default=DEFAULT_ENTITY_CACHE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sources = load_corpus(args.datasets, merged_cache=args.corpus_cache)
    train_sources, val_sources, test_sources = make_splits(sources)
    print(f"sources: {len(train_sources)} train / {len(val_sources)} val / {len(test_sources)} test")

    mention_vecs = precompute_mention_vectors(sources, device=args.device)
    embedding_cache = precompute_candidate_vectors(sources, args.embedding_cache, device=args.device)

    mention_bank, entity_bank, entity_row, mention_row = build_banks(
        mention_vecs, embedding_cache)
    print(f"banks: mentions {tuple(mention_bank.shape)}  "
          f"entities {tuple(entity_bank.shape)}")

    train_dt = build_source_tensors(train_sources, entity_row, mention_row)
    val_dt = build_source_tensors(val_sources, entity_row, mention_row)
    test_dt = build_source_tensors(test_sources, entity_row, mention_row)
    trainable = sum(int((dt.pos_mask.any(1) & (dt.n_cand > 1)).sum()) for dt in train_dt)
    print(f"source tensors: {len(train_dt)}/{len(val_dt)}/{len(test_dt)}; "
          f"{trainable} reachable multi-candidate train mentions")

    model = JointDisambiguator(
        hidden_dim=args.hidden_dim, n_heads=args.n_heads, dropout=args.dropout,
        num_rounds=args.rounds, temperature=args.temperature,
        message_temperature=args.message_temperature, msg_gate=args.msg_gate)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    model = train(model, train_dt, val_dt, mention_bank, entity_bank,
                  epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                  grad_accum=args.grad_accum, patience=args.patience, seed=args.seed,
                  device=args.device, deep_supervision=not args.no_deep_supervision,
                  round0_weight=args.round0_weight,
                  env_of=(get_env_labels(train_sources, args.vrex_ns_purity)
                          if args.vrex_beta > 0 else None),
                  vrex_beta=args.vrex_beta, vrex_warmup=args.vrex_warmup,
                  vrex_min_sources=args.vrex_min_sources, w_ctx=args.w_ctx, w_orth=args.w_orth)

    for name, dts in (("val", val_dt), ("test", test_dt)):
        print(f"FINAL {name}: {format_stats(evaluate(model, dts, mention_bank, entity_bank, args.device))}")

    save_model(model, args.output, train_config={**vars(args)})
