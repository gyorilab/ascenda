"""Builds the padded index tensors that the model trains on.
"""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class SourceTensors:
    """Index-only view of one source.
    """
    source_id: str
    ment_idx: torch.Tensor
    cand_idx: torch.Tensor
    cand_mask: torch.Tensor
    pos_mask: torch.Tensor
    n_cand: torch.Tensor
    ment_pos: tuple = ()


def build_banks(mention_vecs, embedding_cache):
    """Returns the tuple (mention_bank, entity_bank, entity_row, mention_row).
    """
    ent_ids = sorted(embedding_cache)
    entity_row = {k: i for i, k in enumerate(ent_ids)}
    entity_bank = torch.from_numpy(
        np.stack([np.asarray(embedding_cache[k], dtype="float32") for k in ent_ids]))

    keys = sorted(mention_vecs)
    mention_row = {k: i for i, k in enumerate(keys)}
    mention_bank = torch.from_numpy(np.stack([mention_vecs[k] for k in keys]))
    return mention_bank, entity_bank, entity_row, mention_row


def build_source_tensors(sources, entity_row, mention_row) -> list[SourceTensors]:
    """One SourceTensors per source that has at least one mention with a candidate.
    """
    built, skipped = [], 0
    for d in sources:
        rows, cands, pos, mpos = [], [], [], []
        for i, m in enumerate(d.mentions):
            key = (d.source_id, i)
            if not m.candidates or key not in mention_row:
                continue
            idx = [entity_row.get((c.term.db, c.term.id)) for c in m.candidates]
            if any(j is None for j in idx):
                skipped += 1
                continue
            gold = {f"{c.term.db}:{c.term.id}" for c in m.candidates} & set(m.gold_synonyms or ())
            rows.append(mention_row[key])
            mpos.append(i)
            cands.append(idx)
            pos.append([f"{c.term.db}:{c.term.id}" in gold for c in m.candidates])
        if not rows:
            continue
        M, C = len(rows), max(len(c) for c in cands)
        cand_idx = torch.zeros(M, C, dtype=torch.long)
        cand_mask = torch.zeros(M, C, dtype=torch.bool)
        pos_mask = torch.zeros(M, C, dtype=torch.bool)
        for r, (ci, pi) in enumerate(zip(cands, pos)):
            cand_idx[r, :len(ci)] = torch.tensor(ci, dtype=torch.long)
            cand_mask[r, :len(ci)] = True
            pos_mask[r, :len(pi)] = torch.tensor(pi, dtype=torch.bool)
        built.append(SourceTensors(
            d.source_id, torch.tensor(rows, dtype=torch.long), cand_idx, cand_mask,
            pos_mask, cand_mask.sum(1), tuple(mpos)))
    if skipped:
        print(f"Note: {skipped} mentions were dropped (candidate missing from the "
              f"embedding cache)")
    return built


def gather(dt, mention_bank, entity_bank, device=None):
    """Wrapper for mention and entity banks.
    """
    me, ce, mask = mention_bank[dt.ment_idx], entity_bank[dt.cand_idx], dt.cand_mask
    return (me.to(device), ce.to(device), mask.to(device)) if device else (me, ce, mask)
