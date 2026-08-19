"""Corpus loading and preprocessing for joint disambiguation.

Imports utilities for ID normalization, entity type classification,
synonym expansion, and organism priority from ascenda/bioid_compat.py,
which vendors the pure helpers and lazily loads BioIDBenchmarker itself.
"""
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from gilda.grounder import Grounder, ScoredMatch

from .bioid_compat import (
    MODULE,
    URL,
    get_benchmarker,
    get_entity_type,
    normalize_id,
    normalize_ids,
)


@dataclass
class MentionExample:
    """For a single mention's candidates and ground truth label.
    """
    text: str
    entity_type: str
    candidates: list
    gold_curies: set = field(default_factory=set)
    gold_synonyms: set = field(default_factory=set)
    gold_index: Optional[int] = None
    offsets: Optional[list] = None
    source_datasets: set = field(default_factory=set)


@dataclass
class DocumentExample:
    """For all mentions from one document, forming one training example
    """
    doc_id: str
    mentions: list = field(default_factory=list)
    source: Optional[str] = None
    split: Optional[str] = None


_normalize_id = normalize_id
_normalize_ids = normalize_ids
_get_entity_type = get_entity_type

_mesh_chebi_crosswalk: Optional[dict] = None

# for _get_benchmarker
_benchmarker = None

def _get_benchmarker(
    grounder: Optional[Grounder] = None,
    equivalences: Optional[dict] = None,
):
    """For lazily creating a BioIDBenchmarker to do synonym expansion and organism priority
    """
    global _benchmarker
    if _benchmarker is None:
        _benchmarker = get_benchmarker(
            grounder=grounder or Grounder(),
            equivalences=equivalences or {},
        )
    return _benchmarker


def assign_gold_index(mention: MentionExample) -> None:
    """Set mention.gold_index to the position of the first candidate
    whose curie is in gold_synonyms, or None if no candidate matches.
    """
    if mention.gold_index is not None:
        return
    for i, cand in enumerate(mention.candidates):
        if f"{cand.term.db}:{cand.term.id}" in mention.gold_synonyms:
            mention.gold_index = i
            return


def load_bioid_corpus(
    grounder: Optional[Grounder] = None,
    equivalences: Optional[dict] = None,
) -> list[DocumentExample]:
    """Load the BioCreative BioID corpus as a DocumentExample list.

    Params:
    ------
    grounder :
        Gilda Grounder instance. If None, uses default.
    equivalences :
        Equivalence mappings dict (e.g., from equivalences.json).
        
    Returns:
    --------
    list[DocumentExample]
        A list containing DocumentExample objects, each representing a document 
        with its mentions and associated disambiguation information.
    """

    if grounder is None:
        grounder = Grounder()
    benchmarker = _get_benchmarker(grounder, equivalences)

    print("Loading BioID annotations...")
    df = MODULE.ensure_tar_df(
        url=URL,
        inner_path="BioIDtraining_2/annotations.csv",
        read_csv_kwargs=dict(sep=",", low_memory=False),
    )
    df["obj"] = df["obj"].apply(_normalize_ids)
    df["obj_synonyms"] = df["obj"].apply(benchmarker.get_synonym_set)
    df["entity_type"] = df.apply(
        lambda row: _classify_entity_type(row.obj, row.obj_synonyms), axis=1,
    )
    df = df[df["entity_type"] != "other"]

    print("Generating Gilda candidates per document...")
    documents = []
    for doc_id, group in tqdm(df.groupby("don_article"), desc="Documents"):
        organisms = benchmarker._get_organism_priority(doc_id)
        doc = DocumentExample(doc_id=str(doc_id))
        seen_texts: dict[str, MentionExample] = {}
        for _, row in group.iterrows():
            text = row["text"]
            if text not in seen_texts:
                candidates = grounder.ground(text, organisms=organisms)
                mention = MentionExample(
                    text=text,
                    entity_type=row["entity_type"],
                    candidates=candidates,
                    gold_curies=set(row["obj"]),
                    gold_synonyms=set(row["obj_synonyms"]),
                )
                assign_gold_index(mention)
                seen_texts[text] = mention
            else:
                existing = seen_texts[text]
                existing.gold_curies.update(row["obj"])
                existing.gold_synonyms.update(row["obj_synonyms"])
                if existing.gold_index is None:
                    assign_gold_index(existing)
        doc.mentions = list(seen_texts.values())
        if doc.mentions:
            documents.append(doc)

    print(f"Loaded {len(documents)} documents with "
          f"{sum(len(d.mentions) for d in documents)} unique mentions.")
    return documents


# === BigBio data loading ===
def load_bigbio_corpus(
        dataset_name: str,
        mentions_df,
        grounder: Optional[Grounder] = None,
        equivalences: Optional[dict] = None,
        top_k: int = 20,
        cache_path: Optional[str] = None,
        exclude_types=("CompositeMention",),
        umls_crosswalk: Optional[dict] = None,
        drop_other: bool = False,
) -> list[DocumentExample]:
    """Load one BigBio dataset from its flattened parquet mention table into a
    DocumentExample list like load_bioid_corpus.

    drop_other: if True, skip mentions classified as "other" (e.g. MedMentions
    TUIs outside _UMLS_TUI_GROUP). Default is False.
    """
    import pickle
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded cached {dataset_name} corpus ({len(docs)} docs) "
              f"from {cache_path}")
        return docs

    if grounder is None:
        grounder = Grounder()
    benchmarker = _get_benchmarker(grounder, equivalences)

    # Drop unwanted annotation types
    if exclude_types:
        excl = set(exclude_types)
        keep = mentions_df["type"].apply(lambda t: not (set(map(str, t)) & excl))
        dropped = int((~keep).sum())
        mentions_df = mentions_df[keep]
        if dropped:
            print(f"{dataset_name}: excluded {dropped} mentions of "
                  f"types {sorted(excl)}")

    documents = []
    for doc_id, group in tqdm(mentions_df.groupby("document_id"),
                              desc=f"{dataset_name} docs"):
        split = str(group["split"].iloc[0])
        doc = DocumentExample(doc_id=str(doc_id), source=dataset_name, split=split)
        for _, row in group.iterrows():
            db_ids = list(row["db_ids"])
            type_list = list(row["type"])
            gold_synonyms = expand_gold_curies(db_ids, benchmarker,
                                               umls_crosswalk=umls_crosswalk)
            entity_type = classify_entity_type_bigbio(
                dataset_name, type_list, gold_synonyms)
            if drop_other and entity_type == "other":
                continue  # skip out-of-scope concept types

            text = row["text"]
            offsets = [tuple(o) for o in row["offsets"]]
            candidates = grounder.ground(text)[:top_k]

            mention = MentionExample(
                text=text,
                entity_type=entity_type,
                candidates=candidates,
                gold_curies=set(db_ids),
                gold_synonyms=gold_synonyms,
                offsets=offsets,
                source_datasets={dataset_name},
            )
            assign_gold_index(mention)
            doc.mentions.append(mention)

        if doc.mentions:
            documents.append(doc)

    print(f"{dataset_name}: {len(documents)} docs, "
          f"{sum(len(d.mentions) for d in documents)} mentions")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Cached {dataset_name} corpus to {cache_path}")
    return documents


# === For single corpus building ===

_BIGBIO_PARQUET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_mentions",
)

def load_corpus(
        datasets: list[str],
        grounder: Optional[Grounder] = None,
        equivalences: Optional[dict] = None,
        parquet_dir: Optional[str] = None,
        corpus_cache_dir: Optional[str] = None,
        merged_cache: Optional[str] = None,
        top_k: int = 20,
        dedup: bool = True,
) -> list[DocumentExample]:
    """Build a combined corpus from one or more sources.
    """
    import pickle

    # Package-anchored so the caches are found regardless of cwd. The old
    # "ascenda/corpus_cache" default predates the move into caches/.
    if corpus_cache_dir is None:
        corpus_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "caches", "corpus_cache")
    if merged_cache and os.path.exists(merged_cache):
        with open(merged_cache, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded merged corpus ({len(docs)} docs) from {merged_cache}")
        return docs

    if grounder is None:
        grounder = Grounder()
    if parquet_dir is None:
        parquet_dir = _BIGBIO_PARQUET_DIR

    # Load the prebuilt UMLS crosswalk if any MedMentions source is requested.
    umls_crosswalk = None
    if any(d.startswith("medmentions") for d in datasets):
        cw_path = os.path.join(corpus_cache_dir, "umls_crosswalk.json")
        if not os.path.exists(cw_path):
            raise FileNotFoundError(
                f"UMLS crosswalk not found at {cw_path}; build it once with "
                f"build_umls_crosswalk(<MRCONSO.RRF>, cache_path='{cw_path}').")
        with open(cw_path) as f:
            umls_crosswalk = {k: set(v) for k, v in json.load(f).items()}
        print(f"Loaded UMLS crosswalk ({len(umls_crosswalk)} CUIs)")

    all_docs: list[DocumentExample] = []
    for name in datasets:
        if name == "bioid":
            bioid_docs = load_bioid_corpus(grounder=grounder, equivalences=equivalences)
            train, val, test = split_by_document(bioid_docs)
            for d in train:
                d.source, d.split = "bioid", "train"
            for d in val:
                d.source, d.split = "bioid", "validation"
            for d in test:
                d.source, d.split = "bioid", "test"
            all_docs.extend(bioid_docs)
        else:
            import pandas as pd
            path = os.path.join(parquet_dir, f"{name}.parquet")
            df = pd.read_parquet(path)
            cache_path = os.path.join(corpus_cache_dir, f"{name}.pkl")
            docs = load_bigbio_corpus(name, df, grounder=grounder,
                                      equivalences=equivalences, top_k=top_k,
                                      umls_crosswalk=umls_crosswalk,
                                      cache_path=cache_path)
            all_docs.extend(docs)

    if dedup:
        all_docs = resolve_cross_dataset_overlap(all_docs)

    print(f"Combined corpus: {len(all_docs)} docs, "
          f"{sum(len(d.mentions) for d in all_docs)} mentions")
    if merged_cache:
        os.makedirs(os.path.dirname(merged_cache) or ".", exist_ok=True)
        with open(merged_cache, "wb") as f:
            pickle.dump(all_docs, f)
        print(f"Cached merged corpus to {merged_cache}")

    return all_docs


# === For the train/val/test split ===

def split_by_document(
    examples: list[DocumentExample],
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[DocumentExample], list[DocumentExample], list[DocumentExample]]:
    """Split at document level for joint disambiguation
    """
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train)
    n_val = int(n * val)
    train_docs = [examples[i] for i in indices[:n_train]]
    val_docs = [examples[i] for i in indices[n_train : n_train + n_val]]
    test_docs = [examples[i] for i in indices[n_train + n_val :]]
    return train_docs, val_docs, test_docs


def make_splits(docs):
    """Partition a corpus into train, validation, and test sets by each document's
    assigned split.
    """
    train = [d for d in docs if d.split == "train"]
    val = [d for d in docs if d.split == "validation"]
    test = [d for d in docs if d.split == "test"]
    unsplit = [d for d in docs if d.split not in ("train", "validation", "test")]
    if unsplit:
        train_extra, val_extra, test_extra = split_by_document(unsplit)
        train.extend(train_extra)
        val.extend(val_extra)
        test.extend(test_extra)
    return train, val, test



def _normalize_bigbio_curie(curie: str) -> list[str]:
    """Convert a BigBio gold curie to Gilda convention. Simple string
    formatting.
    """
    prefix, _, identifier = curie.partition(":")
    identifier = identifier.strip()
    if not identifier:
        return []
    if prefix == "MESH":
        return [f"MESH:{identifier}"]  # MESH already native to Gilda
    if prefix == "OMIM":
        return [f"OMIM:{identifier}"]
    if prefix == "NCBIGene":
        return [f"NCBI gene:{identifier}"]
    if prefix in ("CHEBI", "GO"):
        return [_normalize_id(f"{prefix}:{identifier}")]
    if prefix == "UMLS":
        return []  # UMLS prefixes are handled in expand_gold_curies
    return [curie]

def build_mesh_chebi_crosswalk(cache_path=None) -> dict:
    """Build a mapping from MeSH id to CHEBI curies by inverting
    bio_ontology's CHEBI to MeSH xrefs. Build once and cache.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    from indra.ontology.bio import bio_ontology
    bio_ontology.initialize()
    inverse = defaultdict(set)
    for node in bio_ontology.nodes:
        if not node.startswith("CHEBI:"):
            continue
        ns, _, rid = node.partition(":")
        for prefix, ident in bio_ontology.get_mappings(ns, rid):
            if prefix == "MESH":
                inverse[ident].add(f"CHEBI:{rid}")
    out = {k: sorted(v) for k, v in inverse.items()}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)

    return {k: set(v) for k, v in out.items()}

def _get_mesh_chebi_crosswalk() -> dict:
    global _mesh_chebi_crosswalk
    if _mesh_chebi_crosswalk is None:
        _mesh_chebi_crosswalk = build_mesh_chebi_crosswalk(
            cache_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mesh_chebi_crosswalk.json"))
    return _mesh_chebi_crosswalk


# Dict to specify allowed namespaces and handle the MSH->MESH translation
_UMLS_SAB_MAP = {"MSH": "MESH", "HGNC": "HGNC", "OMIM": "OMIM", "GO": "GO"}

def build_umls_crosswalk(mrconso_path, cache_path=None, langs=("ENG",)):
    """Build a mapping from CUI ids to Gilda-formatted xref curies from the UMLS Metathesaurus.
    MRCONSO.RRF is read to build the mapping once, then it's cached.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    crosswalk = defaultdict(set)
    with open(mrconso_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            cui, lat, sab, code = p[0], p[1], p[11], p[13]
            if langs and lat not in langs:
                continue
            ns = _UMLS_SAB_MAP.get(sab)
            if not ns:
                continue
            bare = code.split(":")[-1]
            seeds = {"MESH": f"MESH:{bare}", "HGNC": f"HGNC:{bare}",
                    "GO": f"GO:GO:{bare}", "OMIM": f"OMIM:{bare}"}
            seed = seeds[ns]
            crosswalk[cui].add(seed)
    crosswalk = {k: sorted(v) for k, v in crosswalk.items()}
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(crosswalk, f)
        print(f"UMLS crosswalk: {len(crosswalk)} CUIs -> {cache_path}")
    return {k: set(v) for k, v in crosswalk.items()}


# Namespaces gilda/generate_terms.py refuses to collapse MeSH into (see the
# `mapping[0] not in {'EFO', 'HP', 'DOID'}` guards there). Mirrored here so the gold
# closure accepts exactly the identifiers the term index can actually emit.
_MESH_MAP_SKIP_NS = {"EFO", "HP", "DOID"}
_mesh_curated_fwd = None
_mesh_curated_rev = None


def _get_mesh_curated_mappings(skip_ns=_MESH_MAP_SKIP_NS):
    """MeSH <-> other-namespace mappings from gilda/resources/mesh_mappings.tsv.

    This is the SAME curated table gilda.generate_terms consults when deciding which
    namespace a term is indexed under, so closing the gold set over it makes the
    annotation speak the vocabulary the grounder was built to answer in. Read
    straight from the TSV rather than importing generate_terms, which would pull
    term generation in at import time.
    """
    global _mesh_curated_fwd, _mesh_curated_rev
    if _mesh_curated_fwd is None:
        from gilda.resources import MESH_MAPPINGS_PATH
        fwd, rev = {}, {}
        with open(MESH_MAPPINGS_PATH) as fh:
            for line in fh:
                row = line.rstrip("\n").split("\t")
                if len(row) < 5 or row[3] in skip_ns:
                    continue
                fwd[row[1]] = (row[3], row[4])
                rev[(row[3], row[4])] = row[1]
        _mesh_curated_fwd, _mesh_curated_rev = fwd, rev
    return _mesh_curated_fwd, _mesh_curated_rev


def close_over_curated_mesh_mappings(seeds):
    """Add each seed's curated MeSH<->X partner. Symmetric: BigBio annotates most
    chemicals in MeSH, BioID annotates some in CHEBI directly, and the term index
    may have collapsed either direction."""
    fwd, rev = _get_mesh_curated_mappings()
    out = []
    for s in seeds:
        ns, _, ident = s.partition(":")
        if ns == "MESH":
            m = fwd.get(ident)                    # ('CHEBI', 'CHEBI:4167')
            if m:
                out.append(f"{m[0]}:{m[1]}")
        else:
            mesh_id = rev.get((ns, ident))
            if mesh_id:
                out.append(f"MESH:{mesh_id}")
    return out


def expand_gold_curies(db_ids, benchmarker=None, umls_crosswalk=None) -> set[str]:
    """Expand a BigBio mention's gold db ids into synonyms using the
    BioIDBenchmarker's get_synonym_set method.

    Params:
    -------
    db_ids: list[str]
        list of string db ids
    benchmarker:
        BioIDBenchmarker (created lazily)
    umls_crosswalk: dict[str, set[str]]
        Dict with CUI keys and sets of Gilda-formatted curies as values

    Returns:
    --------
    Set of curies formatted in the Gilda convention
    """
    if benchmarker is None:
        benchmarker = _get_benchmarker()
    mesh_chebi = _get_mesh_chebi_crosswalk()
    seeds = []
    for curie in db_ids:
        if curie.startswith("UMLS:") and umls_crosswalk is not None:
            seeds.extend(umls_crosswalk.get(curie.split(":", 1)[1], ()))
        else:
            seeds.extend(_normalize_bigbio_curie(curie))
    # Bridge any MeSH seed to CHEBI
    for s in list(seeds):
        if s.startswith("MESH:"):
            seeds.extend(mesh_chebi.get(s.split(":", 1)[1], ()))
    if not seeds:
        return set()
    out = benchmarker.get_synonym_set(seeds)
    # Then close over the curated table the term index itself was built from. The
    # bridge above is bio_ontology inverted, so it cannot supply a pair bio_ontology
    # lacks -- which is exactly the case for chemicals whose MeSH term was collapsed
    # into CHEBI at index-build time and is therefore unreachable as a candidate.
    #
    # Partners are unioned in WITHOUT being re-expanded. Feeding them back through
    # get_synonym_set instead would follow their xref edges and pull in
    # same-namespace siblings that are NOT the same entity (methionine +
    # methionine zwitterion, glucose + aldehydo-D-glucose). Measured: re-expanding
    # adds 210 such pairs to the training gold and changes no evaluation number.
    return out | set(close_over_curated_mesh_mappings(list(out)))


# === For cross-dataset deduplication and overlap handling ===

# Sources whose doc id is a PMC id (others use PMID).
_PMC_SOURCES = {"nlmchem", "bioid"}

def canonical_doc_key(doc) -> str:
    prefix = "PMC" if doc.source in _PMC_SOURCES else "PMID"
    return f"{prefix}:{doc.doc_id}"

# Set split priority in case of overlap (default to train)
_SPLIT_RANK = {"test": 3, "validation": 2, "train": 1}

def _pick_split(splits) -> str:
    return max((s for s in splits if s in _SPLIT_RANK),
               key=lambda s: _SPLIT_RANK[s], default="train")

def resolve_cross_dataset_overlap(docs, merge: bool = True,
                                  verbose: bool = True):
    """Handle cases where multiple documents refer to the same source article. Make
    sure each canonical document is assigned to one split and identical-offset mentions
    are merged into one.
    """
    by_key = defaultdict(list)
    for d in docs:
        by_key[canonical_doc_key(d)].append(d)

    n_overlap = n_doc_merged = n_collapsed = n_multi_gold = n_reassigned = 0
    out_docs = []

    for key, dlist in by_key.items():
        split = _pick_split([d.split for d in dlist])
        n_reassigned += sum(1 for d in dlist if d.split != split)
        if len(dlist) > 1:
            n_overlap += 1
            n_doc_merged += len(dlist) - 1

        # Gather mentions across all docs for an article
        all_mentions = [m for d in dlist for m in d.mentions]
        sources = sorted({d.source for d in dlist})

        if merge:
            grouped = defaultdict(list)
            order = []
            for m in all_mentions:
                gkey = tuple(m.offsets) if m.offsets else ("__nooff__", id(m))
                if gkey not in grouped:
                    order.append(gkey)
                grouped[gkey].append(m)

            mentions = []
            for gkey in order:
                ms = grouped[gkey]
                base = ms[0]
                if len(ms) > 1:
                    n_collapsed += len(ms) - 1
                    for other in ms[1:]:
                        base.gold_synonyms |= other.gold_synonyms
                        base.gold_curies |= other.gold_curies
                        base.source_datasets |= other.source_datasets
                    base.gold_index = None
                    assign_gold_index(base)
                    if len(base.gold_curies) > 1:
                        n_multi_gold += 1
                mentions.append(base)
        else:
            mentions = all_mentions

        out_docs.append(DocumentExample(
            doc_id=key,
            mentions=mentions,
            source="+".join(sources),
            split=split,
        ))

    if verbose:
        print(f"[dedup] {n_overlap} articles in >= 2 datasets | "
              f"{n_doc_merged} docs merged | {n_collapsed} mentions"
              f" collapsed | {n_multi_gold} spans with multi-source gold | "
              f"{n_reassigned} split reassignments (leakage guard)")

    return out_docs


def report_statistics(examples: list[DocumentExample]) -> dict:
    """Print and return summary statistics on the corpus
    """
    n_docs = len(examples)
    n_mentions = sum(len(d.mentions) for d in examples)
    type_counts = defaultdict(int)
    n_with_candidates = 0
    n_gold_in_candidates = 0
    total_candidates = 0

    for doc in examples:
        for m in doc.mentions:
            type_counts[m.entity_type] += 1
            if m.candidates:
                n_with_candidates += 1
                total_candidates += len(m.candidates)
            if m.gold_index is not None:
                n_gold_in_candidates += 1

    stats = {
        "n_docs": n_docs,
        "n_mentions": n_mentions,
        "entity_types": dict(type_counts),
        "has_candidates_rate": n_with_candidates / n_mentions if n_mentions else 0,
        "gold_in_candidates_rate": n_gold_in_candidates / n_mentions if n_mentions else 0,
        "avg_candidates": total_candidates / n_with_candidates if n_with_candidates else 0,
    }

    print(f"Documents: {n_docs}")
    print(f"Mentions: {n_mentions}")
    print(f"Has candidates: {n_with_candidates}/{n_mentions} "
          f"({stats['has_candidates_rate']:.1%})")
    print(f"Gold in candidates: {n_gold_in_candidates}/{n_mentions} "
          f"({stats['gold_in_candidates_rate']:.1%})")
    print(f"Avg candidates per mention: {stats['avg_candidates']:.1f}")
    print("Entity type distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    return stats


# === Entity type classification ===

def _classify_entity_type(obj: list[str], obj_synonyms: set[str]) -> str:
    """Classify entity type (for the BioID corpus, just distinguishing Human
    vs Nonhuman Gene)
    """
    etype = _get_entity_type(obj)
    if etype == "Gene":
        if any(s.startswith("HGNC") for s in obj_synonyms):
            return "Human Gene"
        return "Nonhuman Gene"
    return "other" if etype == "unknown" else etype


# Namespace-to-semantic type map used in train.py
NS_TYPE = {
    "HGNC": "gene", "FPLX": "gene", "UP": "gene", "UPPRO": "gene", "IP": "gene",
    "PF": "gene", "CHEBI": "chem", "PUBCHEM": "chem", "DRUGBANK": "chem",
    "CHEMBL": "chem", "HMDB": "chem", "MESH": "MeSH", "NCIT": "NCIT",
    "DOID": "dis", "MONDO": "dis", "EFO": "trait", "HP": "phen", "GO": "proc",
    "CL": "cell", "BTO": "tissue", "TAXONOMY": "org",
}

# Map each BigBio dataset's entity types to names that match those used by
# BioIDBenchmarker._get_entity_type for making grouped evaluation tables

_BIGBIO_TYPE_MAP = {
    "bc5cdr": {"Chemical": "Small Molecule", "Disease": "Disease"},
    "nlmchem": {"Chemical": "Small Molecule"},
    "ncbi_disease": {
        "SpecificDisease": "Disease", "DiseaseClass": "Disease",
        "Modifier": "Disease", "CompositeMention": "Disease",
    },
    "gnormplus": {"Gene": "Gene", "FamilyName": "Gene"}
}

_UMLS_TUI_GROUP = {
    "T103": "Small Molecule", # Chemical
    "T033": "Disease", "T037": "Disease", # Finding, Injury or Poisoning
    "T038": "Biological Function", # Biologic Function (some disease-like)
    "T017": "Tissue/Organ", "T022": "Tissue/Organ", "T031": "Tissue/Organ",  # Anatomical Structure, Body System, Body Substance
    "T005": "Taxon", "T007": "Taxon", "T204": "Taxon", # Virus, Bacterium, Eukaryote
}

def classify_entity_type_bigbio(dataset: str, type_list, gold_synonyms) -> str:
    """Assigns an entity_type label for a BigBio mention.

    Params:
    -------
    dataset: str
        Name of the dataset (e.g., "bc5cdr")
    type_list: list[str]
        The mention's `type` column (a list)
    gold_synonyms: set[str]
        The mention's expanded gold curies

    Returns:
    --------
    Entity type label
    """
    if not isinstance(type_list, (list, tuple, set)):
        type_list = [type_list]

    # Group mentions in MedMentions via the semantic-group table _UMLS_TUI_GROUP.
    if dataset.startswith("medmentions"):
        for t in type_list:
            if t in _UMLS_TUI_GROUP:
                label = _UMLS_TUI_GROUP[t]
                if label == "Gene":
                    return ("Human Gene"
                            if any(s.startswith("HGNC") for s in gold_synonyms)
                            else "Nonhuman Gene")
                return label
        return "other"

    mapping = _BIGBIO_TYPE_MAP.get(dataset, {})

    # Loop over datasets's `type` column
    label = None
    for t in type_list:
        if t in mapping:
            label = mapping[t]
            break
    if label is None:
        label = "other"

    # Mirror _classify_entity_type's splitting of human and non-human genes
    if label == "Gene":
        if any(s.startswith("HGNC") for s in gold_synonyms):
            return "Human Gene"
        return "Nonhuman Gene"
    return label


# --- Register the old names of paths as aliases ---
# The corpus caches in caches/ were written while this package was still called
# "joint_disambig", so they store the class paths "joint_disambig.data.DocumentExample"
# and "joint_disambig.data.MentionExample" as literals.
import sys
import types

if "ascenda.data" not in sys.modules:
    sys.modules["ascenda.data"] = sys.modules[__name__]
compat_pkg = sys.modules.setdefault(
    "joint_disambig", types.ModuleType("joint_disambig"))
compat_pkg.data = sys.modules[__name__]
sys.modules.setdefault("joint_disambig.data", sys.modules[__name__])
