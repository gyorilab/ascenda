"""Use a pretrained PubMedBERT model from HuggingFace to embed gilda candidates.
"""
import os
import pickle
from typing import Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = (
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

# Human-readable namespace labels for embedding context
NAMESPACE_LABELS = {
    "HGNC": "gene", "UP": "protein", "FPLX": "protein family",
    "CHEBI": "chemical", "PUBCHEM": "chemical", "GO": "gene ontology term",
    "MESH": "biomedical concept", "DOID": "disease", "HP": "phenotype",
    "EFO": "experimental factor", "CL": "cell type", "BTO": "tissue",
    "NCIT": "NCI concept", "TAXONOMY": "organism", "IP": "InterPro domain",
    "PF": "Pfam domain", "MONDO": "disease"
}

# Common taxonomy IDs to species names
ORGANISM_NAMES = {
    "9606": "human", "10090": "mouse", "10116": "rat",
    "9913": "cow", "9615": "dog", "9823": "pig",
    "7955": "zebrafish", "7227": "Drosophila", "6239": "C. elegans",
    "3702": "Arabidopsis", "559292": "yeast", "4932": "yeast",
}


class DescriptionLookup:
    """Pre-built lookup table for entity descriptions from Gilda's term table.

    Scans the grounder's entries once at construction time to collect:
    - Longest 'name' status text for each (db, id) —> e.g., "estrogen receptor 1"
        for HGNC:3467
    - Longest synonym text for UP entries —> e.g., "Cathepsin D" for UP:P07339

    This avoids scanning millions of entries per candidate at embedding time.
    """

    def __init__(self, grounder):
        self.names = {}  # (db, id): longest name that isn't entry_name
        self.up_synonyms = {}  # (UP, id): longest synonym that isn't entry_name

        for norm_text, terms in grounder.entries.items():
            for t in terms:
                key = (t.db, t.id)
                if t.status == "name":
                    prev = self.names.get(key)
                    if prev is None or len(t.text) > len(prev):
                        self.names[key] = t.text

                # Also collect longest synonym text for UP entries
                # (these often contain descriptive protein names)
                if t.db == "UP" and t.text != t.entry_name:
                    prev = self.up_synonyms.get(key)
                    if prev is None or len(t.text) > len(prev):
                        self.up_synonyms[key] = t.text

    def get_description(self, term) -> Optional[str]:
        """Get the best available description for a term. Returns the longest name
        that differs from entry_name. For UP entries, falls back to the longest
        synonym if no name entry exists.
        """
        key = (term.db, term.id)
        name = self.names.get(key)
        if name and name != term.entry_name:
            return name
        if term.db == "UP":
            syn = self.up_synonyms.get(key)
            if syn and syn != term.entry_name:
                return syn
        return None


_description_lookup: Optional[DescriptionLookup] = None


def get_description_lookup(grounder) -> DescriptionLookup:
    global _description_lookup
    if _description_lookup is None:
        _description_lookup = DescriptionLookup(grounder)
    return _description_lookup


def build_embedding_text(term, grounder=None) -> str:
    """Build a rich text string for embedding a Gilda Term. For example,
    instead of embedding a candidate like HGNC:3467 by only its entry_name
    "ESR1", embed a string that includes this name plus a long-form name
    and a namespace label like "ESR1, estrogen receptor 1 (gene)", which takes
    the form: "{entry_name}, {long_form_name} ({namespace_label})", If no
    grounder is provided, just returns entry_name.

    Params:
    -------
    term :
        Gilda Term
    grounder :
        Gilda Grounder instance (optional)

    Returns:
    --------
    str : rich text string to be embedded
    """
    name = term.entry_name or term.text or term.id
    if grounder is None:
        return name

    lookup = get_description_lookup(grounder)
    parts = [name]

    # Add descriptive name if available
    desc = lookup.get_description(term)
    if desc:
        parts.append(desc)

    # Build namespace label with species for organism-specific entries
    ns_label = NAMESPACE_LABELS.get(term.db)
    if term.organism and term.organism in ORGANISM_NAMES:
        species = ORGANISM_NAMES[term.organism]
        if ns_label:
            return ", ".join(parts) + f" ({species} {ns_label})"
        return ", ".join(parts) + f" ({species})"
    elif ns_label:
        return ", ".join(parts) + f" ({ns_label})"
    return ", ".join(parts)


class EntityEmbedder:
    """Embeds entity mentions and candidates using a pretrained LLM.

    Params:
    -------
    model_name:
        HuggingFace model identifier. Default is PubMedBERT (now known as BiomedBERT).
    device:
        'cpu', 'mps', or 'cuda'.
    cache_path:
        Optional embedding cache location
    grounder:
        Optional Gilda Grounder instance for looking up full entity names
        from the term table. If provided, allows for embedding richer entity texts
        (e.g., "ESR1, estrogen receptor 1 (gene)" instead of "ESR1").
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_path: Optional[str] = None,
        grounder=None,
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self.cache_path = cache_path
        self.grounder = grounder
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self._cache = pickle.load(f)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 32,
                    max_length: int = 128) -> np.ndarray:
        """Embed a list of texts. Returns (N, hidden_dim) array via [CLS] pooling.
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            out = self.model(**tok)
            all_vecs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.concatenate(all_vecs, axis=0)

    def embed_text(self, text: str, max_length: int = 128) -> np.ndarray:
        """Embed a single text string.
        """
        return self.embed_texts([text], max_length=max_length)[0]

    def embed_candidate(self, term) -> np.ndarray:
        """Embed a Gilda Term and cache its embedding by the key "(db, id)".

        If a grounder was provided at init, uses rich text (entry_name +
        full name + namespace label). Otherwise, uses entry_name only.
        """
        key = (term.db, term.id)
        if key not in self._cache:
            text = build_embedding_text(term, self.grounder)
            self._cache[key] = self.embed_text(text)
        return self._cache[key]

    def embed_candidates(self, scored_matches: list) -> np.ndarray:
        """Embed all candidates in a ScoredMatch list. Returns (N, hidden_dim)
        array.
        """
        vecs = [self.embed_candidate(m.term) for m in scored_matches]
        return np.stack(vecs)

    def save_cache(self):
        """Save the embedding cache to disk
        """
        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f)

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size
