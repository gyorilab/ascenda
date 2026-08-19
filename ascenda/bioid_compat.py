"""Standalone replacements for the helpers this package used to borrow from
gilda/benchmarks/bioid_evaluation.py.
"""
from typing import Collection, List

import pystow

MODULE = pystow.module('gilda', 'biocreative')
URL = ('https://biocreative.bioinformatics.udel.edu/media/store/files/2017/'
       'BioIDtraining_2.tar.gz')


def normalize_id(curie):
    """Convert ID into standardized format, f'{namespace}:{id}'.
    """
    if curie.startswith('CVCL'):
        return curie.replace('_', ':')
    split_id = curie.split(':', maxsplit=1)
    if split_id[0] == 'Uberon':
        return split_id[1]
    if split_id[0] == 'Uniprot':
        return f'UP:{split_id[1]}'
    if split_id[0] in ['GO', 'CHEBI']:
        return f'{split_id[0]}:{split_id[0]}:{split_id[1]}'
    return curie


def normalize_ids(curies: str) -> List[str]:
    return [normalize_id(y) for y in curies.split('|')]


def get_entity_type(groundings: Collection[str]) -> str:
    """Get entity type based on entity groundings of text in corpus.
    """
    if any(
        grounding.startswith('NCBI gene') or grounding.startswith('UP')
        for grounding in groundings
    ):
        return 'Gene'
    elif any(grounding.startswith('Rfam') for grounding in groundings):
        return 'miRNA'
    elif any(grounding.startswith('CHEBI') or grounding.startswith('PubChem')
             for grounding in groundings):
        return 'Small Molecule'
    elif any(grounding.startswith('GO') for grounding in groundings):
        return 'Cellular Component'
    elif any(
        grounding.startswith('CVCL') or grounding.startswith('CL')
        for grounding in groundings
    ):
        return 'Cell types/Cell lines'
    elif any(grounding.startswith('UBERON') for grounding in groundings):
        return 'Tissue/Organ'
    elif any(grounding.startswith('NCBI taxon') for grounding in groundings):
        return 'Taxon'
    else:
        return 'unknown'


def get_benchmarker(grounder=None, equivalences=None):
    """Construct a BioIDBenchmarker. Corpus-rebuild path only.
    """
    try:
        from benchmarks.bioid_evaluation import BioIDBenchmarker
    except ImportError as e:
        raise ImportError(
            "BioIDBenchmarker is only needed to rebuild a corpus from "
            "scratch. To run the demo or evaluate a trained model, pass "
            "merged_cache=caches/corpus_cache/final_merged.pkl to "
            "load_corpus()."
        ) from e
    return BioIDBenchmarker(
        grounder=grounder,
        equivalences=equivalences or {},
    )
