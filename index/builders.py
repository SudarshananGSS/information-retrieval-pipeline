"""
Unified Index Package Builder - Task 1
Creates a single on-disk package containing all three sub-indexes.
"""

from typing import List, Dict, Union, Tuple, Any, Optional
from collections import defaultdict
from utils.ngram import make_ngrams_tokens, make_ngrams_chars
from utils.positions import make_positions
from .io import dump

def create_all_indexes(
    tokenized_docs: List[List[str]], 
    index_path: str, 
    doc_ids: Optional[List[int]] = None
) -> None:
    """
    Build a unified index package containing all three sub-indexes in a single pass.
    
    Args:
        tokenized_docs: List of tokenized documents, each document is a list of tokens
        index_path: Path where the unified index package will be saved
        doc_ids: Optional list of document IDs. If None, uses sequential IDs (0, 1, 2, ...)
                 Must be same length as tokenized_docs if provided.
    """
    
    # ------------------------------------------------------------------
    # Validate and align document identifiers
    # ------------------------------------------------------------------
    if doc_ids is None:
        doc_ids = list(range(len(tokenized_docs)))
    elif len(doc_ids) != len(tokenized_docs):
        raise ValueError("doc_ids must be the same length as tokenized_docs")

    # ------------------------------------------------------------------
    # Data structures for the three sub-indexes
    # ------------------------------------------------------------------
    unified_index: Dict[Union[str, Tuple[str, ...]], set] = defaultdict(set)
    wildcard_index: Dict[str, set] = defaultdict(set)
    proximity_index: Dict[Union[str, Tuple[str, ...]], Dict[int, List[int]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    doc_lengths: Dict[int, int] = {}

    max_token_ngram = 3
    max_char_ngram = 3

    # ------------------------------------------------------------------
    # Build indexes in a single pass over documents
    # ------------------------------------------------------------------
    for tokens, doc_id in zip(tokenized_docs, doc_ids):
        doc_lengths[doc_id] = len(tokens)

        # --- Token n-grams & positions (unified + proximity) ---
        for n in range(1, max_token_ngram + 1):
            pos_map = make_positions(tokens, n)
            for term, positions in pos_map.items():
                unified_index[term].add(doc_id)
                proximity_index[term][doc_id] = positions

        # --- Character n-grams for wildcard matching ---
        for token in set(tokens):  # unique terms per document
            for n in range(1, max_char_ngram + 1):
                for ngram in make_ngrams_chars(token, n):
                    # Exclude the boundary-only "$" unigram
                    if n == 1 and ngram == "$":
                        continue
                    wildcard_index[ngram].add(token)

    # ------------------------------------------------------------------
    # Convert sets to sorted lists for deterministic output
    # ------------------------------------------------------------------
    unified_index_sorted: Dict[Union[str, Tuple[str, ...]], List[int]] = {
        term: sorted(doc_ids) for term, doc_ids in unified_index.items()
    }
    wildcard_index_sorted: Dict[str, List[str]] = {
        ngram: sorted(terms) for ngram, terms in wildcard_index.items()
    }
    proximity_index_sorted: Dict[
        Union[str, Tuple[str, ...]], Dict[int, List[int]]
    ] = {
        term: {doc: sorted(pos_list) for doc, pos_list in doc_map.items()}
        for term, doc_map in proximity_index.items()
    }

    # Meta information 
    N = len(tokenized_docs)
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 0.0
    package: Dict[str, Any] = {
        "__META__": {
            "N": N,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "version": "1.0",
            "ngrams_max": max_token_ngram,
            "char_ngrams_max": max_char_ngram,
        },
        "unified": unified_index_sorted,
        "wildcard": wildcard_index_sorted,
        "proximity": proximity_index_sorted,
    }
    
    # Save the unified package to disk
    dump(package, index_path)
