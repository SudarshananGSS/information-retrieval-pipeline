"""TF-IDF variants (Task A-3)."""
import math, numpy as np
from typing import List, Dict, Tuple

def tfidf_variants(
        docs: List[List[str]],
        tf_mode: str = "raw",
        k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Return a TF‑IDF matrix with selectable term‑frequency variants.

    Args:
        docs : list of list of str
            Tokenised documents.
        tf_mode : {"raw", "log", "bm25"}
            Variant of term frequency weighting to apply.
        k : float, optional
            Saturation parameter for the BM25-style TF, by default ``1.2``.

    Returns:
        tuple[np.ndarray, dict[str, int]]
            Matrix of shape [n_docs, vocab_size] and vocabulary mapping.

    Notes:
        tf_mode='raw' uses document-length normalised frequencies. 'log'
        applies 1 + log(tf) when tf > 0. 'bm25' uses the BM25-style
        formulation (k + 1) * tf / (k + tf).
    """

    if not isinstance(docs, list) or any(not isinstance(d, list) for d in docs):
        raise TypeError("docs must be a list of token lists")
    if any(not isinstance(token, str) for doc in docs for token in doc):
        raise TypeError("every token must be a string")    
    if tf_mode not in {"raw", "log", "bm25"}:
        raise ValueError("tf_mode must be 'raw', 'log', or 'bm25'")
    if tf_mode == "bm25" and k <= 0:
        raise ValueError("k must be positive for bm25 mode")
    if len(docs) == 0:
        return np.zeros((0, 0)), {}

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------
    #tokens = sorted({token for doc in docs for token in doc})
    #vocabulary: Dict[str, int] = {token: idx for idx, token in enumerate(tokens)}
    vocabulary = {}
    for doc in docs:
        for token in doc:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    n_docs = len(docs)
    n_terms = len(vocabulary)
    
    tf_counts = np.zeros((n_docs, n_terms), dtype=float)
    for row, doc in enumerate(docs):
        for token in doc:
            tf_counts[row, vocabulary[token]] += 1.0

    # ------------------------------------------------------------------
    # Term-frequency weighting
    # ------------------------------------------------------------------
    if tf_mode == "raw":
        # Normalise by document length
        doc_lengths = np.array([len(doc) for doc in docs], dtype=float)

        # Avoid division by zero               
        doc_lengths = np.where(doc_lengths > 0, doc_lengths, 1.0)
        tf = tf_counts / doc_lengths[:, None]
    
    elif tf_mode == "log":
        # Apply 1 + log(tf) for each term frequency
        tf = np.zeros_like(tf_counts)
        np.log(tf_counts, out=tf, where=tf_counts > 0)
        tf[tf_counts > 0] += 1
    
    else: # BM25-style term frequency
        # Apply BM25-style term frequency
        tf = (k + 1.0) * tf_counts / (k + tf_counts)

    # ------------------------------------------------------------------
    # Inverse document frequency and final TF-IDF
    # ------------------------------------------------------------------

    # Count non-zero term frequencies across documents    
    doc_freq = np.count_nonzero(tf_counts, axis=0)
    
    # Guard against divide-by-zero when a term appears in no documents
    doc_freq = np.where(doc_freq > 0, doc_freq, 1)

    idf = np.log(n_docs / doc_freq)
    
    tfidf_matrix = tf * idf
    return tfidf_matrix, vocabulary
