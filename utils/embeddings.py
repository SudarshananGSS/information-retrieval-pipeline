"""Light-weight GloVe loader & semantic aggregation (Task A-4)."""
import io, zipfile, urllib.request, pathlib
from typing import Dict, List, Tuple

import numpy as np
from .tfidf import tfidf_variants


# ----------------------------------------------------------------------
# 1.  Load a 100-d slice of GloVe (or from cache) and add a random <unk>
# ----------------------------------------------------------------------
_GLOVE_URL  = ("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.100d.txt")
_CACHE      = pathlib.Path.home() / ".cache" / "ir_glove_100.txt"


def _ensure_glove() -> Dict[str, np.ndarray]:
    if not _CACHE.exists():
        url, fname = _GLOVE_URL
        with urllib.request.urlopen(url) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                _CACHE.write_text(
                    zf.read(fname).decode("utf-8"),
                    encoding="utf-8",
                    newline="\n",
                )
    vocab = {}
    with _CACHE.open("r", encoding="utf-8", newline="\n") as f:
        for line in f:
            word, *vec = line.strip().split()
            vocab[word] = np.asarray(vec, dtype=float)

    # deterministic random <unk> – keeps load fast
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim)

    return vocab


_WORD_VEC: Dict[str, np.ndarray] = _ensure_glove()
_DIM: int = next(iter(_WORD_VEC.values())).shape[0]


# ----------------------------------------------------------------------
# 2.  Public helper
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return token itself if in vocab, else '<unk>'."""
    return tok if tok in _WORD_VEC else "<unk>"


# ----------------------------------------------------------------------
# 3.  Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "mean") -> np.ndarray:
    """
    Generate document embeddings by aggregating word embeddings.

    Args:
    ----------
    docs   : list of token lists
    method : "mean" | "max" | "sum" | "tfidf_weighted" | "meanmax"
    
    Returns:
    ----------
    np.ndarray
        Document vectors of shape [n_docs, embedding_dim].
    """
    
    # Validate that docs is a list of lists
    if not isinstance(docs, list) or any(not isinstance(d, list) for d in docs):
        raise TypeError("docs must be a list of token lists")

    # Ensure the method provided is valid
    valid = {"mean", "max", "sum", "tfidf_weighted", "meanmax"}
    if method not in valid:
        raise ValueError("method must be one of 'mean', 'max', 'sum', 'tfidf_weighted', 'meanmax'")

    # If no documents are provided, return an empty array
    if len(docs) == 0:
        embedding_dim = _DIM if method != "meanmax" else 2 * _DIM
        return np.zeros((0, embedding_dim), dtype=float)

    # If tfidf_weighted is requested, compute the TF-IDF matrix
    if method == "tfidf_weighted":
        
        docs_for_tfidf = [
            [tok if tok in _WORD_VEC else "<unk>" for tok in doc]
            for doc in docs
        ]
        tfidf_matrix, vocabulary = tfidf_variants(docs_for_tfidf, tf_mode="raw")

        # tfidf_matrix, vocabulary = tfidf_variants(docs)
        word_embeddings = np.zeros((len(vocabulary), _DIM), dtype=float)
        
        # Fill the word embeddings matrix with GloVe vectors
        for token, token_index in vocabulary.items():
            word_embeddings[token_index] = _WORD_VEC[_key(token)]
        
        weighted_sum = tfidf_matrix @ word_embeddings                        
        denom = tfidf_matrix.sum(axis=1, keepdims=True)                     
        # avoid divide-by-zero; rows with denom==0 should stay all-zeros
        doc_vectors = np.divide(weighted_sum, denom, out=np.zeros_like(weighted_sum), where=denom>0)
        return doc_vectors
       # return tfidf_matrix @ word_embeddings

    # For other methods, compute the document vectors
    document_vectors = []
    for tokens in docs:
        if tokens:
            token_embeddings = np.vstack([_WORD_VEC[_key(token)] for token in tokens])
            if method == "mean":
                doc_vector = token_embeddings.mean(axis=0)
            elif method == "max":
                doc_vector = token_embeddings.max(axis=0)
            elif method == "sum":
                doc_vector = token_embeddings.sum(axis=0)
            elif method == "meanmax":
                doc_vector = np.concatenate([token_embeddings.mean(axis=0), token_embeddings.max(axis=0)])
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            doc_vector = np.zeros(2 * _DIM if method == "meanmax" else _DIM, dtype=float)
        
        document_vectors.append(doc_vector)

    return np.stack(document_vectors)
