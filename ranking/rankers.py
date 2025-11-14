"""Optimized semantic ranking with keyword stuffing penalties"""
import numpy as np
from typing import List, Tuple
import pathlib
import json
import sys
from typing import List, Tuple
import numpy as np

import warnings
from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Ensure repository root is on sys.path when running as a script
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from index.builders import create_all_indexes
from utils.text_preprocessing import preprocess
from utils.tfidf import tfidf_variants

def rank_documents(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    inverted_index_path: str,
    method: str = "default"
) -> Tuple[List[int], List[float]]:
    """
    Rank documents using multi-algorithm approach.
    
    Args:
        query_toks: Tokenized and cleaned query terms
        candidate_docs: List of tokenized and cleaned candidate documents
        doc_ids: Document IDs corresponding to candidate_docs
        inverted_index_path: Path to the unified inverted index
        method: Ranking method ("default", ...)
        
    Returns:
        Tuple of (ranked_doc_ids, ranking_scores) - ALL candidates ranked by relevance
    """
    if len(candidate_docs) != len(doc_ids):
        raise ValueError("candidate_docs and doc_ids must align")
    if not candidate_docs:
        return [], []

    if method == "tfidf":
        all_docs = candidate_docs + [query_toks]
        tfidf_matrix, _ = tfidf_variants(all_docs, tf_mode="raw")
        tfidf_matrix = np.nan_to_num(tfidf_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        doc_matrix, query_vec = tfidf_matrix[:-1], tfidf_matrix[-1]
        with np.errstate(all="ignore"):
            scores = doc_matrix @ query_vec
        ranked_idx = np.argsort(-scores)
        ranked_ids = [doc_ids[i] for i in ranked_idx]
        return ranked_ids, scores[ranked_idx].tolist()
    
    method = "hybrid" if method == "default" else method

    docs_plus_query = candidate_docs + [query_toks]

    lexical = np.zeros(len(candidate_docs), dtype=float)
    if method in {"bm25", "tfidf", "hybrid"}:
        tf_mode = "bm25" if method != "tfidf" else "raw"
        tfidf_matrix, _ = tfidf_variants(docs_plus_query, tf_mode=tf_mode)
        doc_matrix, query_vec = tfidf_matrix[:-1], tfidf_matrix[-1]
        lexical = doc_matrix @ query_vec

    semantic = np.zeros(len(candidate_docs), dtype=float)
    if method in {"semantic", "hybrid"}:
        from utils.embeddings import semantic_vector
        doc_vecs = semantic_vector(candidate_docs)
        query_vec = semantic_vector([query_toks])[0]
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        q_norm = np.linalg.norm(query_vec)
        denom = doc_norms * q_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            semantic = np.divide(doc_vecs @ query_vec, denom, out=np.zeros_like(denom), where=denom != 0)

    if method in {"bm25", "tfidf"}:
        final_scores = lexical
    elif method == "semantic":
        final_scores = semantic
    elif method == "hybrid":
        lex_norm = lexical / lexical.max() if np.max(lexical) > 0 else lexical
        sem_norm = semantic / semantic.max() if np.max(semantic) > 0 else semantic
        final_scores = 0.7 * lex_norm + 0.3 * sem_norm
    else:
        raise ValueError("Unknown ranking method")

    ranked = sorted(zip(doc_ids, final_scores), key=lambda x: (-x[1], x[0]))
    ranked_ids = [d for d, _ in ranked]
    scores = [float(s) for _, s in ranked]
    return ranked_ids, scores

def _pearson(pred: List[float], truth: List[float]) -> float:
    a = np.asarray(pred, dtype=float)
    b = np.asarray(truth, dtype=float)

    # Not enough points to define correlation
    if a.size < 2 or b.size < 2:
        return 0.0

    # If either vector is constant, define r = 0.0 
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0

    # Compute correlation without noisy runtime warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(r) else float(r)


def _load_dev_corpus() -> Tuple[List[int], List[str]]:
    doc_ids: List[int] = []
    texts: List[str] = []
    with open("data/dev/documents.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(int(obj["id"]))
            texts.append(obj["text"])
    return doc_ids, texts


if __name__ == "__main__":
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

    doc_ids, raw_docs = _load_dev_corpus()
    tokenised_docs = preprocess(raw_docs)
    index_path = "task3_dev_index.pkg.gz"
    create_all_indexes(tokenised_docs, index_path, doc_ids=doc_ids)

    with open("data/dev/queries.json", "r", encoding="utf-8") as f:
        queries = {q["qid"]: q["query"] for q in json.load(f)}

    with open("data/dev/relevance_judge.json", "r", encoding="utf-8") as f:
        judges = {j["qid"]: j for j in json.load(f)}

    id2doc = {doc_id: tok for doc_id, tok in zip(doc_ids, tokenised_docs)}

    methods = ["bm25", "tfidf", "semantic", "hybrid", "default"]

    print("Evaluating ranking methods on dev set (Pearson correlation)...")

    method_col = max(len("Method"), max(len(x) for x in methods))
    score_hdr = "Pearson (avg)"
    score_col = max(len(score_hdr), 12)

    print(f"{'Method':<{method_col}}  {score_hdr:>{score_col}}")
    print("-" * (method_col + 2 + score_col))
    
    for m in methods:
        scores = []
        for qid, info in judges.items():
            q_tokens = preprocess([queries[qid]])[0]
            candidates = info["ground_truth_order"]
            docs = [id2doc[d] for d in candidates]
            ranked, pred = rank_documents(q_tokens, docs, candidates, index_path, method=m)
            gold = [info["relevance_scores"].get(str(d), 0.0) for d in ranked]
            scores.append(_pearson(pred, gold))
        mean_score = float(np.mean(scores)) if scores else 0.0
        print(f"{m:<{method_col}}  {mean_score:>{score_col}.4f}")
