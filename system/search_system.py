#!/usr/bin/env python3
"""
Command-line Information Retrieval System - Task 4
Usage: python system/search_system.py <queries_json> <documents_jsonl> <run_output_json>
"""
import json, sys, pathlib, os
from collections import Counter

# Ensure repository root is on the path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import nltk
from nltk.corpus import wordnet

# Add parent directory to path for imports
# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.text_preprocessing import preprocess
from index.builders import create_all_indexes
from query_processing.query_process import process_query
from ranking.rankers import rank_documents
from utils.ngram import make_ngrams_tokens


# ---------------------------------------------------------------------------
# Optional non-ranking system optimisations. Toggle via the variables below
# without changing the command-line interface.
# ---------------------------------------------------------------------------
# When True, expands each query term with WordNet synonyms before searching.
USE_QUERY_EXPANSION = False

# When True, reduces the candidate document set before ranking by keeping only
# the top N documents with the highest term overlap with the query.
USE_CANDIDATE_PRUNING = True
MAX_CANDIDATES = 100  # effective only when USE_CANDIDATE_PRUNING is True


def _load_docs(path):
    """Load documents from JSONL file."""
    docs, raw = [], []
    seen_ids = set()
    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if "id" not in obj or "text" not in obj:
                            print(f"Warning: Line {line_num} missing required fields (id, text)")
                            continue
                        
                        doc_id = obj["id"]
                        if doc_id in seen_ids:
                            print(f"Warning: Duplicate doc_id {doc_id} found, keeping first occurrence")
                            continue
                        
                        seen_ids.add(doc_id)
                        raw.append(obj)
                        docs.append(obj["text"])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)
    return raw, docs

def _load_queries(path):
    """Load queries from JSON file."""
    try:
        with open(path, 'r') as f:
            queries = json.load(f)
        
        # Validate query format
        if not isinstance(queries, list):
            raise ValueError("Queries file must contain a JSON array")
        
        for i, query in enumerate(queries):
            if not isinstance(query, dict) or "qid" not in query or "query" not in query:
                raise ValueError(f"Query {i} missing required fields (qid, query)")
        
        return queries
    except Exception as e:
        print(f"Error loading queries: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python system/search_system.py <queries_json> <documents_jsonl> <run_output_json>")
        print("Example: python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json")
        sys.exit(1)
    
    queries_path, doc_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Load queries and documents
    print(f"Loading queries from: {queries_path}")
    queries = _load_queries(queries_path)
    print(f"Loaded {len(queries)} queries")
    
    print(f"Loading documents from: {doc_path}")
    raw_docs, raw_texts = _load_docs(doc_path)
    print(f"Loaded {len(raw_docs)} documents")
        
    # Build unified index package
    # cache_dir = pathlib.Path(__file__).parent.parent / "cache"
    
    doc_ids = [int(d["id"]) for d in raw_docs]
    tokenised_docs = preprocess([d["text"] for d in raw_docs])

    cache_dir = pathlib.Path(__file__).resolve().parents[1] / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    index_path = cache_dir / "unified_package.pkl.gz"

    print("Building index package...")
    create_all_indexes(tokenised_docs, str(index_path), doc_ids=doc_ids)

    id_to_tokens = {doc_id: toks for doc_id, toks in zip(doc_ids, tokenised_docs)}

    results = []
    for q in queries:
        raw_query = q["query"]
        q_tokens = preprocess([raw_query])[0]
        
        search_query = raw_query
        if USE_QUERY_EXPANSION:
            expansion_terms = set()
            for tok in q_tokens:
                for syn in wordnet.synsets(tok):
                    for lemma in syn.lemma_names():
                        lemma = lemma.replace("_", " ")
                        if lemma and lemma.lower() != tok.lower():
                            expansion_terms.add(lemma)
            if expansion_terms:
                search_query = raw_query + " " + " ".join(sorted(expansion_terms))
                q_tokens = preprocess([search_query])[0]

        candidate_ids = sorted(process_query(search_query, str(index_path)))

        if USE_CANDIDATE_PRUNING and candidate_ids:
            q_uni = set(q_tokens)
            q_bi = set(make_ngrams_tokens(q_tokens, 2))
            scored = []
            for doc_id in candidate_ids:
                tokens = id_to_tokens[doc_id]
                uni = set(tokens)
                bi = set(make_ngrams_tokens(tokens, 2))
                overlap = len(q_uni & uni) + len(q_bi & bi)
                scored.append((doc_id, overlap))
            scored.sort(key=lambda x: (-x[1], x[0]))
            candidate_ids = [doc_id for doc_id, _ in scored[:MAX_CANDIDATES]]
        
        candidate_docs = [id_to_tokens[d] for d in candidate_ids]
        method = "default"  # or "bm25", "semantic", "default", "hybrid"
        ranked, scores = rank_documents(q_tokens, candidate_docs, candidate_ids, str(index_path), method=method)

        TOP_K = 10
        ranked_top = [int(d) for d in ranked[:TOP_K]]
        scores_top = [float(s) for s in scores[:TOP_K]]

        results.append({
            "qid": q["qid"],
            "doc_ids": ranked_top,
            "scores": scores_top
        })

    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} query results to {output_file}")


if __name__ == "__main__":
    main()
