# Information Retrieval Pipeline

A modular Python information retrieval system with:
- unified indexing (token, wildcard, and positional indexes),
- query processing (boolean, wildcard, proximity, natural-language routing),
- multiple rankers (BM25/TF-IDF/semantic/hybrid),
- batch run generation and MAP evaluation.

## Project Structure

```
data/dev/                  # development dataset (documents, queries, relevance judgments)
index/                     # unified index build/load/access
query_processing/          # query type detection + query executors
ranking/                   # ranking algorithms and dev ranking benchmark script
system/search_system.py    # end-to-end batch retrieval CLI
metrics/eval_map.py        # MAP evaluator for run files
runs/                      # output run files (*.json)
cache/                     # generated compressed index package
test_sanity/               # smoke tests for Tasks 1-4 interfaces
```

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`:
  - `numpy`
  - `nltk`
  - `beautifulsoup4`

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Note: NLTK resources (`punkt`, `punkt_tab`, `wordnet`, `stopwords`) are downloaded automatically by runnable scripts.

## Quick Start

Run the full retrieval pipeline on the dev set:

```bash
python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_default.json
```

Evaluate MAP across all run files in `runs/`:

```bash
python metrics/eval_map.py
```

Evaluate ranking methods on the dev set (Pearson correlation against relevance scores):

```bash
python ranking/rankers.py
```

Run smoke tests:

```bash
python test_sanity/check_submission.py
```

## Pipeline Overview

1. Load documents and queries.
2. Preprocess document text (`utils/text_preprocessing.py`).
3. Build a single compressed index package (`index/builders.py`) containing:
   - `unified` postings,
   - `wildcard` character n-gram index,
   - `proximity` positional index,
   - metadata (`N`, `doc_lengths`, `avgdl`).
4. Route each query via `query_processing/query_process.py`:
   - boolean (`AND`, `OR`, `NOT`, parentheses, quoted phrases),
   - wildcard (`*` pattern),
   - proximity (`NEAR/k`),
   - natural-language (converted to OR expression).
5. Optional candidate pruning and query expansion (in `system/search_system.py`).
6. Rank candidates with `ranking/rankers.py`.
7. Write run file JSON (`qid`, `doc_ids`, `scores`).

## Ranking Methods

`ranking.rankers.rank_documents(..., method=...)` supports:
- `bm25`
- `tfidf`
- `semantic`
- `hybrid`
- `default` (mapped internally to `hybrid`)

In `system/search_system.py`, ranking mode is currently set by the `method` variable inside the query loop.

## Query Syntax

Query detection is case-sensitive for operators:
- Boolean: `AND`, `OR`, `NOT`, `(`, `)`, quoted phrase (up to 3 tokens)
- Wildcard: single pattern string containing `*` (no boolean/proximity operators)
- Proximity: `<term_or_phrase> NEAR/k <term_or_phrase>`
- Natural-language: plain text (converted to `token1 OR token2 OR ...`)

Malformed query structures raise `ValueError` during validation in `query_processing/detection.py`.

## Run Output Format

`system/search_system.py` writes:

```json
[
  {
    "qid": "q1",
    "doc_ids": [12, 9, 31],
    "scores": [1.234, 0.992, 0.774]
  }
]
```

`metrics/eval_map.py` also supports a backward-compatible format containing `results: [{doc_id: ...}]`.

## Notes

- Indexes are stored as gzip+pickle packages via `index/io.py`.
- Index access uses in-process caching in `index/access.py`.
- Default top-k output in the batch system is top 10 documents per query.
