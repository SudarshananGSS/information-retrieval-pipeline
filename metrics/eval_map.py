#!/usr/bin/env python3
"""
MAP Evaluation Script for Task 4
Evaluates all JSON run files in ./runs/ against development relevance judgments.
Usage: python metrics/eval_map.py
"""
from __future__ import annotations
import json
import pathlib
import sys
from collections import defaultdict

from typing import Dict, Iterable, List, Sequence, Set

def _load_judgments(path: pathlib.Path) -> Dict[str, Set[int]]:
    """Load relevance judgments into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    judgments: Dict[str, Set[int]] = {}
    for entry in data:
        rel = {
            int(doc_id)
            for doc_id, score in entry.get("relevance_scores", {}).items()
            if score > 0
        }
        judgments[entry["qid"]] = rel
    return judgments


def _average_precision(ranked: Sequence[int], relevant: Set[int]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant)


def _evaluate_run(run_path: pathlib.Path, judgments: Dict[str, Set[int]]) -> float:
    with open(run_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    ap_scores: List[float] = []
    for entry in run_data:
        qid = entry.get("qid")
        if qid not in judgments:
            continue
        # ranked = [int(r["doc_id"]) for r in entry.get("results", [])]
        if "doc_ids" in entry:
            ranked = [int(x) for x in entry["doc_ids"]]
        else:
            # backward compatibility 
            ranked = [int(r["doc_id"]) for r in entry.get("results", [])]

        ap_scores.append(_average_precision(ranked, judgments[qid]))
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0



def main() -> None:
    runs_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("runs")
    judge_path = (
        pathlib.Path(sys.argv[2])
        if len(sys.argv) > 2
        else pathlib.Path("data/dev/relevance_judge.json")
    )

    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    judgments = _load_judgments(judge_path)

    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        print("No run files found.")
        return

    name_w = max((len(p.stem) for p in run_files), default=3)
    print("\nMAP Summary (dev set)")
    print(f"{'Run'.ljust(name_w)}  {'MAP':>8}")
    print(f"{'-'*name_w}  {'-'*8}")

    for run_file in run_files:
        map_score = _evaluate_run(run_file, judgments)
        print(f"{run_file.stem.ljust(name_w)}  {map_score:>8.4f}")
        


if __name__ == "__main__":  
    main()