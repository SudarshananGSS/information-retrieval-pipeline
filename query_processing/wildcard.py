from typing import Set, Optional
import re
from index.access import find_wildcard_matches, get_posting_list
from .detection import _validate_wildcard

def process_wildcard_query(pattern: str, index_path: str) -> Set[int]:
    """
    Process wildcard queries using character n-grams.
    
    Args:
        pattern: Wildcard pattern (e.g., "climat*", "*tion", "learn*ing")
        index_path: Path to the unified index package
        
    Returns:
        Set of document IDs containing terms matching the pattern
    """
    
    _validate_wildcard(pattern)

    def ngrams_for_segment(seg: str, prefix: bool, suffix: bool) -> Set[str]:
        padded = f"{'$' if prefix else ''}{seg}{'$' if suffix else ''}"
        ngrams = set()
        length = len(padded)
        for n in range(1, min(3, length) + 1):
            for i in range(length - n + 1):
                ng = padded[i : i + n]
                if ng == "$":
                    continue
                ngrams.add(ng)
        return ngrams

    segments = pattern.split("*")
    ngram_sets = []
    for i, seg in enumerate(segments):
        if not seg:
            continue
        ngram_sets.append(
            ngrams_for_segment(seg, prefix=(i == 0), suffix=(i == len(segments) - 1))
        )

    if not ngram_sets:
        return set()

    term_candidates: Optional[Set[str]] = None
    for ngrams in ngram_sets:
        for ng in ngrams:
            matches = set(find_wildcard_matches(ng, index_path))
            if term_candidates is None:
                term_candidates = matches
            else:
                term_candidates &= matches
            if not term_candidates:
                return set()

    segments_escaped = [re.escape(seg) for seg in segments]
    regex = ".*".join(segments_escaped)
    if not pattern.startswith("*"):
        regex = "^" + regex
    if not pattern.endswith("*"):
        regex = regex + "$"
    pat = re.compile(regex)

    result: Set[int] = set()
    for term in (term_candidates or set()):
        if not pat.search(term):
            continue
        result.update(get_posting_list(term, index_path))
    return result