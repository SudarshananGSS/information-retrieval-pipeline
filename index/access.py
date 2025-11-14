"""
Unified Index Package Access Functions - Task 1
Provides O(1) access to all three sub-indexes from a single package.
"""

from typing import List, Union, Tuple, Dict, Any
from .io import load
import re

# Global cache to store loaded packages for O(1) access
_package_cache: Dict[str, Dict[str, Any]] = {}

def _load_package(index_path: str) -> Dict[str, Any]:
    """Load and cache the unified index package."""
    if index_path not in _package_cache:
        _package_cache[index_path] = load(index_path)
    return _package_cache[index_path]

def get_posting_list(term: Union[str, Tuple[str, ...]], index_path: str) -> List[int]:
    """
    Returns the posting list for a unigram or n-gram from the unified sub-index.
    
    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        index_path: Path to the unified index package
        
    Returns:
        List of document IDs (sorted, deduplicated) containing the term
    """
    package = _load_package(index_path)
    unified_index = package.get("unified", {})

    postings = unified_index.get(term)
    if postings is None:
        return []
    # Return a copy to prevent external mutation
    return list(postings)

def find_wildcard_matches(ngram: str, index_path: str) -> List[str]:
    """
    Returns the terms for an character n-grams.
    
    Args:
        ngram: an n-gram (e.g., "$cl", "on$")
        index_path: Path to the unified index package
        
    Returns:
        List of matching terms (sorted lexicographically, deduplicated)
    """
    if ngram == "$":
        # Edge case: boundary-only pattern is meaningless
        return []
    package = _load_package(index_path)
    wildcard_index = package.get("wildcard", {})
    
    terms = wildcard_index.get(ngram)
    if terms is None:
        return []
    return list(terms)



def get_term_positions(term: Union[str, Tuple[str, ...]], doc_id: int, index_path: str) -> List[int]:
    """
    Returns the position list for a unigram or n-gram in a specific document.
    
    Args:
        term: A string (unigram) or tuple of strings (n-gram)
        doc_id: Document ID
        index_path: Path to the unified index package
        
    Returns:
        List of positions (0-based, sorted, deduplicated) where the term appears in the document
    """
    package = _load_package(index_path)
    proximity_index = package.get("proximity", {})
    
    doc_positions = proximity_index.get(term, {})
    positions = doc_positions.get(doc_id)
    if positions is None:
        return []
    return list(positions)
    
