from typing import Set
from .detection import detect_query_type
from .boolean import process_boolean_query
from .wildcard import process_wildcard_query
from .proximity import process_proximity_query

def convert_natural_language(nl_query: str) -> str:
    """
    Convert already-cleaned natural language to a Boolean-OR query.

    Args:
        nl_query: Cleaned input string (whitespace-separated tokens).

    Returns:
        Boolean query string with OR operators (e.g., "a OR b OR c").
    """
    # Assume nl_query is already cleaned/lowercased in test cases
    # Split on whitespace and keep tokens as-is
    tokens = [tok for tok in nl_query.split() if tok]
    return " OR ".join(tokens)

def process_query(query: str, index_path: str) -> Set[int]:
    """
    Main query processing with automatic type detection.

    Args:
        query: Structured or natural-language query (already cleaned).
        index_path: Path to the unified index package (Task 1).

    Returns:
        Set of candidate document IDs.
    """
    qtype = detect_query_type(query)

    if qtype == "boolean":
        return process_boolean_query(query, index_path)
    elif qtype == "wildcard":
        return process_wildcard_query(query, index_path)
    elif qtype == "proximity":
        return process_proximity_query(query, index_path)
    else:  # natural_language
        boolean_query = convert_natural_language(query)
        return process_boolean_query(boolean_query, index_path) if boolean_query else set()
