"""Token-position mapping (Task A-2)."""
from collections import defaultdict
from typing import Dict, List, Union, Tuple

def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:
    """
    Build an index of token positions for all n-grams in the sequence.

    Args:
        tokens: A list of string tokens to extract n-grams from.
        n: The size of the n-grams to generate (must be >= 1).

    Returns:
        A dictionary mapping each unique n-gram to the list of 
        starting indices where it occurs in the token sequence.
    """
    
    # Ensure input is a list of strings
    if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
        raise TypeError("tokens must be a list of strings")

    # Ensure n is a positive integer
    if n < 1:
        raise ValueError("n must be >= 1")

    # Dictory that maps n-gram to list of positions
    ngram_positions: Dict[Union[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    limit = len(tokens) - n + 1

    # Iterate over all possible starting positions
    for start_index in range(max(0, limit)):
        ngram_key: Union[str, Tuple[str, ...]]
        # For unigrams, use a string key
        if n == 1:
            ngram_key = tokens[start_index]
        # For n>1, use a tuple of tokens as the key
        else:
            ngram_key = tuple(tokens[start_index : start_index + n])

        # Record the starting index of the n-gram
        ngram_positions[ngram_key].append(start_index)

    return dict(ngram_positions)