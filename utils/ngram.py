"""Word- and char-level n-gram helpers (Task A-2)."""
from typing import List, Tuple

__all__ = ["make_ngrams_tokens", "make_ngrams_chars"]

def make_ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return a list of token n-grams.

    Args:
    ----------
    tokens: List[str]
        Sequence of tokens.
    n: int
        Size of each n-gram (must be > 0).

    Returns:
    -------
    List[Tuple[str, ...]]
        Consecutive token n-grams represented as tuples. 
    """
    if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
        raise TypeError("tokens must be a list of strings")

    if n < 1:
        raise ValueError("n must be >= 1")

    # Pad the sequence with start and end tokens
    padded_sequence = ["<s>"] + tokens + ["</s>"]
    
    if n > len(padded_sequence):
        return []
    
    # Slide a window of length n over the padded sequence
    ngram_list = []
    for start_index in range(len(padded_sequence) - n + 1):
        ngram = tuple(padded_sequence[start_index : start_index + n])
        ngram_list.append(ngram)

    return ngram_list

def make_ngrams_chars(text: str, n: int) -> List[str]:
    """Return a list of character n-grams from ``text``.

    Args:
    ----------
    text: str
        Input string from which to extract n-grams.
    n: int
        Size of each n-gram (must be > 0).

    Returns:
    -------
    List[str]
        Consecutive character n-grams. 
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if n < 1:
        raise ValueError("n must be >= 1")

    # Pad the text with start and end markers
    padded_text = f"${text}$"

    if n > len(padded_text):
        return []
    
    # Slide a window of length n over the padded text
    ngram_list = []
    for start_index in range(len(padded_text) - n + 1):
        ngram = padded_text[start_index : start_index + n]
        ngram_list.append(ngram)

    return ngram_list
