"""Robust HTML→tokens cleaning pipeline (Task A-1)."""
import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

__all__ = ["preprocess"]


def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Convert noisy HTML documents into token lists.
    raw_html_list: List[str] a list of raw html documents, see data/dev/documents.jsonl and queries.json to see possible noisy html documents and inputs
    return: List[List[str]] a list of token lists, each token list is a list of tokens
    Hints:
    - strips tags & entities
    - keeps 1 punctuation
    - no stemming/stop-word removal (Part A spec)
    """
    processed_documents: List[List[str]] = []

    for html_fragment in raw_html_list:
        if not isinstance(html_fragment, str):  # guard against bad inputs
            processed_documents.append([])
            continue

        # Strip HTML tags using BeautifulSoup (fallback to regex on failure)
        try:
            plain_text = BeautifulSoup(html_fragment, "html.parser").get_text(" ")
        except Exception:
            plain_text = re.sub(r"<[^>]+>", " ", html_fragment)

        # Decode HTML entities and normalise unicode
        plain_text = html.unescape(plain_text)
        plain_text = plain_text.replace("&", " and ")
        plain_text = unicodedata.normalize("NFKC", plain_text)

        # Remove non-alphanumeric characters except basic punctuation
        plain_text = re.sub(r"[^0-9A-Za-z\s.,!?()\"'-]", " ", plain_text)
        
        # Break hyphenated words and split contractions/possessives
        plain_text = re.sub(r"-+", " ", plain_text)

        # Ensure dashes are spaced properly
        plain_text = re.sub(r"[-–—]+", " - ", plain_text)

        # Collapse repeated punctuation and whitespace
        plain_text = re.sub(r"([.,!?]){2,}", r"\1", plain_text)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()

        tokens = word_tokenize(plain_text)

        suffix_expansions = {
            "n't": "not",
            "'re": "are",
            "'m": "am",
            "'ll": "will",
            "'d": "would",
            "'ve": "have",
            "'s": "",
        }
        prefix_expansions = {"ca": "can", "wo": "will", "sha": "shall"}

        clean_tokens: List[str] = []

        for token in tokens:
            if token in prefix_expansions:
                clean_tokens.append(prefix_expansions[token])
            elif token in suffix_expansions:
                expanded = suffix_expansions[token]
                if expanded:
                    clean_tokens.append(expanded)
            elif token in {"``", "''"}:
                clean_tokens.append('"')
            elif re.fullmatch(r"[A-Za-z0-9]+|[.,!?()\"']", token):
                clean_tokens.append(token)
                
        processed_documents.append(clean_tokens)

    return processed_documents