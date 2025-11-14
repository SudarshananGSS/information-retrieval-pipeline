from typing import Set, List, Union
from index.access import get_posting_list, _load_package
import re
from .detection import _validate_boolean

def process_boolean_query(query: str, index_path: str) -> Set[int]:
    """
    Process Boolean queries with AND/OR/NOT operators, parentheses, and quoted phrases.
    
    Args:
        query: Boolean query string with operators and optional quotes/parentheses
        index_path: Path to the unified index package
        
    Returns:
        Set of document IDs matching the boolean query
        
    Precedence: NOT > AND > OR
    Supports parentheses and quoted phrases
    """
    _validate_boolean(query)
    tokens = re.findall(r'".*?"|\(|\)|AND|OR|NOT|[^\s()]+', query)
    if not tokens:
        return set()

    precedence = {"OR": 1, "AND": 2, "NOT": 3}
    output: List[str] = []
    stack: List[str] = []
    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if stack:
                stack.pop()
        elif tok in ("AND", "OR", "NOT"):
            if tok == "NOT":
                while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) > precedence[tok]:
                    output.append(stack.pop())
            else:
                while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) >= precedence[tok]:
                    output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(tok)
    while stack:
        output.append(stack.pop())

    package = _load_package(index_path)
    all_docs = set(package.get("__META__", {}).get("doc_lengths", {}).keys())

    def parse_term(token: str) -> Union[str, tuple]:
        if token.startswith('"') and token.endswith('"'):
            parts = token[1:-1].split()
            if len(parts) > 1:
                return tuple(parts)
            return parts[0] if parts else ""
        return token

    eval_stack: List[Set[int]] = []
    for tok in output:
        if tok in ("AND", "OR", "NOT"):
            if tok == "NOT":
                operand = eval_stack.pop() if eval_stack else set()
                eval_stack.append(all_docs - operand)
            else:
                right = eval_stack.pop() if eval_stack else set()
                left = eval_stack.pop() if eval_stack else set()
                if tok == "AND":
                    eval_stack.append(left & right)
                else:
                    eval_stack.append(left | right)
        else:
            term = parse_term(tok)
            postings = set(get_posting_list(term, index_path))
            eval_stack.append(postings)

    return eval_stack[-1] if eval_stack else set()
