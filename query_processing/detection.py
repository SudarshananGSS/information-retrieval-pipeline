import re

def detect_query_type(query: str) -> str:
    """
    Detect query type based on case-sensitive syntax keywords.
    
    Args:
        query: Input query string
        
    Returns:
        Query type: "proximity", "wildcard", "boolean", or "natural_language"
    """
    qtype: str
    if "NEAR" in query and not re.search(r"\bNEAR/\d+\b", query):
        raise ValueError("Malformed proximity query")
    if re.search(r"NEAR/\d+", query):
        qtype = "proximity"
    elif "*" in query:
        qtype = "wildcard"
    elif re.search(r"\b(?:AND|OR|NOT)\b|[()\"]", query):
        qtype = "boolean"
    else:
        qtype = "natural_language"

    _validate_query(query, qtype)
    return qtype


def _validate_query(query: str, qtype: str) -> None:
    if qtype == "wildcard":
        _validate_wildcard(query)
    elif qtype == "proximity":
        _validate_proximity(query)
    elif qtype == "boolean":
        _validate_boolean(query)


def _validate_wildcard(query: str) -> None:
    if re.search(r"\b(?:AND|OR|NOT)\b|NEAR/\d+", query) or '"' in query:
        raise ValueError("Malformed wildcard query")
    if " " in query.strip():
        raise ValueError("Malformed wildcard query")
    if query.strip("*") == "":
        raise ValueError("Malformed wildcard query")


def _validate_proximity(query: str) -> None:
    if len(re.findall(r"NEAR/\d+", query)) != 1:
        raise ValueError("Malformed proximity query")
    m = re.fullmatch(r'(".*?"|[^\s()\"]+)\s+NEAR/(\d+)\s+(".*?"|[^\s()\"]+)', query)
    if not m:
        raise ValueError("Malformed proximity query")
    for op in (m.group(1), m.group(3)):
        if '(' in op or ')' in op:
            raise ValueError("Malformed proximity query")
        if op.startswith('"') and op.endswith('"'):
            parts = op[1:-1].split()
            if len(parts) == 0 or len(parts) > 3:
                raise ValueError("Malformed proximity query")


def _validate_boolean(query: str) -> None:
    if "*" in query or re.search(r"NEAR/\d+", query):
        raise ValueError("Malformed boolean query")
    if query.count('"') % 2 == 1:
        raise ValueError("Malformed boolean query")
    tokens = re.findall(r'".*?"|\(|\)|AND|OR|NOT|[^\s()]+', query)
    if not tokens:
        raise ValueError("Malformed boolean query")
    ops = {"AND", "OR", "NOT"}
    for tok in tokens:
        if re.fullmatch(r"[A-Z]{2,}", tok) and tok not in ops:
            raise ValueError("Malformed boolean query")

    def is_operand(tok: str) -> bool:
        return tok not in ops and tok not in {"(", ")"}

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if is_operand(a) and is_operand(b):
            raise ValueError("Malformed boolean query")
        if is_operand(a) and b == '(':  # operand directly before '('
            raise ValueError("Malformed boolean query")
        if a == ')' and is_operand(b):  # ')' directly before operand
            raise ValueError("Malformed boolean query")
    depth = 0
    for tok in tokens:
        if tok == '(':
            depth += 1
        elif tok == ')':
            depth -= 1
        if depth < 0:
            raise ValueError("Malformed boolean query")
    if depth != 0:
        raise ValueError("Malformed boolean query")
    for i, tok in enumerate(tokens):
        if tok in {"AND", "OR"}:
            if i == 0 or i == len(tokens) - 1:
                raise ValueError("Malformed boolean query")
            if tokens[i - 1] in ops or tokens[i - 1] == '(':
                raise ValueError("Malformed boolean query")
            if tokens[i + 1] in {"AND", "OR"} or tokens[i + 1] == ')':
                raise ValueError("Malformed boolean query")
        elif tok == "NOT":
            if i == len(tokens) - 1 or tokens[i + 1] in {"AND", "OR", ")"}:
                raise ValueError("Malformed boolean query")
        elif tok == '(':
            if i < len(tokens) - 1 and tokens[i + 1] in {"AND", "OR", ")"}:
                raise ValueError("Malformed boolean query")
        elif tok == ')':
            if i > 0 and tokens[i - 1] in {"AND", "OR", "NOT", '('}:
                raise ValueError("Malformed boolean query")
        if tok.startswith('"') and tok.endswith('"'):
            parts = tok[1:-1].split()
            if len(parts) == 0 or len(parts) > 3:
                raise ValueError("Malformed boolean query")
