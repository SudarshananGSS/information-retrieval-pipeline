from typing import Set, List, Union
import re
from index.access import get_term_positions, get_posting_list
from .detection import _validate_proximity

def process_proximity_query(query: str, index_path: str) -> Set[int]:
    """
    Process proximity queries with NEAR/k semantics.
    
    Args:
        query: Proximity query (e.g., "climate NEAR/3 change", '"machine learning" NEAR/2 algorithms')
        index_path: Path to the unified index package
        
    Returns:
        Set of document IDs where operands satisfy NEAR/k distance constraint
        
    NEAR/k semantics:
    - Distance D = min(|q_start - p_end|, |p_start - q_end|)
    - Document satisfies NEAR/k iff D <= k for some occurrence pair
    - Edge-to-edge distance, order-insensitive
    """
    
    _validate_proximity(query)
    m = re.fullmatch(r'(\".*?\"|\S+)\s+NEAR/(\d+)\s+(\".*?\"|\S+)', query)
    if not m:
        raise ValueError("Malformed proximity query")

    op1_raw, k_str, op2_raw = m.groups()
    k = int(k_str)

    def parse_operand(op: str) -> Union[str, tuple]:
        if op.startswith('"') and op.endswith('"'):
            parts = op[1:-1].split()
            if len(parts) > 1:
                return tuple(parts)
            return parts[0] if parts else ""
        return op

    t1 = parse_operand(op1_raw)
    t2 = parse_operand(op2_raw)

    candidate_docs = set(get_posting_list(t1, index_path)) & set(get_posting_list(t2, index_path))

    len1 = len(t1) if isinstance(t1, tuple) else 1
    len2 = len(t2) if isinstance(t2, tuple) else 1

    result: Set[int] = set()
    for doc in candidate_docs:
        pos1 = get_term_positions(t1, doc, index_path)
        pos2 = get_term_positions(t2, doc, index_path)
        for p in pos1:
            s1, e1 = p, p + len1 - 1
            for q in pos2:
                s2, e2 = q, q + len2 - 1
                if s1 == s2 and e1 == e2:
                    continue
                dist = min(abs(s1 - e2), abs(s2 - e1))
                if dist <= k:
                    result.add(doc)
                    break
            else:
                continue
            break

    

    return result