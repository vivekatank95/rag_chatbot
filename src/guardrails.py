from typing import Tuple

BLOCKED_KEYWORDS = ["bomb", "how to make a gun", "explosive", "illegal", "ssn"]

def check_safe(query: str) -> Tuple[bool, str]:
    """Simple guardrails to block unsafe queries."""
    qlow = query.lower()
    for b in BLOCKED_KEYWORDS:
        if b in qlow:
            return False, f"Query contains blocked term: {b}"
    return True, "ok"
