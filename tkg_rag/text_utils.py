import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> set[str]:
    s = (s or "").lower()
    return set(TOKEN_RE.findall(s))


def iou(a: set[str], b: set[str]) -> float:
    # Intersection over union (Jaccard similarity).
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union
