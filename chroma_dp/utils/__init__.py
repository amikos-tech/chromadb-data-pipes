from typing import Optional


def get_max_with_none(a: Optional[int], b: Optional[int]) -> int:
    if a is None:
        return b
    if b is None:
        return a
    if a is None and b is None:
        raise ValueError("Both values cannot be None")
    return max(a, b)
