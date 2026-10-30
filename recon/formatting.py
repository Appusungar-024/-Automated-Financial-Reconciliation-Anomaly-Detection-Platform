"""Display helpers."""
from __future__ import annotations

from . import config


def inr(value, decimals: int = 0, symbol: bool = True) -> str:
    """Indian digit grouping: 17737504 -> 1,77,37,504 (the last three digits, then pairs)."""
    if value is None or value != value:           # None or NaN
        return "-"
    negative = value < 0
    whole, _, frac = f"{abs(float(value)):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    text = whole + (f".{frac}" if frac else "")
    return ("-" if negative else "") + (config.CURRENCY_SYMBOL if symbol else "") + text


def compact_inr(value: float) -> str:
    """Lakh / crore shorthand for tight spaces: 17737504 -> ₹1.77 Cr."""
    v = abs(float(value))
    sign = "-" if value < 0 else ""
    if v >= 1e7:
        return f"{sign}{config.CURRENCY_SYMBOL}{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}{config.CURRENCY_SYMBOL}{v / 1e5:.2f} L"
    return f"{sign}{config.CURRENCY_SYMBOL}{v:,.0f}"
