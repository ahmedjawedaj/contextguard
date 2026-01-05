"""
Numeric parsing and normalization for unit/currency/scale handling.

Design:
- Money-first parsing (currency + scale) via `normalize_amount`.
- Percentage parsing via `normalize_percentage`.
- Generic quantity parsing (number + unit token) via `normalize_quantity`.

Customization:
- Extend `_UNIT_TOKENS` to add domain-specific units (e.g., "requests", "transactions").
- Extend `_SCALE_MULT` if you need more scales (e.g., "trillion").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ..core.specs import UnitConstraint, ReasonCode


@dataclass
class NormalizedNumber:
    value: float               # absolute value in base units (no scale)
    currency: Optional[str]    # ISO currency if known
    scale: Optional[str]       # raw|thousand|million|billion
    text: str                  # original text
    unit: Optional[str] = None # e.g., "%", "users", "kg"


_NUM_RE = re.compile(
    # currency + number + optional scale
    r"(?:(?P<currency>[\$€£]|USD|EUR|GBP)\s*(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|\.\d+)\s*(?P<scale>billion|bn|b|million|m|thousand|k)?)"
    # OR number + mandatory scale (no currency)
    r"|(?P<number2>[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|\.\d+)\s*(?P<scale2>billion|bn|b|million|m|thousand|k)\b",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(
    r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|\.\d+)\s*(?P<unit>%|percent|percentage points|pp)\b",
    re.IGNORECASE,
)

_SCALE_MULT = {
    "raw": 1.0,
    None: 1.0,
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_UNIT_TOKENS = {
    "users",
    "requests",
    "transactions",
    "visits",
    "impressions",
    "clicks",
    "sessions",
    "kg",
    "g",
    "lbs",
    "units",
    "items",
    "shares",
    "downloads",
    "messages",
    "events",
}


def normalize_amount(text: str, units: Optional[UnitConstraint], default_currency: Optional[str] = None) -> Optional[NormalizedNumber]:
    """
    Parse and normalize a numeric amount from text.
    Returns None if no numeric match is found.
    """
    m = _NUM_RE.search(text)
    if not m:
        return None
    currency_raw = m.group("currency")
    num_raw = m.group("number") or m.group("number2")
    scale_raw = m.group("scale") or m.group("scale2")
    scale_raw = scale_raw.lower() if scale_raw else None
    currency = None
    if currency_raw:
        if currency_raw in ["$", "USD"]:
            currency = "USD"
        elif currency_raw in ["€", "EUR"]:
            currency = "EUR"
        elif currency_raw in ["£", "GBP"]:
            currency = "GBP"
        else:
            currency = currency_raw.upper()
    elif default_currency:
        currency = default_currency
    try:
        value = float(num_raw.replace(",", ""))
    except ValueError:
        return None
    mult = _SCALE_MULT.get(scale_raw, 1.0)
    value *= mult
    normalized = NormalizedNumber(value=value, currency=currency, scale=scale_raw, text=text)
    if units:
        _check_units(normalized, units)
    return normalized


def normalize_percentage(text: str) -> Optional[NormalizedNumber]:
    """
    Parse percentages like "12%" or "5 percent".
    Returns value normalized to fraction (0-1).
    """
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    num_raw = m.group("number")
    unit_raw = m.group("unit")
    try:
        val = float(num_raw.replace(",", ""))
    except ValueError:
        return None
    unit_norm = "%" if unit_raw.startswith("%") else "percent"
    return NormalizedNumber(value=val / 100.0, currency=None, scale="raw", unit=unit_norm, text=text)


def normalize_quantity(text: str) -> Optional[Tuple[NormalizedNumber, str]]:
    """
    Parse quantities like "10k users", "5 million transactions", "3 kg".
    Returns (NormalizedNumber, unit_token) or None.
    """
    quantity_re = re.compile(
        r"(?P<number>[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?|\.\d+)\s*(?P<scale>billion|bn|b|million|m|thousand|k)?\s*(?P<unit>[a-zA-Z%]{2,})",
        re.IGNORECASE,
    )
    m = quantity_re.search(text)
    if not m:
        return None
    num_raw = m.group("number")
    scale_raw = m.group("scale")
    unit_raw = m.group("unit").lower()
    if unit_raw not in _UNIT_TOKENS and unit_raw not in {"percent", "percentage"}:
        return None
    try:
        val = float(num_raw.replace(",", ""))
    except ValueError:
        return None
    mult = _SCALE_MULT.get(scale_raw.lower() if scale_raw else None, 1.0)
    val *= mult
    nn = NormalizedNumber(
        value=val,
        currency=None,
        scale=scale_raw.lower() if scale_raw else "raw",
        unit=unit_raw,
        text=text,
    )
    return nn, unit_raw


def _check_units(num: NormalizedNumber, units: UnitConstraint) -> None:
    """
    Raise ReasonCode if currency/scale mismatch with requested units.
    """
    # Currency check
    if units.currency and num.currency and units.currency != num.currency:
        raise ValueError(ReasonCode.CTXT_UNIT_SCALE_MISMATCH.value)
    # Scale differences are handled by normalization; do not reject if convertible
