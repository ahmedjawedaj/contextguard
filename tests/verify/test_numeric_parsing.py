import pytest

from contextguard.verify.numeric import normalize_amount
from contextguard.core.specs import UnitConstraint, ReasonCode


def test_normalize_scale_and_currency():
    units = UnitConstraint(currency="USD", scale="million")
    num = normalize_amount("$0.2B", units=units)
    assert num is not None
    # 0.2B = 200 million
    assert abs(num.value - 200_000_000) < 1e-6
    assert num.currency == "USD"


def test_mismatch_currency_raises():
    units = UnitConstraint(currency="EUR", scale="million")
    with pytest.raises(ValueError) as exc:
        normalize_amount("$50m", units=units)
    assert ReasonCode.CTXT_UNIT_SCALE_MISMATCH.value in str(exc.value)


def test_scale_tolerance():
    units = UnitConstraint(currency="USD", scale="million")
    num = normalize_amount("200 million", units=units)
    assert num is not None
    assert num.value == 200_000_000


def test_no_match_returns_none():
    units = UnitConstraint(currency="USD", scale="million")
    num = normalize_amount("no numbers here", units=units)
    assert num is None

