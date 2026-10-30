import pytest

from recon.formatting import compact_inr, inr


@pytest.mark.parametrize("value,expected", [
    (0, "₹0"), (999, "₹999"), (1000, "₹1,000"), (12345, "₹12,345"), (123456, "₹1,23,456"),
    (1234567, "₹12,34,567"), (17737504, "₹1,77,37,504"), (-45000, "-₹45,000"), (1000000000, "₹1,00,00,00,000")])
def test_indian_grouping(value, expected):
    assert inr(value) == expected


def test_decimals_and_symbol_options():
    assert inr(1234567.891, 2) == "₹12,34,567.89"
    assert inr(1234567, symbol=False) == "12,34,567"
    assert inr(None) == "-" and inr(float("nan")) == "-"


def test_compact_forms():
    assert compact_inr(17737504) == "₹1.77 Cr" and compact_inr(250000) == "₹2.50 L" and compact_inr(9999) == "₹9,999"
