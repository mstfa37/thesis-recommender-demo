"""Unit tests for the storefront's pure helpers (no browser needed)."""

import importlib.util
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _load_helpers():
    # load app.py without running the streamlit script body
    src = APP.read_text()
    helpers = []
    capture = False
    for line in src.splitlines():
        if line.startswith(("def stars", "def price_str", "def price_val")):
            capture = True
        elif capture and line and not line.startswith((" ", "\t", "#")):
            capture = False
        if capture:
            helpers.append(line)
    ns: dict = {}
    exec("\n".join(helpers), ns)  # noqa: S102 - test-only, local file
    return ns


@pytest.fixture(scope="module")
def h():
    return _load_helpers()


def test_stars(h):
    assert h["stars"](5) == "★★★★★"
    assert h["stars"](0) == "☆☆☆☆☆"
    assert h["stars"](3.6) == "★★★★☆"


def test_price_str(h):
    assert h["price_str"]("9.99") == "$9.99"
    assert h["price_str"]("$9.99") == "$9.99"
    assert h["price_str"]("") == ""
    assert h["price_str"]("  ") == ""


def test_price_val(h):
    assert h["price_val"]("$1,299.00") == pytest.approx(1299.0)
    assert h["price_val"]("9.99") == pytest.approx(9.99)
    import math
    assert math.isinf(h["price_val"](""))
    assert math.isnan(h["price_val"]("nan"))
