import pytest

from backend.agent.tools import CalculatorError, LogStore, calculate, extract_expression


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("64/2", 32),
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("2**10", 1024),
        ("2^10", 1024),
        ("-5 + 3", -2),
        ("sqrt(16)", 4),
        ("max(3, 7, 5)", 7),
        ("round(2.567, 2)", 2.57),
        ("10 % 3", 1),
        ("7 // 2", 3),
        ("pi * 2", 6.2831853072),
        ("1/3", 0.3333333333),
    ],
)
def test_calculator_evaluates_arithmetic(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "x + 1",
        "(1).__class__",
        "[i for i in range(10)]",
        "lambda: 1",
        "2 ** 100000",
        "1/0",
        "9" * 300,
        "sqrt(-1)",
        "'a' * 3",
    ],
)
def test_calculator_rejects_non_arithmetic_and_abuse(expr):
    with pytest.raises(CalculatorError):
        calculate(expr)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("if queue depth is 64 what's 64/2?", "64/2"),
        ("what is (120 - 30) * 2 please", "(120 - 30) * 2"),
        ("what's 15% of 400", "(15/100*400)"),
        ("compute 12 times 4", "12 * 4"),
        ("how do I fix EIO-5", None),
    ],
)
def test_extract_expression(text, expected):
    assert extract_expression(text) == expected


def test_log_store_filters_by_host_level_and_text():
    store = LogStore("samples/logs")
    assert "storage-07" in store.hosts()
    errors = store.search(host="storage-07", min_level="ERROR")
    assert errors and all(ln.level == "ERROR" and ln.host == "storage-07" for ln in errors)
    assert all("EIO-5" in ln.message or "degraded" in ln.message for ln in errors)
    assert [ln.host for ln in store.search(contains="redundancy")] == ["rack-b4"]
    assert store.search(host="nope") == []
    assert len(store.search(limit=3)) == 3
