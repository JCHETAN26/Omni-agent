"""Non-retrieval tools the agent can route to.

* `calculate`: arithmetic through an AST whitelist. It never executes code
  (no names, attributes, calls except a few math functions, or comprehensions).
  A general code sandbox needs process isolation and is deliberately not here.
* `LogStore`: searches recent log lines. The development source is a directory
  of `.log` files. A Loki/Elasticsearch-backed store slots in behind the same
  `search` method.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Calculator
# ----------------------------------------------------------------------------


class CalculatorError(ValueError):
    pass


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "ceil": math.ceil,
    "floor": math.floor,
}
_CONSTS = {"pi": math.pi, "e": math.e}
_MAX_ABS = 1e15
_MAX_EXPONENT = 64


def calculate(expression: str) -> float | int:
    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expr) > 200:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not an arithmetic expression: {expression!r}") from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise CalculatorError("exponent too large")
            try:
                value = _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise CalculatorError("division by zero") from exc
            if isinstance(value, complex) or abs(value) > _MAX_ABS:
                raise CalculatorError("result out of range")
            return value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
            and len(node.args) <= 4
        ):
            try:
                return _FUNCS[node.func.id](*[ev(a) for a in node.args])
            except (ValueError, TypeError) as exc:
                raise CalculatorError(str(exc)) from exc
        raise CalculatorError(f"unsupported syntax: {ast.dump(node)[:60]}")

    result = ev(tree)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e12:
        return int(result)
    return round(result, 10) if isinstance(result, float) else result


_EXPR_RE = re.compile(
    r"(?:(?:sqrt|log2|log10|log|abs|round|min|max|ceil|floor)\s*\(|[\d(.])"
    r"[\d\s.+\-*/%^()×÷,a-z]*[\d)]",
    re.IGNORECASE,
)


def extract_expression(text: str) -> str | None:
    """Pull the longest arithmetic-looking span out of free text."""
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100*\2)", text, flags=re.I)
    text = re.sub(r"\b(?:times|multiplied by)\b", "*", text, flags=re.I)
    text = re.sub(r"\bdivided by\b", "/", text, flags=re.I)
    text = re.sub(r"\bplus\b", "+", text, flags=re.I)
    text = re.sub(r"\bminus\b", "-", text, flags=re.I)
    candidates = [m.group(0).strip() for m in _EXPR_RE.finditer(text)]
    candidates = [c for c in candidates if re.search(r"[-+*/%^()]", c) and re.search(r"\d", c)]
    return max(candidates, key=len) if candidates else None


# ----------------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------------

_LOG_RE = re.compile(r"^(?P<ts>\S+)\s+(?P<level>[A-Z]+)\s+(?P<host>\S+)\s+(?P<msg>.*)$")
_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


@dataclass(frozen=True)
class LogLine:
    ts: str
    level: str
    host: str
    message: str

    def render(self) -> str:
        return f"{self.ts} {self.level:<5} {self.host} {self.message}"


class LogStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        if not self.path.exists():
            return out
        files = [self.path] if self.path.is_file() else sorted(self.path.glob("*.log"))
        for f in files:
            for raw in f.read_text().splitlines():
                if m := _LOG_RE.match(raw.strip()):
                    out.append(LogLine(m["ts"], m["level"], m["host"], m["msg"]))
        return sorted(out, key=lambda ln: ln.ts)

    def hosts(self) -> set[str]:
        return {ln.host for ln in self._lines()}

    def search(
        self,
        host: str | None = None,
        min_level: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> list[LogLine]:
        floor = _LEVELS.get((min_level or "").upper(), 0)
        needle = (contains or "").lower()
        hits = [
            ln
            for ln in self._lines()
            if (not host or ln.host == host)
            and _LEVELS.get(ln.level, 0) >= floor
            and (not needle or needle in ln.message.lower())
        ]
        return hits[-limit:]  # most recent
