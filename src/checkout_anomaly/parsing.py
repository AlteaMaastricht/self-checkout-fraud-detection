"""Reading and writing self-checkout visit logs.

Input format (one visit per line)::

    dept seconds price, dept seconds price, ...

Each comma-separated tuple is one scanned item:

* ``dept``    - integer department code of the scanned item
* ``seconds`` - seconds elapsed since the previous scan
* ``price``   - price of the scanned item

Evaluation files used in the course prefixed every line with
``Transaction <id>:``. The original parsers split on whitespace and kept only
3-token tuples, so the prefix silently turned the first tuple into 4 tokens and
the first item of every visit was dropped (see ``docs/PROVENANCE.md``). This
parser strips the prefix and keeps the id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

Item = tuple[int, float, float]

_PREFIX = re.compile(r"^\s*(?:Transaction\s+)?(?P<id>[A-Za-z0-9_-]+)\s*:\s*")


@dataclass
class Visit:
    """A single shopping visit: an ordered list of scanned items."""

    visit_id: str
    items: list[Item] = field(default_factory=list)

    def __len__(self) -> int:  # number of scanned items
        return len(self.items)


class ParseError(ValueError):
    """Raised in strict mode when a tuple cannot be parsed."""


def parse_visit_line(line: str, default_id: str, strict: bool = False) -> Visit:
    """Parse one line of the visit log.

    Malformed tuples are skipped (the original behaviour) unless ``strict``
    is True, in which case a :class:`ParseError` is raised.
    """
    text = line.strip().rstrip(",").strip()
    visit_id = default_id
    m = _PREFIX.match(text)
    if m:
        visit_id = m.group("id")
        text = text[m.end():]

    items: list[Item] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 3:
            if strict:
                raise ParseError(f"visit {visit_id}: expected 3 fields, got {raw!r}")
            continue
        try:
            dept = int(float(parts[0]))
            seconds = float(parts[1])
            price = float(parts[2])
        except ValueError as exc:
            if strict:
                raise ParseError(f"visit {visit_id}: cannot parse {raw!r}") from exc
            continue
        items.append((dept, seconds, price))
    return Visit(visit_id=visit_id, items=items)


def iter_visits(lines: Iterable[str], strict: bool = False) -> Iterator[Visit]:
    """Yield visits from an iterable of lines; blank lines and empty visits are skipped.

    Visits without an explicit id get the 0-based line number as id.
    """
    for lineno, line in enumerate(lines):
        if not line.strip():
            continue
        visit = parse_visit_line(line, default_id=str(lineno), strict=strict)
        if visit.items:
            yield visit


def read_visits(path: str | Path, strict: bool = False) -> list[Visit]:
    """Read a visit log file (CRLF or LF line endings)."""
    with open(path, "r", encoding="utf-8") as fh:
        return list(iter_visits(fh, strict=strict))


def format_visit(visit: Visit, with_id: bool = False) -> str:
    """Serialise a visit back to the log format."""
    body = ", ".join(f"{d} {t:g} {p:.2f}" for d, t, p in visit.items)
    return f"Transaction {visit.visit_id}:{body}" if with_id else body


def write_visits(visits: Iterable[Visit], path: str | Path, with_id: bool = False) -> None:
    """Write visits in the log format, one per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for v in visits:
            fh.write(format_visit(v, with_id=with_id) + "\n")
