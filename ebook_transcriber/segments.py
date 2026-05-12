from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Segment:
    id: str
    title: str | None
    pdf_pages: tuple[int, int]


_SEGMENT_START_RE = re.compile(r"^\s*-\s+id:\s*(?P<value>.+?)\s*$")
_FIELD_RE = re.compile(r"^\s+(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s*(?P<value>.*?)\s*$")
_RANGE_RE = re.compile(r"^\[\s*(\d+)\s*,\s*(\d+)\s*\]$")


def _parse_scalar(value: str) -> str | None:
    value = value.strip()
    if value == "null":
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_pdf_pages(value: str) -> tuple[int, int]:
    match = _RANGE_RE.match(value.strip())
    if not match:
        raise ValueError(f"unsupported pdf_pages value: {value}")
    start = int(match.group(1))
    end = int(match.group(2))
    if start < 1 or end < start:
        raise ValueError(f"invalid pdf_pages range: {value}")
    return start, end


def read_segments(path: str | Path) -> list[Segment]:
    path = Path(path)
    segments: list[Segment] = []
    current: dict[str, object] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        if "id" not in current or "pdf_pages" not in current:
            raise ValueError(f"segment missing id or pdf_pages in {path}")
        segments.append(
            Segment(
                id=str(current["id"]),
                title=current.get("title") if isinstance(current.get("title"), str) else None,
                pdf_pages=current["pdf_pages"],
            )
        )
        current = None

    for line in path.read_text(encoding="utf-8").splitlines():
        start_match = _SEGMENT_START_RE.match(line)
        if start_match:
            flush()
            current = {"id": _parse_scalar(start_match.group("value"))}
            continue
        if current is None:
            continue
        field_match = _FIELD_RE.match(line)
        if not field_match:
            continue
        key = field_match.group("key")
        value = field_match.group("value")
        if key == "pdf_pages":
            current[key] = _parse_pdf_pages(value)
        elif key == "title":
            current[key] = _parse_scalar(value)

    flush()
    if not segments:
        raise ValueError(f"no segments found in {path}")
    return segments


def safe_segment_filename(segment_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment_id).strip("._")
    if not safe:
        raise ValueError(f"invalid segment id for filename: {segment_id!r}")
    return safe


def write_index_markdown(output_dir: str | Path, segments: Iterable[Segment], index_name: str = "index.md") -> Path:
    output_dir = Path(output_dir)
    lines: list[str] = []
    missing: list[str] = []
    for segment in segments:
        filename = f"{safe_segment_filename(segment.id)}.md"
        path = output_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        title = segment.title or segment.id
        lines.append(f"- [{title}]({filename})")
    if missing:
        raise FileNotFoundError("missing segment Markdown files: " + ", ".join(missing))

    index_path = output_dir / index_name
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path
