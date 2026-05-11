from __future__ import annotations

import json
import re
from pathlib import Path


_ASSET_RE = re.compile(r"\[\[ASSET:.*?alt=[\"'](?P<alt>.*?)[\"'].*?\]\]")


def default_output_path(pdf_path: str | Path, output_dir: str | Path = "output") -> Path:
    pdf_path = Path(pdf_path)
    return Path(output_dir) / f"{pdf_path.stem}.md"


def asset_dir_for(output_path: str | Path, assets_dir_name: str) -> Path:
    return Path(output_path).parent / assets_dir_name


def checkpoint_path_for(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_suffix(f"{output_path.suffix}.checkpoint.json")


def relative_asset_path(output_path: str | Path, asset_path: str | Path) -> str:
    return Path(asset_path).relative_to(Path(output_path).parent).as_posix()


def replace_first_asset_placeholder(markdown: str, asset_rel_path: str, alt: str = "figure") -> str:
    match = _ASSET_RE.search(markdown)
    if match:
        alt = match.group("alt") or alt
        return markdown[: match.start()] + f"![{alt}]({asset_rel_path})" + markdown[match.end() :]
    return f"{markdown.rstrip()}\n\n![{alt}]({asset_rel_path})\n"


def read_completed_pages(checkpoint_path: str | Path) -> set[int]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return set()
    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {int(page) for page in data.get("completed_pages", [])}


def write_completed_pages(checkpoint_path: str | Path, completed_pages: set[int]) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"completed_pages": sorted(completed_pages)}
    checkpoint_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_progressive_output(output_path: str | Path, checkpoint_path: str | Path, restart: bool) -> set[int]:
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        output_path.write_text("", encoding="utf-8")
        write_completed_pages(checkpoint_path, set())
        return set()
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
        write_completed_pages(checkpoint_path, set())
        return set()
    return read_completed_pages(checkpoint_path)


def append_page_markdown(output_path: str | Path, page_number: int, markdown: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_body = f"<!-- page {page_number} -->\n\n{markdown.strip()}\n"
    separator = "" if output_path.stat().st_size == 0 else "\n\n"
    with output_path.open("a", encoding="utf-8") as file:
        file.write(separator + page_body)


def write_markdown(output_path: str | Path, parts: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(part.rstrip() for part in parts if part.strip()).rstrip() + "\n"
    output_path.write_text(body, encoding="utf-8")
