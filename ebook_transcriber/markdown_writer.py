from __future__ import annotations

import re
from pathlib import Path


_ASSET_RE = re.compile(r"\[\[ASSET:.*?alt=[\"'](?P<alt>.*?)[\"'].*?\]\]")


def default_output_path(pdf_path: str | Path, output_dir: str | Path = "output") -> Path:
    pdf_path = Path(pdf_path)
    return Path(output_dir) / f"{pdf_path.stem}.md"


def asset_dir_for(output_path: str | Path, assets_dir_name: str) -> Path:
    return Path(output_path).parent / assets_dir_name


def relative_asset_path(output_path: str | Path, asset_path: str | Path) -> str:
    return Path(asset_path).relative_to(Path(output_path).parent).as_posix()


def replace_first_asset_placeholder(markdown: str, asset_rel_path: str, alt: str = "figure") -> str:
    match = _ASSET_RE.search(markdown)
    if match:
        alt = match.group("alt") or alt
        return markdown[: match.start()] + f"![{alt}]({asset_rel_path})" + markdown[match.end() :]
    return f"{markdown.rstrip()}\n\n![{alt}]({asset_rel_path})\n"


def write_markdown(output_path: str | Path, parts: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(part.rstrip() for part in parts if part.strip()).rstrip() + "\n"
    output_path.write_text(body, encoding="utf-8")
