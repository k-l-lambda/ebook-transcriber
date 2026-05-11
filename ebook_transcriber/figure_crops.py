from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

from . import markdown_writer
from .pdf_reader import open_document, render_clip_to_file

_PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$")
_FIGURE_RE = re.compile(r"^\[Figure:\s*(?P<description>.*?)\]\s*$")


@dataclass(frozen=True)
class FigureAnchor:
    line_index: int
    page_number: int
    figure_index: int
    description: str


@dataclass(frozen=True)
class CropResult:
    anchor: FigureAnchor
    asset_path: Path
    rel_path: str
    rect: fitz.Rect
    fallback: bool


def parse_figure_anchors(markdown: str) -> list[FigureAnchor]:
    anchors: list[FigureAnchor] = []
    current_page: int | None = None
    figure_counts: dict[int, int] = {}
    for line_index, line in enumerate(markdown.splitlines()):
        page_match = _PAGE_MARKER_RE.match(line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue
        figure_match = _FIGURE_RE.match(line.strip())
        if figure_match and current_page is not None:
            figure_index = figure_counts.get(current_page, 0) + 1
            figure_counts[current_page] = figure_index
            anchors.append(
                FigureAnchor(
                    line_index=line_index,
                    page_number=current_page,
                    figure_index=figure_index,
                    description=figure_match.group("description").strip(),
                )
            )
    return anchors


def replace_figure_lines(markdown: str, results: list[CropResult]) -> str:
    lines = markdown.splitlines()
    for result in sorted(results, key=lambda item: item.anchor.line_index, reverse=True):
        lines[result.anchor.line_index] = f"![{result.anchor.description}]({result.rel_path})"
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def _ink_bounds_by_rows(image: Image.Image) -> list[tuple[int, int, int, int]]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    row_bounds: list[tuple[int, int] | None] = []
    for y in range(height):
        xs = [x for x in range(width) if pixels[x, y] < 235]
        if xs:
            row_bounds.append((min(xs), max(xs)))
        else:
            row_bounds.append(None)

    bands: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        while y < height and row_bounds[y] is None:
            y += 1
        if y >= height:
            break
        start = y
        min_x = width
        max_x = 0
        blank_run = 0
        while y < height:
            bounds = row_bounds[y]
            if bounds is None:
                blank_run += 1
                if blank_run > 10:
                    break
            else:
                blank_run = 0
                min_x = min(min_x, bounds[0])
                max_x = max(max_x, bounds[1])
            y += 1
        end = max(start, y - blank_run - 1)
        if min_x <= max_x:
            bands.append((start, end, min_x, max_x))
    return bands


def _candidate_rects(page: fitz.Page, zoom: float) -> list[fitz.Rect]:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    mode = "RGB" if pix.n < 4 else "CMYK"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if image.mode != "RGB":
        image = image.convert("RGB")

    candidates: list[fitz.Rect] = []
    for y0, y1, x0, x1 in _ink_bounds_by_rows(image):
        band_width = x1 - x0 + 1
        band_height = y1 - y0 + 1
        if band_width < image.width * 0.45 or band_height < image.height * 0.035:
            continue
        pad_x = int(image.width * 0.02)
        pad_y = int(image.height * 0.01)
        rect = fitz.Rect(
            max(0, x0 - pad_x) / zoom,
            max(0, y0 - pad_y) / zoom,
            min(image.width, x1 + pad_x) / zoom,
            min(image.height, y1 + pad_y) / zoom,
        )
        if rect.width >= page.rect.width * 0.4 and rect.height >= page.rect.height * 0.03:
            candidates.append(rect)
    return candidates


def _full_page_rect(page: fitz.Page) -> fitz.Rect:
    return fitz.Rect(page.rect)


def crop_figures(
    pdf_path: str | Path,
    markdown_path: str | Path,
    assets_dir_name: str,
    zoom: float,
    jpeg_quality: int,
    mode: str,
    write_markdown: bool,
) -> list[CropResult]:
    markdown_path = Path(markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    anchors = parse_figure_anchors(markdown)
    assets_dir = markdown_writer.asset_dir_for(markdown_path, assets_dir_name)
    results: list[CropResult] = []

    doc = open_document(pdf_path)
    try:
        rects_by_page: dict[int, list[fitz.Rect]] = {}
        for anchor in anchors:
            page = doc[anchor.page_number - 1]
            if mode == "heuristic":
                rects_by_page.setdefault(anchor.page_number, _candidate_rects(page, zoom))
                page_rects = rects_by_page[anchor.page_number]
                fallback = anchor.figure_index > len(page_rects)
                rect = page_rects[anchor.figure_index - 1] if not fallback else _full_page_rect(page)
            else:
                fallback = False
                rect = _full_page_rect(page)

            asset_path = assets_dir / f"page{anchor.page_number:03d}_fig{anchor.figure_index:02d}.jpg"
            render_clip_to_file(page, rect, asset_path, zoom, jpeg_quality)
            results.append(
                CropResult(
                    anchor=anchor,
                    asset_path=asset_path,
                    rel_path=markdown_writer.relative_asset_path(markdown_path, asset_path),
                    rect=rect,
                    fallback=fallback,
                )
            )
    finally:
        doc.close()

    if write_markdown:
        markdown_path.write_text(replace_figure_lines(markdown, results), encoding="utf-8")
    return results
