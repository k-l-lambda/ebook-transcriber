from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image


@dataclass(frozen=True)
class ImageRegion:
    index: int
    rect: fitz.Rect
    width: int
    height: int


@dataclass(frozen=True)
class PageInventory:
    page_number: int
    text_blocks: int
    image_regions: tuple[ImageRegion, ...]
    vector_drawings: int


def open_document(path: str | Path) -> fitz.Document:
    return fitz.open(Path(path))


def parse_page_range(spec: str | None, page_count: int) -> list[int]:
    if not spec:
        return list(range(page_count))

    pages: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if start > end:
                raise ValueError(f"invalid descending page range: {part}")
            pages.update(range(start - 1, end))
        else:
            pages.add(int(part) - 1)

    invalid = [page + 1 for page in sorted(pages) if page < 0 or page >= page_count]
    if invalid:
        raise ValueError(f"page out of range: {invalid[0]} (document has {page_count} pages)")
    return sorted(pages)


def inspect_page(page: fitz.Page, page_index: int) -> PageInventory:
    text_blocks = [block for block in page.get_text("blocks") if len(block) > 6 and block[6] == 0]
    image_regions: list[ImageRegion] = []

    for xref_index, image_info in enumerate(page.get_images(full=True)):
        xref = image_info[0]
        width = int(image_info[2])
        height = int(image_info[3])
        for rect in page.get_image_rects(xref):
            image_regions.append(ImageRegion(len(image_regions), rect, width, height))

    return PageInventory(
        page_number=page_index + 1,
        text_blocks=len(text_blocks),
        image_regions=tuple(image_regions),
        vector_drawings=len(page.get_drawings()),
    )


def _pixmap_to_jpeg_bytes(pix: fitz.Pixmap, jpeg_quality: int) -> bytes:
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    mode = "RGB" if pix.n < 4 else "CMYK"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def render_page_b64(page: fitz.Page, zoom: float, jpeg_quality: int) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return base64.b64encode(_pixmap_to_jpeg_bytes(pix, jpeg_quality)).decode("ascii")


def render_clip_b64(page: fitz.Page, rect: fitz.Rect, zoom: float, jpeg_quality: int) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    return base64.b64encode(_pixmap_to_jpeg_bytes(pix, jpeg_quality)).decode("ascii")


def render_clip_to_file(page: fitz.Page, rect: fitz.Rect, output_path: str | Path, zoom: float, jpeg_quality: int = 90) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    output_path.write_bytes(_pixmap_to_jpeg_bytes(pix, jpeg_quality))
