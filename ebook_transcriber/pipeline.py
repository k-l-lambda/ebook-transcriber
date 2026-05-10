from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
from tqdm import tqdm

from . import markdown_writer
from .llm_client import LLMClient
from .pdf_reader import (
    PageInventory,
    inspect_page,
    open_document,
    parse_page_range,
    render_clip_b64,
    render_clip_to_file,
    render_page_b64,
)
from .prompts import IMAGE_CLASSIFY_PROMPT, IMAGE_OCR_PROMPT, PAGE_TO_MARKDOWN_PROMPT, VISION_SMOKE_PROMPT


@dataclass(frozen=True)
class ConvertOptions:
    pdf_path: Path
    output_path: Path
    assets_dir_name: str
    model: str
    pages: str | None
    zoom: float
    jpeg_quality: int
    dry_run: bool
    verbose: bool


def _significant_region(region) -> bool:
    return region.width >= 120 and region.height >= 120 and region.rect.width >= 40 and region.rect.height >= 40


def _inventory_line(inventory: PageInventory) -> str:
    return (
        f"page {inventory.page_number}: text_blocks={inventory.text_blocks} "
        f"images={len(inventory.image_regions)} vector_drawings={inventory.vector_drawings}"
    )


def convert_pdf(options: ConvertOptions) -> Path:
    doc = open_document(options.pdf_path)
    try:
        page_indexes = parse_page_range(options.pages, doc.page_count)
        inventories = [inspect_page(doc[index], index) for index in page_indexes]

        if options.dry_run:
            for inventory in inventories:
                print(_inventory_line(inventory))
            return options.output_path

        client = LLMClient(model=options.model)
        if not page_indexes:
            raise ValueError("no pages selected")

        smoke_page = doc[page_indexes[0]]
        smoke_image = render_page_b64(smoke_page, options.zoom, options.jpeg_quality)
        if options.verbose:
            print(f"vision smoke test: model={options.model} page={page_indexes[0] + 1}")
        client.vision_chat(VISION_SMOKE_PROMPT, smoke_image)

        assets_dir = markdown_writer.asset_dir_for(options.output_path, options.assets_dir_name)
        parts: list[str] = []
        iterator = tqdm(page_indexes, desc="pages") if not options.verbose else page_indexes
        for page_index in iterator:
            page = doc[page_index]
            inventory = inspect_page(page, page_index)
            if options.verbose:
                print(_inventory_line(inventory))

            page_image = render_page_b64(page, options.zoom, options.jpeg_quality)
            page_markdown = client.vision_chat(PAGE_TO_MARKDOWN_PROMPT, page_image).strip()

            for region in inventory.image_regions:
                if not _significant_region(region):
                    continue
                clip_b64 = render_clip_b64(page, region.rect, options.zoom, options.jpeg_quality)
                classification = client.vision_chat(IMAGE_CLASSIFY_PROMPT, clip_b64).strip().upper()
                if classification.startswith("TEXT"):
                    ocr = client.vision_chat(IMAGE_OCR_PROMPT, clip_b64).strip()
                    if ocr:
                        page_markdown = f"{page_markdown.rstrip()}\n\n{ocr}"
                    continue

                asset_name = f"page{page_index + 1:03d}_img{region.index:02d}.jpg"
                asset_path = assets_dir / asset_name
                render_clip_to_file(page, region.rect, asset_path, options.zoom, options.jpeg_quality)
                rel_path = markdown_writer.relative_asset_path(options.output_path, asset_path)
                page_markdown = markdown_writer.replace_first_asset_placeholder(
                    page_markdown,
                    rel_path,
                    alt=f"page {page_index + 1} image {region.index}",
                )

            parts.append(f"<!-- page {page_index + 1} -->\n\n{page_markdown}")

        markdown_writer.write_markdown(options.output_path, parts)
        return options.output_path
    finally:
        doc.close()
