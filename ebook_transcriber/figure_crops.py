from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from . import markdown_writer
from .pdf_reader import open_document, render_clip_to_file

_PAGE_MARKER_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*$")
_FIGURE_RE = re.compile(r"^\[Figure:\s*(?P<description>.*?)\]\s*$")
_EXCLUDED_LOC_CLASSES = {2, 3, 8, 11, 12}


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
    merged_anchors: tuple[FigureAnchor, ...] = ()


@dataclass(frozen=True)
class LocModelConfig:
    starry_ocr_root: Path
    weights_path: Path
    exp_path: Path
    image_short_side: int


@dataclass(frozen=True)
class LocCropPass:
    row_ink_min: float
    row_text_max: float
    dense_row_ink_min: float | None
    dense_row_text_max: float | None
    run_min_height_ratio: float
    run_min_width_ratio: float
    column_ink_min: float
    cluster_gap_ratio: float
    cluster_max_height_ratio: float
    block_min_height_ratio: float
    block_max_height_ratio: float
    pad_x_ratio: float
    pad_y_ratio: float


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
        for merged_anchor in sorted(result.merged_anchors, key=lambda item: item.line_index, reverse=True):
            if merged_anchor.line_index != result.anchor.line_index:
                lines.pop(merged_anchor.line_index)
        lines[result.anchor.line_index] = f"![{result.anchor.description}]({result.rel_path})"
    return "\n".join(lines) + ("\n" if markdown.endswith("\n") else "")


def loc_model_config_from_env() -> LocModelConfig | None:
    weights = os.environ.get("OCR_LOC_WEIGHTS_PATH", "").strip()
    starry_root = os.environ.get("OCR_LOC_STARRY_OCR_ROOT", "").strip()
    if not weights or not starry_root:
        return None

    root = Path(starry_root)
    exp_path = Path(
        os.environ.get(
            "OCR_LOC_EXP_PATH",
            str(root / "DB_gc_loc/experiments/seg_detector/totaltext_resnet18_deform_thre-eval.yaml"),
        )
    )
    return LocModelConfig(
        starry_ocr_root=root,
        weights_path=Path(weights),
        exp_path=exp_path,
        image_short_side=int(os.environ.get("OCR_LOC_IMAGE_SHORT_SIDE", "736")),
    )


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


def _render_page_image(page: fitz.Page, zoom: float) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    mode = "RGB" if pix.n < 4 else "CMYK"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _candidate_rects_from_ink(page: fitz.Page, zoom: float) -> list[fitz.Rect]:
    image = _render_page_image(page, zoom)
    candidates: list[fitz.Rect] = []
    for y0, y1, x0, x1 in _ink_bounds_by_rows(image):
        band_width = x1 - x0 + 1
        band_height = y1 - y0 + 1
        if band_height < image.height * 0.075:
            continue
        if band_height > image.height * 0.45:
            continue
        if band_width < image.width * 0.30:
            continue
        aspect_ratio = band_width / band_height
        if aspect_ratio > 5.5:
            continue
        pad_x = int(image.width * 0.02)
        pad_y = int(image.height * 0.012)
        rect = fitz.Rect(
            max(0, x0 - pad_x) / zoom,
            max(0, y0 - pad_y) / zoom,
            min(image.width, x1 + pad_x) / zoom,
            min(image.height, y1 + pad_y) / zoom,
        )
        if rect.height >= page.rect.height * 0.06:
            candidates.append(rect)
    return candidates


def _candidate_rects(page: fitz.Page, zoom: float) -> list[fitz.Rect]:
    return _candidate_rects_from_ink(page, zoom)


def _install_loc_paths(config: LocModelConfig) -> None:
    if str(config.starry_ocr_root) not in sys.path:
        sys.path.insert(0, str(config.starry_ocr_root))


@lru_cache(maxsize=1)
def _load_loc_model(config: LocModelConfig) -> tuple[Any, Any, Any, Any, Any]:
    _install_loc_paths(config)
    import cv2
    import numpy as np
    import torch
    from DB_gc_loc.concern.config import Config, Configurable

    original_cwd = Path.cwd()
    os.chdir(config.starry_ocr_root)
    try:
        args = {
            "exp": str(config.exp_path),
            "resume": str(config.weights_path),
            "image_short_side": config.image_short_side,
            "box_thresh": 0.01,
            "polygon": False,
            "visualize": False,
            "resize": False,
            "verbose": False,
        }
        conf = Config()
        experiment_args = conf.compile(conf.load(args["exp"]))["Experiment"]
        experiment_args.update(cmd=args)
        experiment = Configurable.construct_class_from_config(experiment_args)
        experiment.load("evaluation", **experiment_args)
        structure = experiment.structure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = structure.builder.build(device)
        states = torch.load(config.weights_path, map_location=device)
        model.load_state_dict(states, strict=False)
        model.eval()
        return model, device, cv2, np, torch
    finally:
        os.chdir(original_cwd)


def _text_heatmap_for_image(image: Image.Image, config: LocModelConfig):
    import torch.nn.functional as F

    model, device, cv2, np, torch = _load_loc_model(config)
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_h < orig_w:
        new_h = config.image_short_side
        new_w = int(math.ceil(new_h / orig_h * orig_w / 32) * 32)
    else:
        new_w = config.image_short_side
        new_h = int(math.ceil(new_w / orig_w * orig_h / 32) * 32)

    rgb_mean = np.array([122.67891434, 116.66876762, 104.00698793])
    resized = cv2.resize(image_bgr.astype("float32"), (new_w, new_h))
    normalized = (resized - rgb_mean) / 255.0
    img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    batch = {"shape": [(orig_h, orig_w)], "image": img_tensor}
    with torch.no_grad():
        pred, mcls = model.forward(batch, training=False)
        binary = pred[0, 0].detach().float().cpu().numpy()
        classes = F.softmax(mcls, dim=1)[0].detach().float().cpu().numpy().argmax(axis=0).astype(np.uint8)

    binary_orig = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    classes_orig = cv2.resize(classes, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    keep_mask = ~np.isin(classes_orig, list(_EXCLUDED_LOC_CLASSES))
    return binary_orig * keep_mask.astype(np.float32)


_STRICT_LOC_CROP_PASS = LocCropPass(
    row_ink_min=0.0025,
    row_text_max=0.035,
    dense_row_ink_min=None,
    dense_row_text_max=None,
    run_min_height_ratio=0.016,
    run_min_width_ratio=0.12,
    column_ink_min=0.003,
    cluster_gap_ratio=0.045,
    cluster_max_height_ratio=0.23,
    block_min_height_ratio=0.025,
    block_max_height_ratio=0.28,
    pad_x_ratio=0.012,
    pad_y_ratio=0.006,
)
_RELAXED_LOC_CROP_PASS = LocCropPass(
    row_ink_min=0.0025,
    row_text_max=0.02,
    dense_row_ink_min=0.035,
    dense_row_text_max=0.22,
    run_min_height_ratio=0.006,
    run_min_width_ratio=0.08,
    column_ink_min=0.002,
    cluster_gap_ratio=0.030,
    cluster_max_height_ratio=0.40,
    block_min_height_ratio=0.012,
    block_max_height_ratio=0.45,
    pad_x_ratio=0.012,
    pad_y_ratio=0.006,
)


def _rect_overlap_over_smaller(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    if intersection.is_empty or first.is_empty or second.is_empty:
        return 0.0
    smaller_area = min(first.get_area(), second.get_area())
    if smaller_area == 0:
        return 0.0
    return intersection.get_area() / smaller_area


def _merge_relaxed_rects(strict_rects: list[fitz.Rect], relaxed_rects: list[fitz.Rect], expected_count: int) -> list[fitz.Rect]:
    if len(strict_rects) >= expected_count:
        return strict_rects

    selected = list(strict_rects)
    relaxed_only = [
        rect
        for rect in relaxed_rects
        if all(_rect_overlap_over_smaller(rect, selected_rect) < 0.65 for selected_rect in selected)
    ]
    needed = expected_count - len(selected)
    selected.extend(sorted(relaxed_only, key=lambda rect: rect.height, reverse=True)[:needed])
    return sorted(selected, key=lambda rect: rect.y0)


def _candidate_rects_from_loc_heatmap_pass(
    page: fitz.Page,
    image: Image.Image,
    zoom: float,
    ink,
    row_ink_density,
    row_text_density,
    crop_pass: LocCropPass,
) -> list[fitz.Rect]:
    import numpy as np

    music_rows = (row_ink_density >= crop_pass.row_ink_min) & (row_text_density <= crop_pass.row_text_max)
    if crop_pass.dense_row_ink_min is not None and crop_pass.dense_row_text_max is not None:
        music_rows = music_rows | (
            (row_ink_density >= crop_pass.dense_row_ink_min) & (row_text_density <= crop_pass.dense_row_text_max)
        )

    runs: list[tuple[int, int, int, int]] = []
    y = 0
    while y < image.height:
        while y < image.height and not music_rows[y]:
            y += 1
        start = y
        while y < image.height and music_rows[y]:
            y += 1
        end = y - 1
        if start >= image.height:
            break

        run_height = end - start + 1
        if run_height < max(8, int(image.height * crop_pass.run_min_height_ratio)):
            continue
        if start <= image.height * 0.03 or end >= image.height * 0.96:
            continue

        segment_ink = ink[start : end + 1, :]
        ink_columns = segment_ink.mean(axis=0) > crop_pass.column_ink_min
        xs = np.flatnonzero(ink_columns)
        if len(xs) == 0:
            continue
        x0 = int(xs[0])
        x1 = int(xs[-1])
        if x1 - x0 + 1 < image.width * crop_pass.run_min_width_ratio:
            continue
        runs.append((start, end, x0, x1))

    if not runs:
        return []

    clusters: list[list[tuple[int, int, int, int]]] = []
    for run in runs:
        if not clusters:
            clusters.append([run])
            continue
        previous = clusters[-1][-1]
        gap = run[0] - previous[1]
        current_start = clusters[-1][0][0]
        merged_height = run[1] - current_start + 1
        if gap <= image.height * crop_pass.cluster_gap_ratio and merged_height <= image.height * crop_pass.cluster_max_height_ratio:
            clusters[-1].append(run)
        else:
            clusters.append([run])

    candidates: list[fitz.Rect] = []
    for cluster in clusters:
        start = min(run[0] for run in cluster)
        end = max(run[1] for run in cluster)
        x0 = min(run[2] for run in cluster)
        x1 = max(run[3] for run in cluster)
        block_height = end - start + 1
        if block_height < image.height * crop_pass.block_min_height_ratio:
            continue
        if block_height > image.height * crop_pass.block_max_height_ratio:
            continue
        if row_text_density[start : end + 1].mean() > 0.02 and row_ink_density[start : end + 1].mean() < 0.02:
            continue

        pad_x = int(image.width * crop_pass.pad_x_ratio)
        pad_y = int(image.height * crop_pass.pad_y_ratio)
        rect = fitz.Rect(
            max(0, x0 - pad_x) / zoom,
            max(0, start - pad_y) / zoom,
            min(image.width, x1 + pad_x) / zoom,
            min(image.height, end + pad_y) / zoom,
        )
        if rect.height >= page.rect.height * crop_pass.block_min_height_ratio:
            candidates.append(rect)
    return candidates


def _candidate_rects_from_loc_heatmap(
    page: fitz.Page, zoom: float, config: LocModelConfig, expected_count: int | None = None
) -> list[fitz.Rect]:
    import numpy as np

    image = _render_page_image(page, zoom)
    text_heat = _text_heatmap_for_image(image, config)
    text_mask = text_heat >= 0.25
    row_text_density = text_mask.mean(axis=1)
    text_window = max(9, int(image.height * 0.012))
    row_text_density = np.convolve(row_text_density, np.ones(text_window) / text_window, mode="same")

    grayscale = np.asarray(image.convert("L"))
    ink = grayscale < 235
    row_ink_density = ink.mean(axis=1)

    strict_rects = _candidate_rects_from_loc_heatmap_pass(
        page, image, zoom, ink, row_ink_density, row_text_density, _STRICT_LOC_CROP_PASS
    )
    if expected_count is None or len(strict_rects) >= expected_count:
        return strict_rects

    relaxed_rects = _candidate_rects_from_loc_heatmap_pass(
        page, image, zoom, ink, row_ink_density, row_text_density, _RELAXED_LOC_CROP_PASS
    )
    return _merge_relaxed_rects(strict_rects, relaxed_rects, expected_count)


def _heuristic_rects(
    page: fitz.Page, zoom: float, loc_config: LocModelConfig | None, expected_count: int | None = None
) -> list[fitz.Rect]:
    if loc_config is None:
        return _candidate_rects_from_ink(page, zoom)
    try:
        return _candidate_rects_from_loc_heatmap(page, zoom, loc_config, expected_count)
    except Exception:
        return _candidate_rects_from_ink(page, zoom)


def _full_page_rect(page: fitz.Page) -> fitz.Rect:
    return fitz.Rect(page.rect)


def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect:
    return fitz.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )


def _assign_rects_to_figures(rects: list[fitz.Rect], figure_count: int) -> list[fitz.Rect]:
    if len(rects) <= figure_count:
        return rects
    selected = sorted(rects, key=lambda rect: rect.height, reverse=True)[:figure_count]
    return sorted(selected, key=lambda rect: rect.y0)


def _group_consecutive_anchors_for_rects(
    markdown_lines: list[str], page_anchors: list[FigureAnchor], rect_count: int
) -> list[tuple[FigureAnchor, ...]]:
    groups: list[tuple[FigureAnchor, ...]] = [(anchor,) for anchor in page_anchors]
    if rect_count <= 0 or len(groups) <= rect_count:
        return groups

    while len(groups) > rect_count:
        merge_index: int | None = None
        for index in range(len(groups) - 1):
            first = groups[index][-1]
            second = groups[index + 1][0]
            if second.figure_index != first.figure_index + 1:
                continue
            between = markdown_lines[first.line_index + 1 : second.line_index]
            if all(not line.strip() for line in between):
                merge_index = index
                break
        if merge_index is None:
            break
        groups[merge_index] = groups[merge_index] + groups[merge_index + 1]
        del groups[merge_index + 1]
    return groups


def crop_figures(
    pdf_path: str | Path,
    markdown_path: str | Path,
    assets_dir_name: str,
    zoom: float,
    jpeg_quality: int,
    mode: str,
    write_markdown: bool,
    loc_model_config: LocModelConfig | None = None,
) -> list[CropResult]:
    markdown_path = Path(markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    anchors = parse_figure_anchors(markdown)
    assets_dir = markdown_writer.asset_dir_for(markdown_path, assets_dir_name)
    results: list[CropResult] = []

    if loc_model_config is None:
        loc_model_config = loc_model_config_from_env()

    doc = open_document(pdf_path)
    try:
        rects_by_page: dict[int, list[fitz.Rect]] = {}
        anchors_by_page: dict[int, list[FigureAnchor]] = {}
        for anchor in anchors:
            anchors_by_page.setdefault(anchor.page_number, []).append(anchor)

        markdown_lines = markdown.splitlines()
        for page_number, page_anchors in anchors_by_page.items():
            page = doc[page_number - 1]
            if mode == "heuristic":
                if page_number not in rects_by_page:
                    rects_by_page[page_number] = _assign_rects_to_figures(
                        _heuristic_rects(
                            page,
                            zoom,
                            loc_model_config,
                            len(page_anchors),
                        ),
                        len(page_anchors),
                    )
                page_rects = rects_by_page[page_number]
                anchor_groups = _group_consecutive_anchors_for_rects(markdown_lines, page_anchors, len(page_rects))
            else:
                page_rects = [_full_page_rect(page) for _ in page_anchors]
                anchor_groups = [(anchor,) for anchor in page_anchors]

            for group_index, anchor_group in enumerate(anchor_groups):
                anchor = anchor_group[0]
                if mode == "heuristic" and group_index >= len(page_rects):
                    fallback = True
                    rect = _full_page_rect(page)
                else:
                    fallback = False
                    rect = page_rects[group_index]

                asset_path = assets_dir / f"page{anchor.page_number:03d}_fig{anchor.figure_index:02d}.jpg"
                render_clip_to_file(page, rect, asset_path, zoom, jpeg_quality)
                results.append(
                    CropResult(
                        anchor=anchor,
                        asset_path=asset_path,
                        rel_path=markdown_writer.relative_asset_path(markdown_path, asset_path),
                        rect=rect,
                        fallback=fallback,
                        merged_anchors=anchor_group,
                    )
                )
    finally:
        doc.close()

    if write_markdown:
        markdown_path.write_text(replace_figure_lines(markdown, results), encoding="utf-8")
    return results
