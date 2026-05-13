from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import fitz

from .llm_client import LLMClient
from .pdf_reader import open_document, render_clip_b64, render_clip_to_file, render_page_b64


@dataclass(frozen=True)
class NormalizedRect:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class CropIteration:
    iteration: int
    rect: NormalizedRect
    done: bool
    confidence: float | None
    rationale: str


@dataclass(frozen=True)
class LlmCropResult:
    page_number: int
    final_rect: NormalizedRect
    output_path: Path
    iterations: tuple[CropIteration, ...]


@dataclass(frozen=True)
class LlmCropJob:
    page_number: int
    figure_index: str
    output_path: Path
    prompt: str
    start_rect: NormalizedRect | None = None
    json_output_path: Path | None = None
    save_iterations_dir: Path | None = None


@dataclass(frozen=True)
class LlmCropJobResult:
    job: LlmCropJob
    result: LlmCropResult | None
    error: str | None = None


ProgressCallback = Callable[[str], None]


def parse_normalized_rect(value: str) -> NormalizedRect:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("region must have four comma-separated values: x0,y0,x1,y1")
    try:
        rect = NormalizedRect(*(float(part) for part in parts))
    except ValueError as exc:
        raise ValueError("region values must be numbers") from exc
    return clamp_normalized_rect(rect)


def clamp_normalized_rect(rect: NormalizedRect) -> NormalizedRect:
    x0 = min(max(rect.x0, 0.0), 1.0)
    y0 = min(max(rect.y0, 0.0), 1.0)
    x1 = min(max(rect.x1, 0.0), 1.0)
    y1 = min(max(rect.y1, 0.0), 1.0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("region must have x1 > x0 and y1 > y0 after clamping")
    return NormalizedRect(x0, y0, x1, y1)


def normalized_to_pdf_rect(rect: NormalizedRect, page_rect: fitz.Rect) -> fitz.Rect:
    width = page_rect.width
    height = page_rect.height
    return fitz.Rect(
        page_rect.x0 + rect.x0 * width,
        page_rect.y0 + rect.y0 * height,
        page_rect.x0 + rect.x1 * width,
        page_rect.y0 + rect.y1 * height,
    )


def pdf_to_normalized_rect(rect: fitz.Rect, page_rect: fitz.Rect) -> NormalizedRect:
    width = page_rect.width
    height = page_rect.height
    return clamp_normalized_rect(
        NormalizedRect(
            (rect.x0 - page_rect.x0) / width,
            (rect.y0 - page_rect.y0) / height,
            (rect.x1 - page_rect.x0) / width,
            (rect.y1 - page_rect.y0) / height,
        )
    )


def rect_delta_ratio(left: NormalizedRect, right: NormalizedRect) -> float:
    return max(
        abs(left.x0 - right.x0),
        abs(left.y0 - right.y0),
        abs(left.x1 - right.x1),
        abs(left.y1 - right.y1),
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    elif not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("model response did not contain a JSON object")
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("model response JSON must be an object")
    return data


def parse_model_response(text: str) -> tuple[NormalizedRect, bool, float | None, str]:
    data = extract_json_object(text)
    raw_rect = data.get("rect")
    if not isinstance(raw_rect, dict):
        raise ValueError("model response must include rect object")
    try:
        rect = clamp_normalized_rect(
            NormalizedRect(
                float(raw_rect["x0"]),
                float(raw_rect["y0"]),
                float(raw_rect["x1"]),
                float(raw_rect["y1"]),
            )
        )
    except KeyError as exc:
        raise ValueError(f"model response rect missing {exc.args[0]}") from exc

    raw_confidence = data.get("confidence")
    confidence = None if raw_confidence is None else float(raw_confidence)
    done = bool(data.get("done", False))
    rationale = str(data.get("rationale", "")).strip()
    return rect, done, confidence, rationale


def _initial_prompt(user_prompt: str, page_number: int) -> str:
    return f"""You are choosing a crop rectangle on PDF page {page_number}.

Goal: {user_prompt}

The image is the full page. Return only JSON with this schema:
{{"rect":{{"x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0}},"done":false,"confidence":0.0,"rationale":"short reason"}}

Coordinates are normalized to the full page: origin is top-left, x grows right, y grows down, and all values must be between 0 and 1. Choose a rectangle that includes all target visual content with a small margin. Exclude unrelated prose, captions, and neighboring figures unless the goal asks for them.
"""


def _iteration_prompt(user_prompt: str, page_number: int, current_rect: NormalizedRect) -> str:
    return f"""You are refining a crop rectangle on PDF page {page_number}.

Goal: {user_prompt}

The image is the current crop from the full page. Its current full-page normalized rectangle is:
{{"x0":{current_rect.x0:.6f},"y0":{current_rect.y0:.6f},"x1":{current_rect.x1:.6f},"y1":{current_rect.y1:.6f}}}

Return only JSON with this schema:
{{"rect":{{"x0":0.0,"y0":0.0,"x1":1.0,"y1":1.0}},"done":true,"confidence":0.0,"rationale":"short reason"}}

The returned rect must still be in full-page normalized coordinates, not crop-local coordinates. If the current crop captures the target visual content with small margins and no important content cut off, return the current rectangle and set done=true. Otherwise adjust the rectangle to include missing target content or remove unrelated content.
"""


def write_result_json(path: str | Path, result: LlmCropResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "page_number": result.page_number,
        "output_path": str(result.output_path),
        "final_rect": result.final_rect.__dict__,
        "iterations": [
            {
                "iteration": item.iteration,
                "rect": item.rect.__dict__,
                "done": item.done,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in result.iterations
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _default_batch_prompt(figure_index: str, description: str) -> str:
    target = f"figure {figure_index}"
    if description:
        target += f": {description}"
    return (
        f"Crop {target} only on this page. Include the complete visual figure with a small margin. "
        "Exclude captions, page numbers, headers, footers, and surrounding prose."
    )


def read_llm_crop_jobs_tsv(
    path: str | Path,
    output_dir: str | Path,
    prompt_template: str | None = None,
    json_output: bool = True,
    default_region: NormalizedRect | None = None,
    save_iterations_dir: str | Path | None = None,
) -> list[LlmCropJob]:
    output_dir = Path(output_dir)
    save_root = Path(save_iterations_dir) if save_iterations_dir else None
    jobs: list[LlmCropJob] = []
    with Path(path).open(encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = "asset" in sample.splitlines()[0].split("\t") if sample.splitlines() else False
        if has_header:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                page = row.get("page") or row.get("page_number")
                figure = row.get("figure") or row.get("figure_index") or row.get("fig") or "1"
                asset = row.get("asset") or row.get("filename") or row.get("output")
                description = row.get("description") or row.get("prompt") or ""
                if not page or not asset:
                    raise ValueError("TSV rows must include page and asset columns")
                prompt = prompt_template.format(page=page, figure=figure, asset=asset, description=description) if prompt_template else _default_batch_prompt(figure, description)
                output_path = output_dir / asset
                jobs.append(
                    LlmCropJob(
                        page_number=int(page),
                        figure_index=str(figure),
                        output_path=output_path,
                        prompt=prompt,
                        start_rect=default_region,
                        json_output_path=output_path.with_suffix(".json") if json_output else None,
                        save_iterations_dir=(save_root / output_path.stem) if save_root else None,
                    )
                )
        else:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 4:
                    continue
                page = row[1]
                figure = row[2]
                asset = row[3]
                description = row[4] if len(row) > 4 else ""
                prompt = prompt_template.format(page=page, figure=figure, asset=asset, description=description) if prompt_template else _default_batch_prompt(figure, description)
                output_path = output_dir / asset
                jobs.append(
                    LlmCropJob(
                        page_number=int(page),
                        figure_index=str(figure),
                        output_path=output_path,
                        prompt=prompt,
                        start_rect=default_region,
                        json_output_path=output_path.with_suffix(".json") if json_output else None,
                        save_iterations_dir=(save_root / output_path.stem) if save_root else None,
                    )
                )
    return jobs


def find_crop_with_llm(
    pdf_path: str | Path,
    page_number: int,
    user_prompt: str,
    output_path: str | Path,
    model: str,
    start_rect: NormalizedRect | None = None,
    zoom: float = 2.0,
    jpeg_quality: int = 85,
    max_iterations: int = 6,
    min_change_ratio: float = 0.01,
    save_iterations_dir: str | Path | None = None,
    json_output_path: str | Path | None = None,
    verbose: bool = False,
) -> LlmCropResult:
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    doc = open_document(pdf_path)
    if page_number < 1 or page_number > doc.page_count:
        raise ValueError(f"page out of range: {page_number} (document has {doc.page_count} pages)")

    page = doc[page_number - 1]
    rect = start_rect or NormalizedRect(0.0, 0.0, 1.0, 1.0)
    client = LLMClient(model=model)
    iterations: list[CropIteration] = []
    save_dir = Path(save_iterations_dir) if save_iterations_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, max_iterations + 1):
        if iteration == 1:
            image_b64 = render_page_b64(page, zoom=zoom, jpeg_quality=jpeg_quality)
            prompt = _initial_prompt(user_prompt, page_number)
        else:
            pdf_rect = normalized_to_pdf_rect(rect, page.rect)
            image_b64 = render_clip_b64(page, pdf_rect, zoom=zoom, jpeg_quality=jpeg_quality)
            prompt = _iteration_prompt(user_prompt, page_number, rect)
            if save_dir:
                render_clip_to_file(page, pdf_rect, save_dir / f"iteration_{iteration - 1:02d}.jpg", zoom=zoom, jpeg_quality=jpeg_quality)

        response = client.vision_chat(prompt, image_b64)
        next_rect, done, confidence, rationale = parse_model_response(response)
        delta = rect_delta_ratio(rect, next_rect)
        rect = next_rect
        if delta < min_change_ratio and iteration > 1:
            done = True
        record = CropIteration(iteration, rect, done, confidence, rationale)
        iterations.append(record)
        if verbose:
            print(
                f"iteration {iteration}: rect={rect.x0:.4f},{rect.y0:.4f},{rect.x1:.4f},{rect.y1:.4f} "
                f"done={done} confidence={confidence} delta={delta:.4f} rationale={rationale}"
            )
        if done:
            break

    final_pdf_rect = normalized_to_pdf_rect(rect, page.rect)
    output_path = Path(output_path)
    render_clip_to_file(page, final_pdf_rect, output_path, zoom=zoom, jpeg_quality=jpeg_quality)
    result = LlmCropResult(page_number, rect, output_path, tuple(iterations))
    if json_output_path:
        write_result_json(json_output_path, result)
    return result


def run_llm_crop_jobs(
    pdf_path: str | Path,
    jobs: list[LlmCropJob],
    model: str,
    zoom: float = 2.0,
    jpeg_quality: int = 85,
    max_iterations: int = 6,
    min_change_ratio: float = 0.01,
    max_concurrency: int = 1,
    skip_existing: bool = False,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
) -> list[LlmCropJobResult]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    def emit(message: str) -> None:
        if progress:
            progress(message)

    def run_job(index: int, job: LlmCropJob) -> LlmCropJobResult:
        if skip_existing and job.output_path.exists() and (job.json_output_path is None or job.json_output_path.exists()):
            emit(f"[{index}/{len(jobs)}] skip {job.output_path.name}")
            return LlmCropJobResult(job=job, result=None)
        emit(f"[{index}/{len(jobs)}] start {job.output_path.name} page={job.page_number} figure={job.figure_index}")
        try:
            result = find_crop_with_llm(
                pdf_path=pdf_path,
                page_number=job.page_number,
                user_prompt=job.prompt,
                output_path=job.output_path,
                model=model,
                start_rect=job.start_rect,
                zoom=zoom,
                jpeg_quality=jpeg_quality,
                max_iterations=max_iterations,
                min_change_ratio=min_change_ratio,
                save_iterations_dir=job.save_iterations_dir,
                json_output_path=job.json_output_path,
                verbose=verbose,
            )
        except Exception as exc:
            emit(f"[{index}/{len(jobs)}] failed {job.output_path.name}: {exc}")
            return LlmCropJobResult(job=job, result=None, error=str(exc))
        emit(f"[{index}/{len(jobs)}] done {result.output_path}")
        return LlmCropJobResult(job=job, result=result)

    if not jobs:
        return []
    worker_count = min(max_concurrency, len(jobs))
    if worker_count == 1:
        return [run_job(index, job) for index, job in enumerate(jobs, 1)]

    results: list[LlmCropJobResult | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_job, index, job): index - 1 for index, job in enumerate(jobs, 1)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]
