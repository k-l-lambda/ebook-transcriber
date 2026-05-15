from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .config import env_float, env_int, env_str, load_project_env
from .figure_crops import crop_figures
from .llm_crop import find_crop_with_llm, parse_normalized_rect, read_llm_crop_jobs_tsv, run_llm_crop_jobs
from .markdown_translator import TranslateMarkdownOptions, translate_markdown_tree
from .markdown_writer import default_output_path
from .pipeline import ConvertOptions, convert_pdf
from .segments import read_segments, safe_segment_filename, write_index_markdown


load_project_env()


@click.group()
def main() -> None:
    """Convert PDF pages to Markdown with an LLM vision/OCR API."""


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_path", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output Markdown path.")
@click.option("--assets-dir", default=lambda: env_str("ASSETS_DIR", "assets"), show_default="env ASSETS_DIR or assets", help="Assets subdirectory beside the Markdown file.")
@click.option("--model", default=lambda: env_str("MODEL", "deepseek/deepseek-v4-pro"), show_default="env MODEL or deepseek/deepseek-v4-pro", help="OpenAI-compatible model name.")
@click.option("--pages", default=None, help="Page range, e.g. 1, 1-5, or 1,3,8-10. Pages are 1-based.")
@click.option("--zoom", default=lambda: env_float("ZOOM", 2.0), show_default="env ZOOM or 2.0", type=float, help="PDF render zoom factor.")
@click.option("--jpeg-quality", default=lambda: env_int("JPEG_QUALITY", 85), show_default="env JPEG_QUALITY or 85", type=click.IntRange(1, 100), help="JPEG quality for rendered page images.")
@click.option("--output-language", default=lambda: env_str("OUTPUT_LANGUAGE", ""), show_default="env OUTPUT_LANGUAGE or empty", help="Translate transcribed text to this language when non-empty, e.g. zh.")
@click.option("--restart", is_flag=True, help="Clear the output Markdown and checkpoint before converting selected pages.")
@click.option("--dry-run", is_flag=True, help="Inspect pages without calling the API.")
@click.option("--verbose", "-v", is_flag=True, help="Print per-page progress.")
def convert(
    pdf_path: Path,
    output_path: Path | None,
    assets_dir: str,
    model: str,
    pages: str | None,
    zoom: float,
    jpeg_quality: int,
    output_language: str,
    restart: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Convert PDF_PATH to Markdown."""
    resolved_output = output_path or default_output_path(pdf_path, env_str("OUTPUT_DIR", "output"))
    options = ConvertOptions(
        pdf_path=pdf_path,
        output_path=resolved_output,
        assets_dir_name=assets_dir,
        model=model,
        pages=pages,
        zoom=zoom,
        jpeg_quality=jpeg_quality,
        output_language=output_language.strip() or None,
        restart=restart,
        dry_run=dry_run,
        verbose=verbose,
    )
    try:
        result = convert_pdf(options)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if dry_run:
        click.echo(f"dry run complete; output would be {result}")
    else:
        click.echo(f"wrote {result}")


@main.command("convert-segments")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("segments_yaml", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=lambda: Path(env_str("OUTPUT_DIR", "output")), show_default="env OUTPUT_DIR or output", help="Directory for per-segment Markdown files.")
@click.option("--assets-dir", default=lambda: env_str("ASSETS_DIR", "assets"), show_default="env ASSETS_DIR or assets", help="Assets subdirectory beside each Markdown file.")
@click.option("--model", default=lambda: env_str("MODEL", "deepseek/deepseek-v4-pro"), show_default="env MODEL or deepseek/deepseek-v4-pro", help="OpenAI-compatible model name.")
@click.option("--segment", "segment_ids", multiple=True, help="Only transcribe this segment id. Can be repeated.")
@click.option("--zoom", default=lambda: env_float("ZOOM", 2.0), show_default="env ZOOM or 2.0", type=float, help="PDF render zoom factor.")
@click.option("--jpeg-quality", default=lambda: env_int("JPEG_QUALITY", 85), show_default="env JPEG_QUALITY or 85", type=click.IntRange(1, 100), help="JPEG quality for rendered page images.")
@click.option("--output-language", default=lambda: env_str("OUTPUT_LANGUAGE", ""), show_default="env OUTPUT_LANGUAGE or empty", help="Translate transcribed text to this language when non-empty, e.g. zh.")
@click.option("--max-concurrency", default=lambda: env_int("MAX_CONCURRENCY", 10), show_default="env MAX_CONCURRENCY or 10", type=click.IntRange(1), help="Maximum number of segments to transcribe concurrently.")
@click.option("--restart", is_flag=True, help="Clear each selected segment output and checkpoint before converting.")
@click.option("--dry-run", is_flag=True, help="List segment outputs and page ranges without calling the API.")
@click.option("--verbose", "-v", is_flag=True, help="Print per-segment and per-page progress.")
def convert_segments(
    pdf_path: Path,
    segments_yaml: Path,
    output_dir: Path,
    assets_dir: str,
    model: str,
    segment_ids: tuple[str, ...],
    zoom: float,
    jpeg_quality: int,
    output_language: str,
    max_concurrency: int,
    restart: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Convert PDF_PATH by segment ranges from SEGMENTS_YAML."""
    segments = read_segments(segments_yaml)
    wanted = set(segment_ids)
    selected = [segment for segment in segments if not wanted or segment.id in wanted]
    missing = wanted - {segment.id for segment in selected}
    if missing:
        raise click.ClickException(f"segment id not found: {', '.join(sorted(missing))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        index_path = write_index_markdown(output_dir, segments)
    except FileNotFoundError as exc:
        click.echo(f"index not written: {exc}", err=True)
    else:
        click.echo(f"wrote index {index_path}")

    def run_segment(segment) -> Path:
        start, end = segment.pdf_pages
        output_path = output_dir / f"{safe_segment_filename(segment.id)}.md"
        if verbose:
            click.echo(f"segment {segment.id}: pages {start}-{end} -> {output_path}")
        options = ConvertOptions(
            pdf_path=pdf_path,
            output_path=output_path,
            assets_dir_name=assets_dir,
            model=model,
            pages=f"{start}-{end}",
            zoom=zoom,
            jpeg_quality=jpeg_quality,
            output_language=output_language.strip() or None,
            restart=restart,
            dry_run=False,
            verbose=verbose,
        )
        return convert_pdf(options)

    for segment in selected:
        start, end = segment.pdf_pages
        output_path = output_dir / f"{safe_segment_filename(segment.id)}.md"
        click.echo(f"segment {segment.id}: pages {start}-{end} -> {output_path}")
    if dry_run:
        return

    worker_count = min(max_concurrency, len(selected))
    if worker_count <= 1:
        for segment in selected:
            try:
                run_segment(segment)
            except Exception as exc:
                raise click.ClickException(f"segment {segment.id} failed: {exc}") from exc
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_segment, segment): segment for segment in selected}
        for future in as_completed(futures):
            segment = futures[future]
            try:
                result = future.result()
                click.echo(f"segment {segment.id} done -> {result}")
            except Exception as exc:
                failures.append(f"{segment.id}: {exc}")
                click.echo(f"segment {segment.id} failed: {exc}", err=True)
    if failures:
        raise click.ClickException("segment failures: " + "; ".join(failures))


@main.command("translate-markdown")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--model", default=lambda: env_str("MODEL", "deepseek/deepseek-v4-pro"), show_default="env MODEL or deepseek/deepseek-v4-pro", help="OpenAI-compatible model name.")
@click.option("--output-language", default=lambda: env_str("OUTPUT_LANGUAGE", "zh"), show_default="env OUTPUT_LANGUAGE or zh", help="Translate Markdown prose to this language.")
@click.option("--pages", default=None, help="Page range within each Markdown file, e.g. 1, 1-5, or 1,3,8-10.")
@click.option("--max-concurrency", default=lambda: env_int("MAX_CONCURRENCY", 3), show_default="env MAX_CONCURRENCY or 3", type=click.IntRange(1), help="Maximum number of Markdown files to translate concurrently.")
@click.option("--restart", is_flag=True, help="Clear translated Markdown and checkpoints before translating.")
@click.option("--dry-run", is_flag=True, help="List files/pages without calling the API.")
@click.option("--verbose", "-v", is_flag=True, help="Print per-file and per-page progress.")
def translate_markdown_command(
    input_path: Path,
    output_path: Path,
    model: str,
    output_language: str,
    pages: str | None,
    max_concurrency: int,
    restart: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Translate Markdown file(s) page-by-page with the text LLM API."""
    options = TranslateMarkdownOptions(
        input_path=input_path,
        output_path=output_path,
        model=model,
        output_language=output_language,
        pages=pages,
        restart=restart,
        dry_run=dry_run,
        verbose=verbose,
        max_concurrency=max_concurrency,
    )
    try:
        outputs = translate_markdown_tree(options, progress=click.echo)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    action = "dry run complete" if dry_run else "translated"
    click.echo(f"{action} {len(outputs)} file(s) -> {output_path}")


@main.command("crop-figures")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("markdown_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--assets-dir", default=lambda: env_str("ASSETS_DIR", "assets"), show_default="env ASSETS_DIR or assets", help="Assets subdirectory beside the Markdown file.")
@click.option("--zoom", default=lambda: env_float("ZOOM", 2.0), show_default="env ZOOM or 2.0", type=float, help="PDF render zoom factor.")
@click.option("--jpeg-quality", default=lambda: env_int("JPEG_QUALITY", 85), show_default="env JPEG_QUALITY or 85", type=click.IntRange(1, 100), help="JPEG quality for cropped figure images.")
@click.option("--mode", type=click.Choice(["heuristic", "page"]), default="heuristic", show_default=True, help="Crop heuristic figure bands or full pages.")
@click.option("--write", "write_markdown", is_flag=True, help="Replace [Figure: ...] lines with Markdown image links.")
@click.option("--verbose", "-v", is_flag=True, help="Print each generated crop.")
def crop_figures_command(
    pdf_path: Path,
    markdown_path: Path,
    assets_dir: str,
    zoom: float,
    jpeg_quality: int,
    mode: str,
    write_markdown: bool,
    verbose: bool,
) -> None:
    """Crop screenshots for [Figure: ...] blocks in MARKDOWN_PATH."""
    try:
        results = crop_figures(
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            assets_dir_name=assets_dir,
            zoom=zoom,
            jpeg_quality=jpeg_quality,
            mode=mode,
            write_markdown=write_markdown,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for result in results:
        if verbose or not write_markdown:
            note = " fallback=page" if result.fallback else ""
            click.echo(
                f"page {result.anchor.page_number} figure {result.anchor.figure_index} -> "
                f"{result.asset_path}{note}"
            )
    action = "cropped and updated" if write_markdown else "cropped"
    click.echo(f"{action} {len(results)} figure(s) for {markdown_path}")


@main.command("llm-crop")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("page", type=click.IntRange(1))
@click.option("--prompt", "user_prompt", required=True, help="Description of the target visual content to crop.")
@click.option("--output", "output_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Output JPEG crop path.")
@click.option("--region", default="0,0,1,1", show_default=True, help="Starting normalized full-page rectangle x0,y0,x1,y1.")
@click.option("--model", default=lambda: env_str("MODEL", "deepseek/deepseek-v4-pro"), show_default="env MODEL or deepseek/deepseek-v4-pro", help="OpenAI-compatible vision model name.")
@click.option("--zoom", default=lambda: env_float("ZOOM", 2.0), show_default="env ZOOM or 2.0", type=float, help="PDF render zoom factor.")
@click.option("--jpeg-quality", default=lambda: env_int("JPEG_QUALITY", 85), show_default="env JPEG_QUALITY or 85", type=click.IntRange(1, 100), help="JPEG quality for rendered crop images.")
@click.option("--max-iterations", default=6, show_default=True, type=click.IntRange(1), help="Maximum LLM refinement iterations.")
@click.option("--min-change-ratio", default=0.01, show_default=True, type=float, help="Stop when normalized rectangle changes less than this amount.")
@click.option("--save-iterations", type=click.Path(file_okay=False, path_type=Path), help="Directory for intermediate candidate crop images.")
@click.option("--json-output", type=click.Path(dir_okay=False, path_type=Path), help="Optional JSON metadata output path.")
@click.option("--verbose", "verbose", "-v", is_flag=True, help="Print per-iteration rectangle and rationale.")
def llm_crop_command(
    pdf_path: Path,
    page: int,
    user_prompt: str,
    output_path: Path,
    region: str,
    model: str,
    zoom: float,
    jpeg_quality: int,
    max_iterations: int,
    min_change_ratio: float,
    save_iterations: Path | None,
    json_output: Path | None,
    verbose: bool,
) -> None:
    """Iteratively find a PDF crop rectangle with a multimodal LLM."""
    try:
        start_rect = parse_normalized_rect(region)
        result = find_crop_with_llm(
            pdf_path=pdf_path,
            page_number=page,
            user_prompt=user_prompt,
            output_path=output_path,
            model=model,
            start_rect=start_rect,
            zoom=zoom,
            jpeg_quality=jpeg_quality,
            max_iterations=max_iterations,
            min_change_ratio=min_change_ratio,
            save_iterations_dir=save_iterations,
            json_output_path=json_output,
            verbose=verbose,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    rect = result.final_rect
    click.echo(
        f"wrote {result.output_path}; rect={rect.x0:.4f},{rect.y0:.4f},{rect.x1:.4f},{rect.y1:.4f}; "
        f"iterations={len(result.iterations)}"
    )


@main.command("llm-crop-batch")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("jobs_tsv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="Directory for output JPEG crops.")
@click.option("--prompt-template", help="Optional Python format template using {page}, {figure}, {asset}, and {description}.")
@click.option("--region", default="0,0,1,1", show_default=True, help="Starting normalized full-page rectangle x0,y0,x1,y1.")
@click.option("--model", default=lambda: env_str("MODEL", "deepseek/deepseek-v4-pro"), show_default="env MODEL or deepseek/deepseek-v4-pro", help="OpenAI-compatible vision model name.")
@click.option("--zoom", default=lambda: env_float("ZOOM", 2.0), show_default="env ZOOM or 2.0", type=float, help="PDF render zoom factor.")
@click.option("--jpeg-quality", default=lambda: env_int("JPEG_QUALITY", 85), show_default="env JPEG_QUALITY or 85", type=click.IntRange(1, 100), help="JPEG quality for rendered crop images.")
@click.option("--max-iterations", default=6, show_default=True, type=click.IntRange(1), help="Maximum LLM refinement iterations per job.")
@click.option("--min-change-ratio", default=0.01, show_default=True, type=float, help="Stop when normalized rectangle changes less than this amount.")
@click.option("--max-concurrency", default=lambda: env_int("MAX_CONCURRENCY", 3), show_default="env MAX_CONCURRENCY or 3", type=click.IntRange(1), help="Maximum number of crop jobs to run concurrently.")
@click.option("--save-iterations", type=click.Path(file_okay=False, path_type=Path), help="Directory for per-job intermediate candidate crop images.")
@click.option("--no-json-output", is_flag=True, help="Do not write per-crop JSON metadata files.")
@click.option("--skip-existing", is_flag=True, help="Skip jobs whose output image and JSON already exist.")
@click.option("--verbose", "verbose", "-v", is_flag=True, help="Print per-iteration rectangle and rationale.")
def llm_crop_batch_command(
    pdf_path: Path,
    jobs_tsv: Path,
    output_dir: Path,
    prompt_template: str | None,
    region: str,
    model: str,
    zoom: float,
    jpeg_quality: int,
    max_iterations: int,
    min_change_ratio: float,
    max_concurrency: int,
    save_iterations: Path | None,
    no_json_output: bool,
    skip_existing: bool,
    verbose: bool,
) -> None:
    """Run multiple LLM crop jobs concurrently from a TSV file."""
    try:
        start_rect = parse_normalized_rect(region)
        jobs = read_llm_crop_jobs_tsv(
            jobs_tsv,
            output_dir,
            prompt_template=prompt_template,
            json_output=not no_json_output,
            default_region=start_rect,
            save_iterations_dir=save_iterations,
        )
        results = run_llm_crop_jobs(
            pdf_path=pdf_path,
            jobs=jobs,
            model=model,
            zoom=zoom,
            jpeg_quality=jpeg_quality,
            max_iterations=max_iterations,
            min_change_ratio=min_change_ratio,
            max_concurrency=max_concurrency,
            skip_existing=skip_existing,
            verbose=verbose,
            progress=click.echo,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    failures = [item for item in results if item.error]
    completed = [item for item in results if item.result]
    skipped = len(results) - len(completed) - len(failures)
    click.echo(f"llm-crop-batch complete: completed={len(completed)} skipped={skipped} failed={len(failures)}")
    if failures:
        raise click.ClickException("crop failures: " + "; ".join(f"{item.job.output_path.name}: {item.error}" for item in failures))


if __name__ == "__main__":
    main()
