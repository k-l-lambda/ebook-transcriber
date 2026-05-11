from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from .config import env_float, env_int, env_str, load_project_env
from .figure_crops import crop_figures
from .markdown_writer import default_output_path
from .pipeline import ConvertOptions, convert_pdf
from .segments import read_segments, safe_segment_filename


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


if __name__ == "__main__":
    main()
