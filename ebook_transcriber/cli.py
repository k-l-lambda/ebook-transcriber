from __future__ import annotations

from pathlib import Path

import click

from .config import env_float, env_int, env_str, load_project_env
from .markdown_writer import default_output_path
from .pipeline import ConvertOptions, convert_pdf


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


if __name__ == "__main__":
    main()
