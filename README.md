# ebook-transcriber

Convert PDF pages to Markdown with an OpenAI-compatible LLM API that supports image input.

## Environment

Copy `.env.example` to `.env` for project defaults such as API credentials, model, output directory, asset directory, render zoom, and JPEG quality. Use `.env.local` for machine-specific overrides.

The code reads generic configuration keys:

- `API_KEY`
- `API_BASE_URL`
- `MODEL`
- `OUTPUT_DIR`
- `ASSETS_DIR`
- `ZOOM`
- `JPEG_QUALITY`
- `OUTPUT_LANGUAGE`
- `MAX_CONCURRENCY`

Values in `.env` may reference shell environment variables with `${NAME}` syntax, for example:

```env
API_KEY=${YOUR_PROVIDER_API_KEY}
API_BASE_URL=${YOUR_PROVIDER_OPENAI_BASE_URL}
```

Do not store API keys in this repository.

## Usage

```bash
python3 -m ebook_transcriber.cli convert input.pdf --output output/book.md --model deepseek/deepseek-v4-pro
```

Useful options:

```bash
python3 -m ebook_transcriber.cli convert input.pdf --dry-run --verbose
python3 -m ebook_transcriber.cli convert input.pdf --pages 1-3 --output output/sample.md
python3 -m ebook_transcriber.cli convert input.pdf --output-language zh --output output/book.zh.md
python3 -m ebook_transcriber.cli convert input.pdf --output output/book.md --restart
```

Generated image assets are written beside the Markdown file under `assets/` by default.

## Figure crop post-processing

For scanned PDFs where figures are baked into full-page images, the converter may emit textual figure notes such as `[Figure: Musical notation ...]` without creating `assets/`. Use `crop-figures` to post-process an existing Markdown file and crop screenshots from the source PDF using those figure notes as anchors:

```bash
python3 -m ebook_transcriber.cli crop-figures input.pdf output/chapter_08.md --mode heuristic --verbose
python3 -m ebook_transcriber.cli crop-figures input.pdf output/chapter_08.md --mode heuristic --write --verbose
```

By default the command writes crop files but leaves Markdown unchanged. Add `--write` to replace `[Figure: ...]` lines with Markdown image links. `--mode heuristic` tries to crop figure-like page bands; use `--mode page` for reliable full-page screenshots when heuristic crops are poor.

Optional OCR localization settings can improve heuristic crops when a Starry OCR DB localization model is available. Configure `OCR_LOC_STARRY_OCR_ROOT` and `OCR_LOC_WEIGHTS_PATH` in `.env`; when either is unset, `crop-figures` falls back to the built-in ink-band heuristic. Run the tool from an environment that has both `ebook-transcriber` and the Starry OCR dependencies installed. The localization heatmap excludes instrument, alternative, and tempo classes before looking for text-sparse vertical regions.

## Segment conversion

Convert a PDF according to a segments YAML file containing `segments[].id` and `segments[].pdf_pages` ranges:

```bash
python3 -m ebook_transcriber.cli convert-segments input.pdf segments.yaml \
  --output-dir output/book_segments \
  --output-language zh \
  --max-concurrency 10 \
  --verbose
```

Each segment is written to `<output-dir>/<segment-id>.md`. Use `--segment <id>` to run selected segments only; repeat it for multiple segment ids. Segments run concurrently by default with `--max-concurrency 10`; use `--max-concurrency 1` for sequential execution.

### Segments YAML format

The segments file is a YAML document with optional book-level metadata and a required `segments` list. Each segment must include an `id` and a 1-based PDF page range in `pdf_pages`:

```yaml
source_pdf: "/path/to/book.pdf"
title: "Example Book"
author: "Example Author"
total_pdf_pages: 120
page_numbering_note: >-
  Ranges use PDF page sequence numbers, counted from the cover as page 1.

segments:
  - id: cover
    title: "Cover / title page"
    pdf_pages: [1, 1]
    printed_pages: null
    type: front_matter

  - id: contents
    title: "Contents"
    pdf_pages: [2, 4]
    printed_pages: ["ix", "xi"]
    type: front_matter

  - id: chapter_01
    chapter: 1
    title: "Chapter One"
    pdf_pages: [5, 18]
    printed_pages: [1, 14]
    type: chapter
    note: "Optional notes are allowed and ignored by the converter."
```

Required segment fields:

- `id`: unique segment identifier. It becomes the Markdown filename, e.g. `chapter_01` -> `chapter_01.md`.
- `pdf_pages`: two-item inclusive range `[start, end]` using PDF sequence numbers, not necessarily printed book page numbers. Page `1` means the first page in the PDF file.

Optional segment fields are allowed and ignored by the converter, including:

- `title`
- `chapter`
- `printed_pages`
- `type`
- `subtitle`
- `note`

Top-level metadata such as `source_pdf`, `title`, `author`, `translator`, `total_pdf_pages`, and notes are also allowed and ignored by the converter.


## Checkpoints

Conversion writes each completed page to the Markdown file immediately. A checkpoint file is written beside the Markdown output using the name `<output>.checkpoint.json`.

If a run stops midway, rerun the same command and completed pages listed in the checkpoint are skipped. Use `--restart` to clear the Markdown output and checkpoint before converting again.
