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
```

Generated image assets are written beside the Markdown file under `assets/` by default.
