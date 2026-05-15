def markdown_translation_prompt(markdown: str, output_language: str) -> str:
    return f"""Translate this Markdown page to {output_language}.

Rules:
- Preserve Markdown structure, headings, lists, tables, footnotes, blockquotes, code blocks, mathematical notation, image links, HTML comments, URLs, file paths, citation labels, and asset paths.
- Translate prose, headings, captions, footnotes, and explanatory text.
- Keep code identifiers, formulas, URLs, file paths, citation keys, and image paths unchanged.
- Do not add commentary about the task.
- Output only the translated Markdown.

Markdown:
{markdown}
"""


VISION_SMOKE_PROMPT = """Read this PDF page image and return only this JSON shape:
{"vision_supported": true, "visible_title": "<short visible title or first heading>"}
If you cannot read the image, return {"vision_supported": false, "visible_title": ""}.
"""

PAGE_TO_MARKDOWN_PROMPT = """Convert this visible PDF page to Markdown.

Rules:
- Preserve the original reading order.
- Preserve headings, paragraphs, emphasis, lists, numbered lists, tables, code blocks, captions, footnotes, and mathematical notation as Markdown where possible.
- For text inside screenshots or scanned regions, perform OCR and include it as Markdown.
- For non-text diagrams, charts, photos, logos, or other graphical content that should remain an image, insert a concise Markdown note such as: [Figure: short description].
- Do not invent content that is not visible.
- Do not include commentary about the task.
- Output only Markdown.
"""

IMAGE_CLASSIFY_PROMPT = """Classify this cropped PDF image region.

Return exactly one word:
TEXT - if the image primarily contains readable text, code, table text, formulas, or scanned prose that should be OCR'd.
GRAPHIC - if the image is primarily a diagram, chart, photo, logo, illustration, or visual figure that should be preserved as an image.
"""

IMAGE_OCR_PROMPT = """OCR this cropped PDF image region into Markdown.

Preserve line breaks, lists, tables, code formatting, formulas, and emphasis when visible.
Do not add commentary. Output only Markdown.
"""


def with_output_language(prompt: str, output_language: str | None) -> str:
    if not output_language:
        return prompt
    return (
        f"{prompt.rstrip()}\n\n"
        f"Output language: translate all visible prose to {output_language}. "
        f"The final Markdown must be written in {output_language} except for the explicitly preserved items below. "
        "This translation requirement applies to headings, body paragraphs, footnotes, captions, notes, quoted sentences, parenthetical explanations, and prose inside bibliographic notes. "
        "Do not leave source-language sentences untranslated merely because they are quoted, footnoted, cited, or split across pages. "
        "Keep Markdown syntax, code identifiers, URLs, file paths, citation labels, proper names, bibliographic titles, music symbols, and asset placeholders unchanged. "
        "Preserve the original document structure and formatting."
    )
