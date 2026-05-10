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
