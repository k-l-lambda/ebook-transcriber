import tempfile
import unittest
from pathlib import Path

from ebook_transcriber.segments import read_segments, safe_segment_filename, write_index_markdown


class SegmentTests(unittest.TestCase):
    def test_read_segments(self):
        text = '''source_pdf: "book.pdf"
segments:
  - id: cover
    title: "Cover / title page"
    pdf_pages: [1, 1]
    type: front_matter

  - id: chapter_01
    title: "Chapter One"
    pdf_pages: [10, 12]
    type: chapter
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "segments.yaml"
            path.write_text(text, encoding="utf-8")
            segments = read_segments(path)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].id, "cover")
        self.assertEqual(segments[0].pdf_pages, (1, 1))
        self.assertEqual(segments[1].title, "Chapter One")
        self.assertEqual(segments[1].pdf_pages, (10, 12))

    def test_safe_segment_filename(self):
        self.assertEqual(safe_segment_filename("chapter 01/intro"), "chapter_01_intro")

    def test_write_index_markdown_uses_links(self):
        text = '''source_pdf: "book.pdf"
segments:
  - id: cover
    title: "Cover / title page"
    pdf_pages: [1, 1]

  - id: chapter_01
    title: "Chapter One"
    pdf_pages: [10, 12]
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "segments.yaml"
            path.write_text(text, encoding="utf-8")
            (root / "cover.md").write_text("cover content", encoding="utf-8")
            (root / "chapter_01.md").write_text("chapter content", encoding="utf-8")
            index_path = write_index_markdown(root, read_segments(path))
            self.assertEqual(
                index_path.read_text(encoding="utf-8"),
                "- [Cover / title page](cover.md)\n- [Chapter One](chapter_01.md)\n",
            )


if __name__ == "__main__":
    unittest.main()
