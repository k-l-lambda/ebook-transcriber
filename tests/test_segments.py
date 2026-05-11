import tempfile
import unittest
from pathlib import Path

from ebook_transcriber.segments import read_segments, safe_segment_filename


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


if __name__ == "__main__":
    unittest.main()
