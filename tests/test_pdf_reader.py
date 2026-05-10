import unittest
from pathlib import Path

from ebook_transcriber.pdf_reader import inspect_page, open_document


SAMPLE = Path("/home/camus/work/diary-job/temp/Language Oriented Programming.pdf")


class PdfReaderTests(unittest.TestCase):
    def test_inspect_sample_pdf_first_page_if_available(self):
        if not SAMPLE.exists():
            self.skipTest(f"sample PDF not found: {SAMPLE}")
        doc = open_document(SAMPLE)
        try:
            inventory = inspect_page(doc[0], 0)
        finally:
            doc.close()
        self.assertEqual(inventory.page_number, 1)
        self.assertGreaterEqual(inventory.text_blocks, 0)
        self.assertGreaterEqual(inventory.vector_drawings, 0)


if __name__ == "__main__":
    unittest.main()
