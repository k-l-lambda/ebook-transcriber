import unittest

from ebook_transcriber.pdf_reader import parse_page_range


class PageRangeTests(unittest.TestCase):
    def test_parse_all_pages(self):
        self.assertEqual(parse_page_range(None, 3), [0, 1, 2])

    def test_parse_single_and_ranges(self):
        self.assertEqual(parse_page_range("1,3-4", 5), [0, 2, 3])

    def test_parse_rejects_descending_range(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_page_range("4-2", 5)

    def test_parse_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            parse_page_range("6", 5)


if __name__ == "__main__":
    unittest.main()
