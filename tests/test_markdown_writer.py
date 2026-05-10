import unittest
from pathlib import Path

from ebook_transcriber.markdown_writer import (
    asset_dir_for,
    default_output_path,
    relative_asset_path,
    replace_first_asset_placeholder,
)


class MarkdownWriterTests(unittest.TestCase):
    def test_default_output_path(self):
        self.assertEqual(default_output_path("/tmp/My Book.pdf"), Path("output/My Book.md"))

    def test_default_output_path_with_configured_dir(self):
        self.assertEqual(default_output_path("/tmp/My Book.pdf", "dist"), Path("dist/My Book.md"))

    def test_asset_dir_for(self):
        self.assertEqual(asset_dir_for("output/book.md", "assets"), Path("output/assets"))

    def test_relative_asset_path(self):
        self.assertEqual(
            relative_asset_path("output/book.md", "output/assets/page001_img00.jpg"),
            "assets/page001_img00.jpg",
        )

    def test_replace_placeholder(self):
        md = 'before\n\n[[ASSET:page=1,index=0,alt="diagram"]]\n\nafter'
        self.assertIn("![diagram](assets/a.jpg)", replace_first_asset_placeholder(md, "assets/a.jpg"))


if __name__ == "__main__":
    unittest.main()
