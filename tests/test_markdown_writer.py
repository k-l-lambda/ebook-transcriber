import unittest
from pathlib import Path

from ebook_transcriber.markdown_writer import (
    append_page_markdown,
    asset_dir_for,
    checkpoint_path_for,
    default_output_path,
    prepare_progressive_output,
    read_completed_pages,
    relative_asset_path,
    replace_first_asset_placeholder,
    write_completed_pages,
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

    def test_checkpoint_path_for(self):
        self.assertEqual(checkpoint_path_for("output/book.md"), Path("output/book.md.checkpoint.json"))

    def test_completed_pages_roundtrip(self):
        root = Path("/tmp/ebook_transcriber_test_checkpoint")
        checkpoint = root / "book.md.checkpoint.json"
        write_completed_pages(checkpoint, {3, 1})
        self.assertEqual(read_completed_pages(checkpoint), {1, 3})

    def test_append_page_markdown(self):
        output = Path("/tmp/ebook_transcriber_test_append/book.md")
        checkpoint = checkpoint_path_for(output)
        prepare_progressive_output(output, checkpoint, restart=True)
        append_page_markdown(output, 1, "# One")
        append_page_markdown(output, 2, "# Two")
        self.assertEqual(output.read_text(encoding="utf-8"), "<!-- page 1 -->\n\n# One\n\n\n<!-- page 2 -->\n\n# Two\n")


if __name__ == "__main__":
    unittest.main()
