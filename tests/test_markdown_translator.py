from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ebook_transcriber.markdown_translator import (
    TranslateMarkdownOptions,
    parse_page_numbers,
    split_markdown_pages,
    translate_markdown_file,
    translate_markdown_tree,
)


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def text_chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        marker = f"translated {len(self.prompts)}"
        return marker


class MarkdownTranslatorTests(unittest.TestCase):
    def test_split_markdown_pages(self) -> None:
        pages = split_markdown_pages("intro\n<!-- page 3 -->\nA\n<!-- page 4 -->\nB\n")
        self.assertEqual([page.page_number for page in pages], [3, 4])
        self.assertEqual(pages[0].body, "A")
        self.assertEqual(pages[1].body, "B")

    def test_split_markerless_markdown_as_page_one(self) -> None:
        pages = split_markdown_pages("# Title\n\nBody")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_number, 1)
        self.assertEqual(pages[0].body, "# Title\n\nBody")

    def test_parse_page_numbers(self) -> None:
        selected = parse_page_numbers("1,3,5-6", [1, 2, 3, 4, 5, 6])
        self.assertEqual(selected, {1, 3, 5, 6})

    def test_translate_file_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            output = root / "out.md"
            source.write_text("<!-- page 1 -->\nA\n<!-- page 2 -->\nB\n", encoding="utf-8")
            client = FakeClient()

            translate_markdown_file(
                TranslateMarkdownOptions(source, output, "model", "zh"),
                client=client,
            )

            out = output.read_text(encoding="utf-8")
            self.assertIn("<!-- page 1 -->", out)
            self.assertIn("translated 1", out)
            self.assertIn("<!-- page 2 -->", out)
            self.assertIn("translated 2", out)
            checkpoint = json.loads(output.with_suffix(".md.checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["completed_pages"], [1, 2])

    def test_translate_file_skips_completed_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            output = root / "out.md"
            source.write_text("<!-- page 1 -->\nA\n<!-- page 2 -->\nB\n", encoding="utf-8")
            output.write_text("<!-- page 1 -->\n\nold\n", encoding="utf-8")
            output.with_suffix(".md.checkpoint.json").write_text('{"completed_pages": [1]}\n', encoding="utf-8")
            client = FakeClient()

            translate_markdown_file(
                TranslateMarkdownOptions(source, output, "model", "zh"),
                client=client,
            )

            self.assertEqual(len(client.prompts), 1)
            out = output.read_text(encoding="utf-8")
            self.assertIn("old", out)
            self.assertIn("<!-- page 2 -->", out)

    def test_restart_clears_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            output = root / "out.md"
            source.write_text("<!-- page 1 -->\nA\n", encoding="utf-8")
            output.write_text("old", encoding="utf-8")
            output.with_suffix(".md.checkpoint.json").write_text('{"completed_pages": [1]}\n', encoding="utf-8")

            translate_markdown_file(
                TranslateMarkdownOptions(source, output, "model", "zh", restart=True),
                client=FakeClient(),
            )

            out = output.read_text(encoding="utf-8")
            self.assertNotIn("old", out)
            self.assertIn("translated 1", out)

    def test_translate_tree_preserves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            (source_dir / "parts").mkdir(parents=True)
            (source_dir / "chapter_01.md").write_text("A", encoding="utf-8")
            (source_dir / "parts" / "chapter_02.md").write_text("B", encoding="utf-8")

            outputs = translate_markdown_tree(
                TranslateMarkdownOptions(source_dir, output_dir, "model", "zh", dry_run=True)
            )

            self.assertEqual(
                sorted(path.relative_to(output_dir).as_posix() for path in outputs),
                ["chapter_01.md", "parts/chapter_02.md"],
            )

    def test_translate_tree_supports_file_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src"
            output_dir = root / "out"
            source_dir.mkdir()
            (source_dir / "a.md").write_text("A", encoding="utf-8")
            (source_dir / "b.md").write_text("B", encoding="utf-8")

            outputs = translate_markdown_tree(
                TranslateMarkdownOptions(source_dir, output_dir, "model", "zh", max_concurrency=2),
                client_factory=FakeClient,
            )

            self.assertEqual(sorted(path.name for path in outputs), ["a.md", "b.md"])
            self.assertTrue((output_dir / "a.md.checkpoint.json").exists())
            self.assertTrue((output_dir / "b.md.checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
