import unittest

from ebook_transcriber.figure_crops import CropResult, FigureAnchor, parse_figure_anchors, replace_figure_lines


class FigureCropTests(unittest.TestCase):
    def test_parse_figure_anchors_tracks_pages(self):
        markdown = """# Intro

<!-- page 8 -->

[Figure: first]

text

[Figure: second]

<!-- page 9 -->

[Figure: third]
"""
        anchors = parse_figure_anchors(markdown)
        self.assertEqual(
            [(anchor.page_number, anchor.figure_index, anchor.description) for anchor in anchors],
            [(8, 1, "first"), (8, 2, "second"), (9, 1, "third")],
        )

    def test_parse_figure_anchors_ignores_figures_before_page_marker(self):
        markdown = """[Figure: before]

<!-- page 3 -->
[Figure: after]
"""
        anchors = parse_figure_anchors(markdown)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].description, "after")

    def test_replace_figure_lines(self):
        markdown = """<!-- page 8 -->

[Figure: first]

[Figure: second]
"""
        results = [
            CropResult(FigureAnchor(2, 8, 1, "first"), None, "assets/page008_fig01.jpg", None, False),
            CropResult(FigureAnchor(4, 8, 2, "second"), None, "assets/page008_fig02.jpg", None, False),
        ]
        self.assertEqual(
            replace_figure_lines(markdown, results),
            """<!-- page 8 -->

![first](assets/page008_fig01.jpg)

![second](assets/page008_fig02.jpg)
""",
        )


if __name__ == "__main__":
    unittest.main()
