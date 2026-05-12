import fitz
import unittest

from ebook_transcriber.figure_crops import (
    CropResult,
    FigureAnchor,
    _assign_rects_to_figures,
    parse_figure_anchors,
    replace_figure_lines,
)


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
    def test_assign_rects_to_figures_keeps_short_candidate_list(self):
        rects = [fitz.Rect(1, 2, 3, 4), fitz.Rect(5, 6, 7, 8)]
        self.assertEqual(_assign_rects_to_figures(rects, 3), rects)

    def test_assign_rects_to_figures_prefers_taller_candidates_in_page_order(self):
        rects = [
            fitz.Rect(0, 0, 10, 10),
            fitz.Rect(20, 20, 30, 60),
            fitz.Rect(25, 70, 50, 100),
        ]
        self.assertEqual(_assign_rects_to_figures(rects, 2), rects[1:])


if __name__ == "__main__":
    unittest.main()
