import unittest

import fitz

from ebook_transcriber.llm_crop import (
    NormalizedRect,
    clamp_normalized_rect,
    extract_json_object,
    normalized_to_pdf_rect,
    parse_model_response,
    parse_normalized_rect,
    pdf_to_normalized_rect,
    rect_delta_ratio,
)


class LlmCropTests(unittest.TestCase):
    def test_parse_normalized_rect(self):
        rect = parse_normalized_rect("0.1, 0.2, 0.8, 0.9")
        self.assertEqual(rect, NormalizedRect(0.1, 0.2, 0.8, 0.9))

    def test_clamp_normalized_rect(self):
        rect = clamp_normalized_rect(NormalizedRect(-1, 0.2, 2, 0.9))
        self.assertEqual(rect, NormalizedRect(0.0, 0.2, 1.0, 0.9))

    def test_reject_degenerate_rect(self):
        with self.assertRaises(ValueError):
            parse_normalized_rect("0.5,0.1,0.5,0.8")

    def test_normalized_pdf_roundtrip(self):
        page_rect = fitz.Rect(10, 20, 210, 420)
        normalized = NormalizedRect(0.25, 0.125, 0.75, 0.625)
        pdf_rect = normalized_to_pdf_rect(normalized, page_rect)
        self.assertEqual(pdf_rect, fitz.Rect(60, 70, 160, 270))
        self.assertEqual(pdf_to_normalized_rect(pdf_rect, page_rect), normalized)

    def test_extract_json_object_from_fence(self):
        self.assertEqual(extract_json_object('```json\n{"done": true}\n```'), {"done": True})

    def test_extract_json_object_from_surrounding_text(self):
        self.assertEqual(extract_json_object('Here: {"done": false} thanks'), {"done": False})

    def test_parse_model_response(self):
        rect, done, confidence, rationale = parse_model_response(
            '{"rect":{"x0":0.1,"y0":0.2,"x1":0.8,"y1":0.9},"done":true,"confidence":0.7,"rationale":"ok"}'
        )
        self.assertEqual(rect, NormalizedRect(0.1, 0.2, 0.8, 0.9))
        self.assertTrue(done)
        self.assertEqual(confidence, 0.7)
        self.assertEqual(rationale, "ok")

    def test_rect_delta_ratio(self):
        self.assertAlmostEqual(
            rect_delta_ratio(NormalizedRect(0.1, 0.2, 0.8, 0.9), NormalizedRect(0.1, 0.25, 0.75, 0.9)),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
