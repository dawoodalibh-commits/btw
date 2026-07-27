"""Phase 7b — Full-Question Image Export.

Renders one cropped image per page a question appears on, covering its
*entire* content -- stem text, options, diagrams, tables, everything --
unlike Phases 5-7, which each export one isolated formula/image/table
region. Useful for showing "the question exactly as it appeared on the
page" without reassembling separate text/image/table pieces client-side.

The crop uses the full page width (options are often laid out in a grid,
and diagrams can extend past any single span's bbox) and a Y-range that's
the union of every span, image, table, and formula region belonging to the
question on that page, padded by a small margin. A question that spans a
page break gets one crop per page.

Usage:
    python question_image_exporter.py 9709_s24_qp_12.pdf --merged output/merged \
        --questions output/questions --output-dir output/question_images
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pymupdf

from schemas import read_json, write_json

_CROP_DPI = 200
_MARGIN_PT = 6.0


def _page_dims(merged_dir: Path) -> dict[int, tuple[float, float]]:
    pages = read_json(merged_dir / "merged.json")
    return {p["page"]: (p["width"], p["height"]) for p in pages}


def _question_y_ranges(q: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Per page, the (min_y0, max_y1) spanning every span/image/table/formula
    belonging to this question on that page."""
    ranges: dict[int, tuple[float, float]] = {}

    def _extend(page: int, y0: float, y1: float) -> None:
        lo, hi = ranges.get(page, (y0, y1))
        ranges[page] = (min(lo, y0), max(hi, y1))

    for s in q.get("spans", []):
        _extend(s["page"], s["bbox"]["y0"], s["bbox"]["y1"])
    for key in ("images", "tables", "formulas"):
        for ref in q.get(key, []):
            _extend(ref["page"], ref["bbox"]["y0"], ref["bbox"]["y1"])

    return ranges


def export_question_images(
    pdf_path: Path, merged_dir: Path, questions_dir: Path, output_dir: Path, dpi: int = _CROP_DPI
) -> list[dict[str, Any]]:
    questions = read_json(questions_dir / "questions.json")
    page_dims = _page_dims(merged_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    results: list[dict[str, Any]] = []
    try:
        for q in questions:
            y_ranges = _question_y_ranges(q)
            images: list[dict[str, Any]] = []
            for page_number, (y0, y1) in sorted(y_ranges.items()):
                dims = page_dims.get(page_number)
                if dims is None:
                    continue
                width, height = dims
                y0 = max(0.0, y0 - _MARGIN_PT)
                y1 = min(height, y1 + _MARGIN_PT)

                page = doc[page_number - 1]
                pix = page.get_pixmap(clip=pymupdf.Rect(0, y0, width, y1), dpi=dpi)
                filename = f"q{q['question']}_p{page_number}.png"
                pix.save(output_dir / filename)

                images.append(
                    {"page": page_number, "file": filename, "bbox": {"x0": 0.0, "y0": y0, "x1": width, "y1": y1}}
                )
            results.append({"question": q["question"], "images": images})
    finally:
        doc.close()

    write_json(results, output_dir / "question_images.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7b: export a full-page crop of each question.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--merged", type=Path, default=Path("output/merged"), help="Phase 3 output directory (page dimensions)")
    parser.add_argument("--questions", type=Path, default=Path("output/questions"), help="Phase 4 output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/question_images"))
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    args = parser.parse_args()

    results = export_question_images(args.pdf, args.merged, args.questions, args.output_dir, dpi=args.dpi)
    n_images = sum(len(r["images"]) for r in results)
    print(f"Exported {n_images} question images across {len(results)} questions -> {args.output_dir}")


if __name__ == "__main__":
    main()
