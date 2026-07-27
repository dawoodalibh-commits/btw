"""Phase 5 — Formula Extraction.

Only processes regions Phase 2 labeled "formula". Crops each one straight
from the source PDF at high resolution (formula boxes are tiny -- a few
dozen points -- so a plain page render would be too low-res for OCR) and
feeds the crop to Pix2Tex. This is the only stage in the pipeline that reads
a specific region *type*; swap Pix2Tex for a different equation-OCR model by
changing only `_run_ocr` below.

Usage:
    python formula_extractor.py 9709_s24_qp_12.pdf --merged output/merged --output-dir output/formulas
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from schemas import BBox, read_json, write_json

_CROP_DPI = 400
_PADDING_PT = 2.0  # a hair of margin around the detected box helps OCR context
_MIN_DIM_PX = 64  # upscale crops smaller than this so the model has enough signal


def _crop_region(page: pymupdf.Page, bbox: BBox, dpi: int) -> pymupdf.Pixmap:
    rect = pymupdf.Rect(bbox.x0 - _PADDING_PT, bbox.y0 - _PADDING_PT, bbox.x1 + _PADDING_PT, bbox.y1 + _PADDING_PT)
    rect &= page.rect
    return page.get_pixmap(clip=rect, dpi=dpi)


def _run_ocr(model, image: Image.Image) -> str | None:
    if min(image.size) < _MIN_DIM_PX:
        scale = _MIN_DIM_PX / min(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
    try:
        return model(image)
    except Exception:
        return None


def extract_formulas(pdf_path: Path, merged_dir: Path, output_dir: Path, dpi: int = _CROP_DPI) -> list[dict[str, Any]]:
    from pix2tex.cli import LatexOCR

    model = LatexOCR()
    pages_data = read_json(merged_dir / "merged.json")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    results: list[dict[str, Any]] = []
    try:
        for page_data in pages_data:
            page = doc[page_data["page"] - 1]
            for block in page_data["blocks"]:
                if block["type"] != "formula":
                    continue
                bbox = BBox.from_dict(block["bbox"])
                pix = _crop_region(page, bbox, dpi)
                filename = f"{block['id']}.png"
                pix.save(crops_dir / filename)

                latex = _run_ocr(model, Image.open(crops_dir / filename))
                results.append(
                    {"id": block["id"], "page": page_data["page"], "bbox": block["bbox"], "image": filename, "latex": latex}
                )
    finally:
        doc.close()

    write_json(results, output_dir / "formulas.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: OCR formula regions to LaTeX via Pix2Tex.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--merged", type=Path, default=Path("output/merged"), help="Phase 3 output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/formulas"))
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    args = parser.parse_args()

    results = extract_formulas(args.pdf, args.merged, args.output_dir, dpi=args.dpi)
    n_ok = sum(1 for r in results if r["latex"])
    print(f"Extracted {len(results)} formula regions ({n_ok} OCR'd successfully) -> {args.output_dir}")


if __name__ == "__main__":
    main()
