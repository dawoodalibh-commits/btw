"""Phase 6 — Image Processing.

Images stay as images: no captioning, no vision model calls here (that only
happens later, on demand, when a student actually asks about a specific
question -- captioning every diagram up front would burn a lot of compute
for images nobody ever looks at).

For each region Phase 2 labeled "image": if Phase 1 already found an
embedded raster image inside it, reuse that file directly (it's higher
quality than a re-render). Otherwise the diagram was drawn with vector
graphics (common in exam papers) and PyMuPDF's raw image extraction can't
see it, so crop it straight from the PDF page instead.

Usage:
    python image_exporter.py 9709_s24_qp_12.pdf --merged output/merged --output-dir output/images
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pymupdf

from schemas import BBox, read_json, write_json

_CROP_DPI = 200


def export_images(pdf_path: Path, merged_dir: Path, extracted_dir: Path, output_dir: Path, dpi: int = _CROP_DPI) -> list[dict[str, Any]]:
    pages_data = read_json(merged_dir / "merged.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    results: list[dict[str, Any]] = []
    try:
        for page_data in pages_data:
            page = doc[page_data["page"] - 1]
            for block in page_data["blocks"]:
                if block["type"] != "image":
                    continue

                if block["images"]:
                    # Reuse the embedded raster Phase 1 already saved -- higher
                    # quality than re-rendering the region from the page.
                    embedded = block["images"][0]
                    src = extracted_dir / "images" / embedded["file"]
                    filename = f"{block['id']}{Path(embedded['file']).suffix}"
                    shutil.copy(src, output_dir / filename)
                else:
                    bbox = BBox.from_dict(block["bbox"])
                    pix = page.get_pixmap(clip=pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1), dpi=dpi)
                    filename = f"{block['id']}.png"
                    pix.save(output_dir / filename)

                results.append({"id": block["id"], "page": page_data["page"], "bbox": block["bbox"], "file": filename})
    finally:
        doc.close()

    write_json(results, output_dir / "images.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: export image regions as standalone files.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--merged", type=Path, default=Path("output/merged"), help="Phase 3 output directory")
    parser.add_argument("--extracted", type=Path, default=Path("output/extracted"), help="Phase 1 output directory (for embedded raster images)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/images"))
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    args = parser.parse_args()

    results = export_images(args.pdf, args.merged, args.extracted, args.output_dir, dpi=args.dpi)
    print(f"Exported {len(results)} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
