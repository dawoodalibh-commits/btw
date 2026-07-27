"""Phase 7 — Table Extraction.

Crops regions Phase 2 labeled "table" and reconstructs them into headers +
rows. Rather than relying on PPStructureV3's table-structure sub-model
(which expects to do its own layout detection first, and second-guesses a
region we already know is a table), this reads plain OCR tokens with their
bounding boxes and reconstructs the grid geometrically: cluster tokens into
rows by y-position, then order each row left-to-right by x-position. That's
enough for the simple data tables (mass/volume, before/after, etc.) that show
up in maths/physics/chemistry papers -- swap `_ocr_tokens` for a real
table-structure model later if papers with merged cells or nested headers
turn up.

Usage:
    python table_extractor.py 9709_s24_qp_12.pdf --merged output/merged --output-dir output/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pymupdf

from schemas import BBox, read_json, resolve_batch_jobs, write_json

_CROP_DPI = 300
_ROW_TOLERANCE_FRAC = 0.6  # fraction of median token height that still counts as "same row"


def _ocr_tokens(ocr, image_path: Path) -> list[tuple[str, float, float, float]]:
    """Returns (text, x_center, y_center, height) for every recognized token."""
    result = ocr.predict(str(image_path))[0].json["res"]
    tokens = []
    for text, box in zip(result["rec_texts"], result["rec_boxes"]):
        x0, y0, x1, y1 = box
        tokens.append((text, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0))
    return tokens


def _cluster_into_rows(tokens: list[tuple[str, float, float, float]]) -> list[list[str]]:
    if not tokens:
        return []
    heights = [h for *_, h in tokens]
    tol = (sorted(heights)[len(heights) // 2]) * _ROW_TOLERANCE_FRAC

    ordered = sorted(tokens, key=lambda t: t[2])  # by y_center
    rows: list[list[tuple[str, float, float, float]]] = [[ordered[0]]]
    for token in ordered[1:]:
        if abs(token[2] - rows[-1][-1][2]) <= tol:
            rows[-1].append(token)
        else:
            rows.append([token])

    return [[text for text, *_ in sorted(row, key=lambda t: t[1])] for row in rows]


def _build_ocr(device: str):
    """Loads PaddleOCR (detection + recognition only) on `device`."""
    import paddle
    from paddleocr import PaddleOCR

    # Paddle spells CUDA "gpu" and has no Metal backend, so mps can only mean
    # cpu here. "auto" is left to paddle, which uses a GPU when its GPU build
    # is the one installed.
    paddle_device = {"auto": None, "cpu": "cpu", "cuda": "gpu", "mps": "cpu"}[device]
    ocr_kwargs: dict[str, Any] = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if paddle_device is not None:
        ocr_kwargs["device"] = paddle_device
    ocr = PaddleOCR(**ocr_kwargs)
    print(f"[tables] device={paddle.device.get_device()}")
    return ocr


def extract_tables(
    pdf_path: Path,
    merged_dir: Path,
    output_dir: Path,
    dpi: int = _CROP_DPI,
    device: str = "auto",
    ocr=None,
) -> list[dict[str, Any]]:
    # An already-loaded OCR model can be passed in so a batch of PDFs pays the
    # model-load cost once instead of once per PDF.
    if ocr is None:
        ocr = _build_ocr(device)
    pages_data = read_json(merged_dir / "merged.json")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    results: list[dict[str, Any]] = []
    try:
        for page_data in pages_data:
            page = doc[page_data["page"] - 1]
            for block in page_data["blocks"]:
                if block["type"] != "table":
                    continue
                bbox = BBox.from_dict(block["bbox"])
                pix = page.get_pixmap(clip=pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1), dpi=dpi)
                filename = f"{block['id']}.png"
                pix.save(crops_dir / filename)

                rows = _cluster_into_rows(_ocr_tokens(ocr, crops_dir / filename))
                headers, data_rows = (rows[0], rows[1:]) if rows else ([], [])

                results.append(
                    {
                        "id": block["id"],
                        "page": page_data["page"],
                        "bbox": block["bbox"],
                        "image": filename,
                        "headers": headers,
                        "rows": data_rows,
                    }
                )
    finally:
        doc.close()

    write_json(results, output_dir / "tables.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: reconstruct table regions into headers/rows.")
    parser.add_argument("pdf", type=Path, nargs="+", help="One or more input PDFs")
    parser.add_argument("--merged", type=Path, default=None, help="Phase 3 output directory (single PDF)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (single PDF)")
    parser.add_argument("--output-root", type=Path, default=None, help="Batch mode: <root>/<stem>/{merged,tables}")
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Accelerator for PaddleOCR. Needs the paddlepaddle-gpu build for cuda.",
    )
    args = parser.parse_args()
    try:
        jobs = resolve_batch_jobs(
            args.pdf,
            args.output_root,
            ["merged", "tables"],
            [args.merged, args.output_dir],
            ["output/merged", "output/tables"],
        )
    except ValueError as exc:
        parser.error(str(exc))

    ocr = _build_ocr(args.device)

    failed = 0
    for pdf, merged_dir, out_dir in jobs:
        try:
            results = extract_tables(pdf, merged_dir, out_dir, dpi=args.dpi, ocr=ocr)
        except Exception as exc:  # one bad PDF shouldn't abandon the rest of the batch
            failed += 1
            print(f"!!! FAILED tables for {pdf}: {exc}", file=sys.stderr)
            continue
        print(f"Extracted {len(results)} tables -> {out_dir}")

    if failed == len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
