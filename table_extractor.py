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

All of a paper's table crops go through PaddleOCR in one call, and the text
lines inside each crop are recognized in batches rather than one at a time
(`--rec-batch-size`) -- recognition is the per-line stage, so that setting is
what decides whether the GPU is actually busy during this phase. Crops for
the next paper are rasterized on a background thread meanwhile.

Usage:
    python table_extractor.py 9709_s24_qp_12.pdf --merged output/merged --output-dir output/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

import pymupdf

from accel import DEVICES, prefetch, resolve_paddle_device
from schemas import BBox, read_json, report_paper_failure, resolve_batch_jobs, write_json

_CROP_DPI = 300
_ROW_TOLERANCE_FRAC = 0.6  # fraction of median token height that still counts as "same row"

# Text lines recognized per forward pass. A table crop holds a few dozen
# short lines, and recognizing them one at a time is what leaves the GPU idle
# through this phase.
DEFAULT_REC_BATCH_SIZE = 16


def _ocr_tokens(ocr, image_paths: list[Path]) -> list[list[tuple[str, float, float, float]]]:
    """Returns (text, x_center, y_center, height) per token, per input crop."""
    if not image_paths:
        return []
    tokens_per_image: list[list[tuple[str, float, float, float]]] = []
    for prediction in ocr.predict([str(p) for p in image_paths]):
        result = prediction.json["res"]
        tokens = []
        for text, box in zip(result["rec_texts"], result["rec_boxes"]):
            x0, y0, x1, y1 = box
            tokens.append((text, (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0))
        tokens_per_image.append(tokens)
    return tokens_per_image


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


def build_ocr(device: str = "auto", rec_batch_size: int = DEFAULT_REC_BATCH_SIZE):
    """Loads PaddleOCR (detection + recognition only) on `device`."""
    from paddleocr import PaddleOCR

    resolved = resolve_paddle_device(device)
    ocr = PaddleOCR(
        device=resolved,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=rec_batch_size,
    )
    print(f"[tables] device={resolved} rec_batch_size={rec_batch_size}")
    return ocr


def render_crops(pdf_path: Path, merged_dir: Path, output_dir: Path, dpi: int = _CROP_DPI) -> tuple[list[dict[str, Any]], list[Path]]:
    """Rasterize this paper's table regions.

    Returns the result records (headers/rows still unfilled) alongside the
    crop paths, so rasterization can run ahead of the OCR that consumes them.
    """
    pages_data = read_json(merged_dir / "merged.json")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    records: list[dict[str, Any]] = []
    crop_paths: list[Path] = []
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

                crop_paths.append(crops_dir / filename)
                records.append(
                    {
                        "id": block["id"],
                        "page": page_data["page"],
                        "bbox": block["bbox"],
                        "image": filename,
                        "headers": [],
                        "rows": [],
                    }
                )
    finally:
        doc.close()
    return records, crop_paths


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
        ocr = build_ocr(device)
    records, crop_paths = render_crops(pdf_path, merged_dir, output_dir, dpi)
    _fill_records(records, _ocr_tokens(ocr, crop_paths))
    write_json(records, output_dir / "tables.json")
    return records


def _fill_records(records: list[dict[str, Any]], tokens_per_crop: list[list[tuple[str, float, float, float]]]) -> None:
    for record, tokens in zip(records, tokens_per_crop):
        rows = _cluster_into_rows(tokens)
        record["headers"], record["rows"] = (rows[0], rows[1:]) if rows else ([], [])


def _render_all(jobs: list[tuple[Path, ...]], dpi: int) -> Iterator[tuple[tuple[Path, ...], Any]]:
    """Rasterize each paper's crops in turn, yielding (job, outcome).

    Runs on `prefetch`'s background thread so the next paper is rasterized
    while the current one's crops are on the GPU. `outcome` is either
    (records, crop_paths) or the exception that paper died of -- raising here
    would strand every paper behind it.
    """
    for job in jobs:
        pdf, merged_dir, out_dir = job
        try:
            yield job, render_crops(pdf, merged_dir, out_dir, dpi)
        except Exception as exc:
            yield job, exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: reconstruct table regions into headers/rows.")
    parser.add_argument("pdf", type=Path, nargs="+", help="One or more input PDFs")
    parser.add_argument("--merged", type=Path, default=None, help="Phase 3 output directory (single PDF)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (single PDF)")
    parser.add_argument("--output-root", type=Path, default=None, help="Batch mode: <root>/<stem>/{merged,tables}")
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="Accelerator for PaddleOCR. Needs the paddlepaddle-gpu build for cuda.",
    )
    parser.add_argument(
        "--rec-batch-size",
        type=int,
        default=DEFAULT_REC_BATCH_SIZE,
        help="Text lines recognized per forward pass. Higher keeps the GPU busier but uses more VRAM.",
    )
    args = parser.parse_args()
    if args.rec_batch_size < 1:
        parser.error("--rec-batch-size must be at least 1")
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

    try:
        ocr = build_ocr(args.device, args.rec_batch_size)
    except (RuntimeError, ValueError) as exc:  # unusable device: nothing to salvage
        sys.exit(str(exc))

    failed = 0
    for job, outcome in prefetch(lambda: _render_all(jobs, args.dpi)):
        pdf, _merged_dir, out_dir = job
        try:
            if isinstance(outcome, Exception):
                raise outcome
            records, crop_paths = outcome
            _fill_records(records, _ocr_tokens(ocr, crop_paths))
        except Exception as exc:  # one bad PDF shouldn't abandon the rest of the batch
            failed += 1
            report_paper_failure("tables", pdf, exc)
            continue
        write_json(records, out_dir / "tables.json")
        print(f"Extracted {len(records)} tables -> {out_dir}")

    if failed == len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
