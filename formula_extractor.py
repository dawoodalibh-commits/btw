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
import sys
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from schemas import BBox, read_json, resolve_batch_jobs, write_json

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


def _build_model(device: str):
    """Loads Pix2Tex on `device` ("auto" | "cpu" | "cuda").

    pix2tex's own default arguments set no_cuda=True, so a bare LatexOCR()
    pins itself to CPU even on a CUDA box -- the device has to be requested
    explicitly. It only knows cuda/cpu, so mps falls back to cpu.
    """
    from munch import Munch
    from pix2tex.cli import LatexOCR

    import torch

    use_cuda = torch.cuda.is_available() if device in ("auto", "cuda") else False
    print(f"[formulas] device={'cuda' if use_cuda else 'cpu'}")
    # Mirrors pix2tex's own defaults (paths are relative to the package dir,
    # which its @in_model_path decorator cd's into) with no_cuda made explicit.
    arguments = Munch(
        {
            "config": "settings/config.yaml",
            "checkpoint": "checkpoints/weights.pth",
            "no_cuda": not use_cuda,
            "no_resize": False,
        }
    )
    return LatexOCR(arguments)


def extract_formulas(
    pdf_path: Path,
    merged_dir: Path,
    output_dir: Path,
    dpi: int = _CROP_DPI,
    device: str = "auto",
    model=None,
) -> list[dict[str, Any]]:
    # An already-loaded model can be passed in so a batch of PDFs pays the
    # ~100MB checkpoint load once instead of once per PDF.
    if model is None:
        model = _build_model(device)
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
    parser.add_argument("pdf", type=Path, nargs="+", help="One or more input PDFs")
    parser.add_argument("--merged", type=Path, default=None, help="Phase 3 output directory (single PDF)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (single PDF)")
    parser.add_argument("--output-root", type=Path, default=None, help="Batch mode: <root>/<stem>/{merged,formulas}")
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Accelerator for Pix2Tex. 'auto' uses CUDA when available (mps is unsupported -> cpu).",
    )
    args = parser.parse_args()
    try:
        jobs = resolve_batch_jobs(
            args.pdf,
            args.output_root,
            ["merged", "formulas"],
            [args.merged, args.output_dir],
            ["output/merged", "output/formulas"],
        )
    except ValueError as exc:
        parser.error(str(exc))

    model = _build_model(args.device)

    failed = 0
    for pdf, merged_dir, out_dir in jobs:
        try:
            results = extract_formulas(pdf, merged_dir, out_dir, dpi=args.dpi, model=model)
        except Exception as exc:  # one bad PDF shouldn't abandon the rest of the batch
            failed += 1
            print(f"!!! FAILED formulas for {pdf}: {exc}", file=sys.stderr)
            continue
        n_ok = sum(1 for r in results if r["latex"])
        print(f"Extracted {len(results)} formula regions ({n_ok} OCR'd successfully) -> {out_dir}")

    if failed == len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
