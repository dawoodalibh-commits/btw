"""Phase 5 — Formula Extraction.

Only processes regions Phase 2 labeled "formula". Crops each one straight
from the source PDF at high resolution (formula boxes are tiny -- a few
dozen points -- so a plain page render would be too low-res for OCR) and
feeds the crop to Pix2Tex. This is the only stage in the pipeline that reads
a specific region *type*; swap Pix2Tex for a different equation-OCR model by
changing only `Pix2TexBatchOCR` below.

Pix2Tex is autoregressive, so this is the phase where batch size matters
most: decoding one crop at a time means every generated token is its own
kernel launch over a batch of one, and the card spends the run waiting on
launch latency rather than doing arithmetic. Crops are therefore pooled
*across papers*, grouped by tensor shape and decoded together, with the next
paper's crops rasterized on a background thread while the current batch is
on the GPU.

Usage:
    python formula_extractor.py 9709_s24_qp_12.pdf --merged output/merged --output-dir output/formulas
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any, Iterator

import pymupdf
from PIL import Image

from accel import DEVICES, chunked, describe_torch_device, prefetch, resolve_torch_device
from schemas import BBox, read_json, report_paper_failure, resolve_batch_jobs, write_json

_CROP_DPI = 400
_PADDING_PT = 2.0  # a hair of margin around the detected box helps OCR context
_MIN_DIM_PX = 64  # upscale crops smaller than this so the model has enough signal

# Crops decoded in one generate() call. Crops only batch together when their
# preprocessed tensors agree on a shape, so this is an upper bound rather
# than the batch actually used.
DEFAULT_BATCH_SIZE = 16

# How many crops to pool before decoding. Pooling across papers is what makes
# the shape buckets big enough to fill a batch -- a single paper rarely has
# enough same-shaped formulas.
#
# This has to stay well above the number of distinct shapes or the buckets
# never fill and the effective batch collapses towards 1: pix2tex pads crops
# to multiples of 32 within its 672x192 maximum, so there are ~126 shapes a
# crop can land in. At 256 that averaged two crops per bucket and measured
# ~15% GPU utilization. Memory is not the constraint it looks like -- a crop
# tensor is ~0.5 MB at the model's largest input and far less in practice.
DEFAULT_QUEUE_SIZE = 1024


def _crop_region(page: pymupdf.Page, bbox: BBox, dpi: int) -> pymupdf.Pixmap:
    rect = pymupdf.Rect(bbox.x0 - _PADDING_PT, bbox.y0 - _PADDING_PT, bbox.x1 + _PADDING_PT, bbox.y1 + _PADDING_PT)
    rect &= page.rect
    return page.get_pixmap(clip=rect, dpi=dpi)


def _upscale_tiny(image: Image.Image) -> Image.Image:
    if min(image.size) >= _MIN_DIM_PX:
        return image
    scale = _MIN_DIM_PX / min(image.size)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)


class Pix2TexBatchOCR:
    """Pix2Tex with its batch dimension actually used.

    pix2tex's own LatexOCR.__call__ handles exactly one image, and its
    default arguments also set no_cuda=True, so a bare LatexOCR() pins itself
    to the CPU even on a CUDA box. Both are fixed here: the device is
    requested explicitly, and `decode` runs the model's generate() -- which
    has always accepted a batch -- over groups of crops at once.
    """

    def __init__(self, device: str = "auto", precision: str = "auto", resize: bool = True) -> None:
        from munch import Munch
        from pix2tex.cli import LatexOCR

        requested = resolve_torch_device(device)
        # pix2tex only knows cuda and cpu, so mps falls back to cpu.
        use_cuda = requested == "cuda"
        # Mirrors pix2tex's own defaults (paths are relative to the package
        # dir, which its @in_model_path decorator cd's into) with no_cuda
        # made explicit.
        #
        # resize=False drops pix2tex's iterative image-resizer, the one part of
        # this phase batching cannot help: it re-renders each crop at whatever
        # width a small ResNet asks for next, up to ten times, one crop at a
        # time -- up to ten batch-of-one launches per crop however large the
        # decode batch is.
        #
        # Measure before reaching for this. The resizer also converges crops
        # onto common widths, so it is what makes the shape buckets in
        # decode() fill: on one 9702 paper, turning it off took the crops from
        # 15 distinct shapes (avg 3.9 per batch) to 35 (avg 1.7). It removes
        # per-crop launches and fragments the decode batches at the same time,
        # and which wins depends on the papers.
        self._ocr = LatexOCR(
            Munch(
                {
                    "config": "settings/config.yaml",
                    "checkpoint": "checkpoints/weights.pth",
                    "no_cuda": not use_cuda,
                    "no_resize": not resize,
                }
            )
        )
        # What pix2tex itself settled on, not what we asked for.
        self.device = self._ocr.args.device
        # fp16 on the encoder and decoder roughly halves the memory traffic
        # this model is bound by. CUDA only -- CPU fp16 is emulated.
        self.half = self.device == "cuda" if precision == "auto" else precision == "fp16"

    def _autocast(self):
        import torch

        if self.half and self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def preprocess(self, image: Image.Image):
        """Turn one crop into the (1, H, W) tensor generate() expects.

        Lifted from LatexOCR.__call__ so that the model call can be batched
        separately from the per-image preparation in front of it. The resizer
        loop is inherently per-image -- it iteratively re-renders the crop at
        whatever width the resizer asks for next -- so it stays here.
        """
        import numpy as np
        import torch
        from pix2tex.cli import minmax_size
        from pix2tex.dataset.transforms import test_transform
        from pix2tex.utils import pad

        args = self._ocr.args
        img = minmax_size(pad(image), args.max_dimensions, args.min_dimensions)
        resizer = self._ocr.image_resizer
        if resizer is not None and not args.no_resize:
            with torch.no_grad():
                source = img.convert("RGB").copy()
                ratio, width, height = 1, source.size[0], source.size[1]
                for _ in range(10):
                    height = int(height * ratio)
                    resample = Image.Resampling.BILINEAR if ratio > 1 else Image.Resampling.LANCZOS
                    img = pad(
                        minmax_size(
                            source.resize((width, height), resample), args.max_dimensions, args.min_dimensions
                        )
                    )
                    tensor = test_transform(image=np.array(img.convert("RGB")))["image"][:1].unsqueeze(0)
                    width = (resizer(tensor.to(args.device)).argmax(-1).item() + 1) * 32
                    if width == img.size[0]:
                        break
                    ratio = width / img.size[0]
        else:
            tensor = test_transform(image=np.array(pad(img).convert("RGB")))["image"][:1].unsqueeze(0)
        return tensor[0]

    def _row_to_latex(self, row) -> str:
        """Decode one generated token row, cut at its own EOS.

        A batched generate() only stops once *every* row in the batch has
        emitted EOS, so a short formula that finished early keeps sampling
        noise while a long one catches up. token2str drops the EOS token
        itself but keeps everything printed after it, so the row has to be
        truncated here or short formulas come back with garbage glued on.
        """
        from pix2tex.utils import post_process, token2str

        eos = (row == self._ocr.args.eos_token).nonzero()
        if len(eos):
            row = row[: int(eos[0])]
        return post_process(token2str(row[None, :], self._ocr.tokenizer)[0])

    def decode(self, tensors: list, batch_size: int = DEFAULT_BATCH_SIZE) -> list[str | None]:
        """Run every preprocessed crop through the model, returning LaTeX per crop.

        Crops are bucketed by tensor shape because the encoder takes a single
        rectangular batch; within a bucket they're decoded `batch_size` at a
        time. A crop that fails is reported as None rather than taking the
        rest of the batch down with it.
        """
        import torch

        results: list[str | None] = [None] * len(tensors)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for index, tensor in enumerate(tensors):
            buckets.setdefault(tuple(tensor.shape), []).append(index)

        batches = 0
        for indices in buckets.values():
            for chunk in chunked(indices, batch_size):
                batches += 1
                try:
                    self._decode_chunk(tensors, chunk, results)
                except Exception:
                    # Most likely the batch was too big for VRAM. Retry the
                    # same crops one at a time so a single oversized batch
                    # costs throughput rather than results.
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    for index in chunk:
                        try:
                            self._decode_chunk(tensors, [index], results)
                        except Exception:
                            results[index] = None
        # The number that decides this phase's GPU utilization. Crops only
        # batch with others of the same shape, so the average here lands well
        # under --batch-size whenever --queue-size is too small to fill the
        # buckets, and that gap is invisible from the outside otherwise.
        if tensors:
            print(
                f"[formulas] decoded {len(tensors)} crops in {batches} batches "
                f"(avg {len(tensors) / max(1, batches):.1f}/batch, cap {batch_size}, "
                f"{len(buckets)} distinct shapes)",
                flush=True,
            )
        return results

    def _decode_chunk(self, tensors: list, chunk: list[int], results: list[str | None]) -> None:
        import torch

        batch = torch.stack([tensors[i] for i in chunk]).to(self.device)
        with self._autocast():
            decoded = self._ocr.model.generate(batch, temperature=self._ocr.args.get("temperature", 0.25))
        for index, row in zip(chunk, decoded):
            results[index] = self._row_to_latex(row)


def render_crops(pdf_path: Path, merged_dir: Path, output_dir: Path, dpi: int = _CROP_DPI) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Rasterize this paper's formula regions.

    Returns the result records (with `latex` still unfilled) alongside the
    crop images, so the caller can pool crops from several papers into one
    GPU batch and fill the records in afterwards.
    """
    pages_data = read_json(merged_dir / "merged.json")
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    records: list[dict[str, Any]] = []
    images: list[Image.Image] = []
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

                with Image.open(crops_dir / filename) as raw:
                    images.append(_upscale_tiny(raw.convert("RGB")))
                records.append(
                    {
                        "id": block["id"],
                        "page": page_data["page"],
                        "bbox": block["bbox"],
                        "image": filename,
                        "latex": None,
                    }
                )
    finally:
        doc.close()
    return records, images


def extract_formulas(
    pdf_path: Path,
    merged_dir: Path,
    output_dir: Path,
    dpi: int = _CROP_DPI,
    device: str = "auto",
    ocr: Pix2TexBatchOCR | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Extract one paper's formulas, batching within that paper only."""
    if ocr is None:
        ocr = Pix2TexBatchOCR(device)
    records, images = render_crops(pdf_path, merged_dir, output_dir, dpi)
    for record, latex in zip(records, ocr.decode([ocr.preprocess(im) for im in images], batch_size)):
        record["latex"] = latex
    write_json(records, output_dir / "formulas.json")
    return records


def _render_all(jobs: list[tuple[Path, ...]], dpi: int) -> Iterator[tuple[tuple[Path, ...], Any]]:
    """Rasterize each paper's crops in turn, yielding (job, outcome).

    Runs on `prefetch`'s background thread so the next paper is being
    rasterized while the current one's crops are on the GPU. `outcome` is
    either (records, images) or the exception that paper died of -- raising
    here would strand every paper behind it.
    """
    for job in jobs:
        pdf, merged_dir, out_dir = job
        try:
            yield job, render_crops(pdf, merged_dir, out_dir, dpi)
        except Exception as exc:
            yield job, exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: OCR formula regions to LaTeX via Pix2Tex.")
    parser.add_argument("pdf", type=Path, nargs="+", help="One or more input PDFs")
    parser.add_argument("--merged", type=Path, default=None, help="Phase 3 output directory (single PDF)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (single PDF)")
    parser.add_argument("--output-root", type=Path, default=None, help="Batch mode: <root>/<stem>/{merged,formulas}")
    parser.add_argument("--dpi", type=int, default=_CROP_DPI)
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="Accelerator for Pix2Tex. 'auto' uses CUDA when available (mps is unsupported -> cpu).",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="Inference precision. 'auto' is fp16 on CUDA, fp32 everywhere else.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Crops decoded per model call. Higher keeps the GPU busier but uses more VRAM.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Crops pooled across papers before decoding. Higher fills batches better, costs memory.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip pix2tex's per-crop iterative image resizer. Much less launch overhead (it cannot "
             "be batched), at some accuracy cost on oddly-scaled crops.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.queue_size < 1:
        parser.error("--batch-size and --queue-size must be at least 1")
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

    try:
        ocr = Pix2TexBatchOCR(args.device, args.precision, resize=not args.no_resize)
    except (RuntimeError, ValueError) as exc:  # unusable device: nothing to salvage
        sys.exit(str(exc))
    print(
        f"[formulas] device={describe_torch_device(ocr.device)} precision={'fp16' if ocr.half else 'fp32'} "
        f"batch_size={args.batch_size} queue_size={args.queue_size} resizer={not args.no_resize}"
    )

    # Crops waiting on the GPU, and the papers they belong to. A paper's
    # formulas.json is written once the flush that covers its crops lands.
    pending_tensors: list = []
    pending_records: list[dict[str, Any]] = []
    in_flight: list[tuple[Path, list[dict[str, Any]]]] = []
    failed = 0

    def flush() -> None:
        for record, latex in zip(pending_records, ocr.decode(pending_tensors, args.batch_size)):
            record["latex"] = latex
        pending_tensors.clear()
        pending_records.clear()
        for out_dir, records in in_flight:
            write_json(records, out_dir / "formulas.json")
            n_ok = sum(1 for r in records if r["latex"])
            print(f"Extracted {len(records)} formula regions ({n_ok} OCR'd successfully) -> {out_dir}")
        in_flight.clear()

    for job, outcome in prefetch(lambda: _render_all(jobs, args.dpi)):
        pdf, _merged_dir, out_dir = job
        try:
            if isinstance(outcome, Exception):
                raise outcome
            records, images = outcome
            # Preprocessed before anything is queued: a crop that fails to
            # prepare must not leave half a paper in the pending lists, where
            # it would silently shift every later record's LaTeX by one.
            tensors = [ocr.preprocess(image) for image in images]
        except Exception as exc:
            failed += 1
            report_paper_failure("formulas", pdf, exc)
            continue
        pending_records.extend(records)
        pending_tensors.extend(tensors)
        in_flight.append((out_dir, records))
        # Pooling is silent by nature -- nothing is written until the pool is
        # full enough to decode -- and with a large --queue-size that silence
        # runs long enough to look like a hang, especially since preprocessing
        # is where pix2tex's per-crop resizer spends its time. Say so.
        print(
            f"[formulas] pooled {len(records)} crops from {pdf.name} "
            f"({len(pending_tensors)}/{args.queue_size} before decode)",
            flush=True,
        )
        if len(pending_tensors) >= args.queue_size:
            flush()
    flush()

    if failed == len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
