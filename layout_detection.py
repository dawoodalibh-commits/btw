"""Phase 2 — Layout Detection.

Understands the *structure* of each page (text / title / image / formula /
table / header / footer regions) without reading any of the content. No OCR
happens here — only bounding boxes and region types.

The detector backend is swappable behind the `LayoutDetector` interface:
today it's PP-DocLayout (PaddleOCR's PPStructureV3 layout model) or
DocLayout-YOLO, but any future model just needs a new subclass that maps its
own labels onto `schemas.LayoutType` and returns `LayoutRegion`s in PDF
point space. Nothing outside this file needs to know which backend ran.

Pages are fed to the detector in batches rather than one at a time: a single
page image is nowhere near enough work to saturate a GPU, so per-page calls
spend most of their time in launch overhead with the card mostly idle. Both
backends infer a whole list of images in one call, and rasterization for the
next batch runs on a background thread while the current one is on the GPU.

Usage:
    python layout_detection.py 9709_s24_qp_12.pdf --backend ppstructure
    python layout_detection.py 9709_s24_qp_12.pdf --backend doclayout_yolo --device mps
"""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pymupdf

from accel import DEVICES, describe_torch_device, prefetch, resolve_paddle_device, resolve_torch_device
from schemas import (
    BBox,
    LayoutRegion,
    LayoutType,
    PageLayout,
    report_paper_failure,
    resolve_batch_jobs,
    write_json,
)

# Pages are rasterized at this resolution before being fed to a layout
# detector (detectors work on images, not vector PDFs). Boxes returned by
# the detector are in pixel space at this DPI and get rescaled back to PDF
# point space (72 dpi) before being wrapped in a LayoutRegion, so callers
# never need to know the DPI.
DEFAULT_DPI = 200
_POINTS_PER_INCH = 72.0

# Pages per detector call. Eight 200-dpi pages is a few hundred MB of
# activations at most, which fits comfortably on any card worth running this
# on, while being enough work per launch that the GPU stops idling between
# calls. Raise it for more headroom, lower it if VRAM is tight.
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class PageImage:
    """One rasterized page, handed to detectors as a decoded BGR array.

    Both backends take the array rather than a file path. That isn't a
    micro-optimization: at 200 dpi, encoding the PNG costs ~46 ms/page
    against ~5 ms to rasterize it, and handing back a path makes the detector
    spend another ~16 ms decoding what was just written. Skipping the round
    trip cuts the CPU cost of feeding the GPU from ~68 ms/page to ~14 ms,
    which is the difference between the detector waiting on the card and the
    card waiting on the detector.
    """

    page: int
    width: float
    height: float
    array: Any  # np.ndarray, BGR — typed loosely to keep numpy out of import time
    path: Path | None = None  # only set when --save-renders asked for the PNG


class LayoutDetector(ABC):
    """Common interface every layout-detection backend must implement.

    Subclasses set `self.device` to the accelerator they actually ended up
    running on, read back from the framework rather than echoed from the
    request, so the banner each phase prints can't claim a GPU it never got.
    """

    device: str = "cpu"

    @abstractmethod
    def detect_batch(self, pages: list[PageImage]) -> list[list[tuple[str, float, tuple[float, float, float, float]]]]:
        """Detect on several pages in one call.

        Returns one list of (raw_label, score, (x0, y0, x1, y1)) per input
        page, in input order, with coordinates in *pixel* space.
        """

    def label_map(self) -> dict[str, LayoutType]:
        """Backend label -> canonical LayoutType. Unmapped labels fall back to OTHER."""
        return {}

    def canonicalize(self, raw_label: str) -> LayoutType:
        return self.label_map().get(raw_label, LayoutType.OTHER)


class PPStructureLayoutDetector(LayoutDetector):
    """Backend using PaddleOCR's PP-DocLayout model (the layout stage of PPStructureV3),
    run standalone so no OCR/table-recognition work happens."""

    _LABEL_MAP = {
        "doc_title": LayoutType.TITLE,
        "paragraph_title": LayoutType.TITLE,
        "figure_title": LayoutType.TITLE,
        "text": LayoutType.TEXT,
        "abstract": LayoutType.TEXT,
        "content": LayoutType.TEXT,
        "reference": LayoutType.TEXT,
        "reference_content": LayoutType.TEXT,
        "footnote": LayoutType.TEXT,
        "aside_text": LayoutType.TEXT,
        "algorithm": LayoutType.TEXT,
        "image": LayoutType.IMAGE,
        "chart": LayoutType.IMAGE,
        "seal": LayoutType.IMAGE,
        "formula": LayoutType.FORMULA,
        "formula_number": LayoutType.FORMULA,
        "table": LayoutType.TABLE,
        "header": LayoutType.HEADER,
        "footer": LayoutType.FOOTER,
        "number": LayoutType.OTHER,
    }

    def __init__(self, device: str = "auto") -> None:
        from paddleocr import LayoutDetection  # local import: keep paddle out of the base module

        self.device = resolve_paddle_device(device)
        self._model = LayoutDetection(device=self.device)

    def label_map(self) -> dict[str, LayoutType]:
        return self._LABEL_MAP

    def detect_batch(self, pages: list[PageImage]) -> list[list[tuple[str, float, tuple[float, float, float, float]]]]:
        # batch_size is not optional here: paddlex's predictors default their
        # batch sampler to 1, so handing predict() a list without it walks the
        # list one image per forward pass and looks like batching while
        # leaving the GPU exactly as idle as before.
        #
        # Arrays must be BGR. Feeding RGB is not an error paddle reports -- it
        # returns the same number of boxes with quietly different scores and
        # coordinates. BGR was verified byte-identical to passing file paths.
        results = self._model.predict([p.array for p in pages], batch_size=len(pages))
        return [
            [(b["label"], float(b["score"]), tuple(b["coordinate"])) for b in r.json["res"]["boxes"]]
            for r in results
        ]


class DocLayoutYOLODetector(LayoutDetector):
    """Backend using DocLayout-YOLO (DocStructBench weights)."""

    _LABEL_MAP = {
        "title": LayoutType.TITLE,
        "plain text": LayoutType.TEXT,
        "figure": LayoutType.IMAGE,
        "figure_caption": LayoutType.CAPTION,
        "table": LayoutType.TABLE,
        "table_caption": LayoutType.CAPTION,
        "table_footnote": LayoutType.CAPTION,
        "isolate_formula": LayoutType.FORMULA,
        "formula_caption": LayoutType.CAPTION,
        "abandon": LayoutType.OTHER,  # headers/footers/page numbers/watermarks in DocStructBench
    }

    _HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
    _HF_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

    def __init__(self, device: str = "auto", confidence: float = 0.2, imgsz: int = 1024, half: bool | None = None) -> None:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=self._HF_REPO, filename=self._HF_FILENAME)
        self._model = YOLOv10(weights_path)
        self._confidence = confidence
        self._imgsz = imgsz
        self.device = resolve_torch_device(device)
        # fp16 roughly halves the weight/activation traffic this model is
        # bound by, for a detector whose scores move in the third decimal.
        # CUDA only: CPU fp16 is emulated and slower, and MPS gains nothing.
        self._half = (self.device == "cuda") if half is None else half

    def label_map(self) -> dict[str, LayoutType]:
        return self._LABEL_MAP

    def detect_batch(self, pages: list[PageImage]) -> list[list[tuple[str, float, tuple[float, float, float, float]]]]:
        # Arrays rather than paths: a list of ndarrays is loaded as one batch
        # and skips re-decoding the PNGs we just encoded. Verified to give
        # bit-identical boxes to the per-path calls this replaced.
        results = self._model.predict(
            [p.array for p in pages],
            imgsz=self._imgsz,
            conf=self._confidence,
            device=self.device,
            half=self._half,
            verbose=False,
        )
        out = []
        for result in results:
            page_out = []
            # One .tolist() for the whole page beats three .item() calls per
            # box: each of those is a separate device-to-host sync.
            for (x0, y0, x1, y1), cls, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                page_out.append((result.names[int(cls)], float(conf), (x0, y0, x1, y1)))
            out.append(page_out)
        return out


_BACKENDS = {
    "ppstructure": PPStructureLayoutDetector,
    "doclayout_yolo": DocLayoutYOLODetector,
}


def build_detector(backend: str, device: str = "auto") -> LayoutDetector:
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}")
    return _BACKENDS[backend](device=device)


def _render_batches(
    pdf_path: Path, render_dir: Path, dpi: int, batch_size: int, save_renders: bool
) -> Iterator[list[PageImage]]:
    """Rasterize the PDF, yielding one batch of pages at a time.

    Runs on `prefetch`'s background thread, so rasterizing the next batch
    overlaps the GPU's work on the current one. A single pymupdf Document is
    not safe to share between threads, so it's opened and closed entirely
    inside this generator, which only ever runs on one.

    This generator is what has to keep ahead of the GPU, which is why the PNG
    write is off by default: at ~46 ms/page it is nine times the cost of the
    rasterization itself and dominates everything else here.
    """
    import numpy as np

    doc = pymupdf.open(pdf_path)
    try:
        batch: list[PageImage] = []
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            pix = page.get_pixmap(dpi=dpi)
            image_path = None
            if save_renders:
                image_path = render_dir / f"page{page_number}.png"
                pix.save(image_path)
            # Straight from the pixmap's own buffer. Pixmaps are RGB; both
            # detectors follow OpenCV's BGR convention, hence the reverse.
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            batch.append(
                PageImage(
                    page=page_number,
                    width=round(page.rect.width, 2),
                    height=round(page.rect.height, 2),
                    array=np.ascontiguousarray(rgb[:, :, 2::-1]),
                    path=image_path,
                )
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        doc.close()


def detect_layout(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "ppstructure",
    dpi: int = DEFAULT_DPI,
    device: str = "auto",
    detector: LayoutDetector | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_renders: bool = False,
) -> list[PageLayout]:
    # An already-built detector can be passed in so a batch of PDFs pays the
    # model-load cost once instead of once per PDF.
    if detector is None:
        detector = build_detector(backend, device)
        print(f"[layout] backend={backend} device={describe_torch_device(detector.device)}")
    render_dir = output_dir / "page_renders"
    if save_renders:
        render_dir.mkdir(parents=True, exist_ok=True)
    scale = _POINTS_PER_INCH / dpi  # converts pixel coords at `dpi` back to PDF points

    pages: list[PageLayout] = []
    for batch in prefetch(lambda: _render_batches(pdf_path, render_dir, dpi, batch_size, save_renders)):
        for page_image, raw_regions in zip(batch, detector.detect_batch(batch)):
            regions = [
                LayoutRegion(
                    id=f"p{page_image.page}_r{i}",
                    type=detector.canonicalize(raw_label),
                    bbox=BBox.from_xyxy(xyxy).scaled(scale),
                    score=score,
                    raw_label=raw_label,
                )
                for i, (raw_label, score, xyxy) in enumerate(raw_regions)
            ]
            page_layout = PageLayout(
                page=page_image.page,
                width=page_image.width,
                height=page_image.height,
                regions=regions,
            )
            pages.append(page_layout)
            write_json(page_layout.to_dict(), output_dir / f"page{page_image.page}_layout.json")

    write_json([p.to_dict() for p in pages], output_dir / "layout.json")
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: detect page layout regions (no OCR).")
    parser.add_argument("pdf", type=Path, nargs="+", help="One or more input PDFs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for JSON + page renders (single PDF)")
    parser.add_argument("--output-root", type=Path, default=None, help="Batch mode: write to <root>/<stem>/layout")
    parser.add_argument("--backend", choices=list(_BACKENDS), default="ppstructure")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rasterization DPI fed to the layout model")
    parser.add_argument(
        "--device",
        choices=list(DEVICES),
        default="auto",
        help="Accelerator for the detector. 'auto' picks the best one the backend supports.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Pages per detector call. Higher keeps the GPU busier but uses more VRAM.",
    )
    parser.add_argument(
        "--save-renders",
        action="store_true",
        help="Also write each page render to <output>/page_renders/. Debugging only -- no later "
             "phase reads them, and the PNG encode is ~9x the cost of the rasterization.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    try:
        jobs = resolve_batch_jobs(args.pdf, args.output_root, ["layout"], [args.output_dir], ["output/layout"])
    except ValueError as exc:
        parser.error(str(exc))

    try:
        detector = build_detector(args.backend, args.device)
    except (RuntimeError, ValueError) as exc:  # unusable device/backend: nothing to salvage
        sys.exit(str(exc))
    print(
        f"[layout] backend={args.backend} device={describe_torch_device(detector.device)} "
        f"batch_size={args.batch_size}"
    )

    failed = 0
    for pdf, out_dir in jobs:
        try:
            pages = detect_layout(
                pdf,
                out_dir,
                backend=args.backend,
                dpi=args.dpi,
                detector=detector,
                batch_size=args.batch_size,
                save_renders=args.save_renders,
            )
        except Exception as exc:  # one bad PDF shouldn't abandon the rest of the batch
            failed += 1
            report_paper_failure("layout", pdf, exc)
            continue
        n_regions = sum(len(p.regions) for p in pages)
        print(f"Detected layout on {len(pages)} pages, {n_regions} regions (backend={args.backend}) -> {out_dir}")

    if failed == len(jobs):
        sys.exit(1)


if __name__ == "__main__":
    main()
