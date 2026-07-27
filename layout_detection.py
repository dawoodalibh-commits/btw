"""Phase 2 — Layout Detection.

Understands the *structure* of each page (text / title / image / formula /
table / header / footer regions) without reading any of the content. No OCR
happens here — only bounding boxes and region types.

The detector backend is swappable behind the `LayoutDetector` interface:
today it's PP-DocLayout (PaddleOCR's PPStructureV3 layout model) or
DocLayout-YOLO, but any future model just needs a new subclass that maps its
own labels onto `schemas.LayoutType` and returns `LayoutRegion`s in PDF
point space. Nothing outside this file needs to know which backend ran.

Usage:
    python layout_detection.py 9709_s24_qp_12.pdf --backend ppstructure
    python layout_detection.py 9709_s24_qp_12.pdf --backend doclayout_yolo --device mps
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf

from schemas import BBox, LayoutRegion, LayoutType, PageLayout, write_json

# Pages are rasterized at this resolution before being fed to a layout
# detector (detectors work on images, not vector PDFs). Boxes returned by
# the detector are in pixel space at this DPI and get rescaled back to PDF
# point space (72 dpi) before being wrapped in a LayoutRegion, so callers
# never need to know the DPI.
DEFAULT_DPI = 200
_POINTS_PER_INCH = 72.0

# Accepted --device values. "auto" asks the backend to pick the best
# accelerator it can actually use; the rest force a specific one.
DEVICES = ("auto", "cpu", "cuda", "mps")


def _best_torch_device() -> str:
    """Best torch device available here: CUDA > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LayoutDetector(ABC):
    """Common interface every layout-detection backend must implement.

    Subclasses set `self.device` to the accelerator they actually ended up
    running on, which is not always what was requested — a backend that
    can't use the requested device falls back rather than failing.
    """

    device: str = "cpu"

    @abstractmethod
    def detect(self, image_path: Path, dpi: int) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        """Return (raw_label, score, (x0, y0, x1, y1)) in *pixel* space for one page image."""

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

        # Paddle has no Metal backend, so "mps" (and "auto" on a Mac) can only
        # mean CPU here; it spells CUDA "gpu". "auto" is left to paddle itself,
        # which picks a GPU when its GPU build is installed.
        self.device = "cpu" if device == "mps" else device
        paddle_device = {"auto": None, "cpu": "cpu", "cuda": "gpu", "mps": "cpu"}[device]
        self._model = LayoutDetection() if paddle_device is None else LayoutDetection(device=paddle_device)

    def label_map(self) -> dict[str, LayoutType]:
        return self._LABEL_MAP

    def detect(self, image_path: Path, dpi: int) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        results = self._model.predict(str(image_path))
        boxes = results[0].json["res"]["boxes"]
        return [(b["label"], float(b["score"]), tuple(b["coordinate"])) for b in boxes]


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

    def __init__(self, device: str = "auto", confidence: float = 0.2, imgsz: int = 1024) -> None:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=self._HF_REPO, filename=self._HF_FILENAME)
        self._model = YOLOv10(weights_path)
        self._confidence = confidence
        self._imgsz = imgsz
        self.device = _best_torch_device() if device == "auto" else device

    def label_map(self) -> dict[str, LayoutType]:
        return self._LABEL_MAP

    def detect(self, image_path: Path, dpi: int) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        results = self._model.predict(
            str(image_path), imgsz=self._imgsz, conf=self._confidence, device=self.device, verbose=False
        )
        result = results[0]
        out = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            score = float(box.conf.item())
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            out.append((label, score, (x0, y0, x1, y1)))
        return out


_BACKENDS = {
    "ppstructure": PPStructureLayoutDetector,
    "doclayout_yolo": DocLayoutYOLODetector,
}


def build_detector(backend: str, device: str = "auto") -> LayoutDetector:
    if device not in DEVICES:
        raise ValueError(f"Unknown device {device!r}. Choose from {list(DEVICES)}")
    try:
        return _BACKENDS[backend](device=device)
    except KeyError:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}")


def detect_layout(
    pdf_path: Path,
    output_dir: Path,
    backend: str = "ppstructure",
    dpi: int = DEFAULT_DPI,
    device: str = "auto",
) -> list[PageLayout]:
    detector = build_detector(backend, device)
    print(f"[layout] backend={backend} device={detector.device}")
    render_dir = output_dir / "page_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    scale = _POINTS_PER_INCH / dpi  # converts pixel coords at `dpi` back to PDF points

    doc = pymupdf.open(pdf_path)
    pages: list[PageLayout] = []
    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            pix = page.get_pixmap(dpi=dpi)
            image_path = render_dir / f"page{page_number}.png"
            pix.save(image_path)

            regions = [
                LayoutRegion(
                    id=f"p{page_number}_r{i}",
                    type=detector.canonicalize(raw_label),
                    bbox=BBox.from_xyxy(xyxy).scaled(scale),
                    score=score,
                    raw_label=raw_label,
                )
                for i, (raw_label, score, xyxy) in enumerate(detector.detect(image_path, dpi))
            ]
            page_layout = PageLayout(
                page=page_number,
                width=round(page.rect.width, 2),
                height=round(page.rect.height, 2),
                regions=regions,
            )
            pages.append(page_layout)
            write_json(page_layout.to_dict(), output_dir / f"page{page_number}_layout.json")
    finally:
        doc.close()

    write_json([p.to_dict() for p in pages], output_dir / "layout.json")
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: detect page layout regions (no OCR).")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--output-dir", type=Path, default=Path("output/layout"), help="Directory for JSON + page renders")
    parser.add_argument("--backend", choices=list(_BACKENDS), default="ppstructure")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rasterization DPI fed to the layout model")
    parser.add_argument(
        "--device",
        choices=list(DEVICES),
        default="auto",
        help="Accelerator for the detector. 'auto' picks the best one the backend supports.",
    )
    args = parser.parse_args()

    pages = detect_layout(args.pdf, args.output_dir, backend=args.backend, dpi=args.dpi, device=args.device)
    n_regions = sum(len(p.regions) for p in pages)
    print(f"Detected layout on {len(pages)} pages, {n_regions} regions (backend={args.backend}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
