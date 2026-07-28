"""Shared data structures for the extraction pipeline.

Every stage reads and writes plain JSON on disk using these schemas as the
contract between stages. A stage's *internal* implementation (which OCR
model, which layout detector) can change freely as long as it still emits
this shape — that's what lets Phase 2's backend be swapped from PPStructure
to DocLayout-YOLO (or anything else) without touching Phase 1, Phase 3, etc.

Coordinate convention: every bbox in every stage's output is in PDF point
space (72 points/inch, origin top-left, y increasing downward) — the same
space PyMuPDF's `page.rect` uses. Stages that work from a rasterized image
(e.g. layout detection) must rescale their pixel-space boxes back into this
space before writing output, so that boxes from different stages can be
compared directly (e.g. in Phase 3's merge step) without unit conversion.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

# Printed on stdout by a batched phase when one paper in its batch fails.
# The batch driver watches for this: a phase that sweeps many papers in one
# process still exits 0 when only some of them broke, so its exit code can't
# say which papers are still healthy, and a paper whose layout failed has to
# be dropped before it reaches the phases that read that layout.
PAPER_FAILED_PREFIX = "!!! PAPER-FAILED"


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x0": round(self.x0, 2),
            "y0": round(self.y0, 2),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "BBox":
        return cls(d["x0"], d["y0"], d["x1"], d["y1"])

    @classmethod
    def from_xyxy(cls, xyxy: Sequence[float]) -> "BBox":
        x0, y0, x1, y1 = xyxy
        return cls(float(x0), float(y0), float(x1), float(y1))

    def scaled(self, factor: float) -> "BBox":
        return BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection_area(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class FontInfo:
    name: str
    size: float
    color: int
    bold: bool
    italic: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextSpan:
    text: str
    bbox: BBox
    font: FontInfo

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "bbox": self.bbox.to_dict(), "font": self.font.to_dict()}


@dataclass
class ImageRef:
    id: int
    bbox: BBox
    file: str
    width: int
    height: int
    ext: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "ext": self.ext,
        }


@dataclass
class PageExtraction:
    """Phase 1 output: everything PyMuPDF can read straight out of the PDF."""

    page: int
    width: float
    height: float
    spans: list[TextSpan] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "spans": [s.to_dict() for s in self.spans],
            "images": [i.to_dict() for i in self.images],
        }


class LayoutType(str, Enum):
    """Canonical region taxonomy that every layout-detector backend must map its
    own label set onto, so downstream stages never see backend-specific labels."""

    TEXT = "text"
    TITLE = "title"
    IMAGE = "image"
    FORMULA = "formula"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    OTHER = "other"


@dataclass
class LayoutRegion:
    type: LayoutType
    bbox: BBox
    score: float
    raw_label: str  # original backend label, kept for debugging/auditing
    id: str = ""  # stable "p{page}_r{index}" key later stages join on

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "bbox": self.bbox.to_dict(),
            "score": round(self.score, 4),
            "raw_label": self.raw_label,
        }


@dataclass
class PageLayout:
    """Phase 2 output: detected regions on a page, no OCR performed."""

    page: int
    width: float
    height: float
    regions: list[LayoutRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
        }


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def report_paper_failure(phase: str, pdf: Path, exc: BaseException) -> None:
    """Announce that one paper failed, without abandoning the rest of the batch.

    Two lines on purpose: a machine-readable one on stdout for the batch
    driver to parse, and the human-readable reason on stderr where the rest
    of the phase's diagnostics go.
    """
    print(f"{PAPER_FAILED_PREFIX}\t{phase}\t{pdf}", flush=True)
    print(f"!!! FAILED {phase} for {pdf}: {exc}", file=sys.stderr, flush=True)


def resolve_batch_jobs(
    pdfs: Sequence[Path],
    output_root: Path | None,
    subdirs: Sequence[str],
    singles: Sequence[Path | None],
    defaults: Sequence[str],
) -> list[tuple[Path, ...]]:
    """Pairs each PDF with the directories a phase reads from and writes to.

    The model-loading phases (2, 5, 7) can sweep a whole batch in one process
    so the model is loaded once rather than once per paper. In that mode
    (`--output-root`) every directory is derived as <root>/<stem>/<subdir>, so
    papers can't clobber each other. Without it the phase keeps its original
    single-PDF behaviour and uses the explicitly passed directories.

    Returns one (pdf, *dirs) tuple per PDF, with dirs in `subdirs` order.
    """
    if output_root is not None:
        return [tuple([pdf, *(output_root / pdf.stem / sub for sub in subdirs)]) for pdf in pdfs]
    if len(pdfs) > 1:
        raise ValueError("--output-root is required when passing more than one PDF")
    return [tuple([pdfs[0], *(single or Path(default) for single, default in zip(singles, defaults))])]
