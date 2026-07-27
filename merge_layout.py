"""Phase 3 — Merge Extraction Results.

Combines Phase 1 (PyMuPDF spans/images, exact but structure-blind) with
Phase 2 (layout regions, structure-aware but content-blind): every span and
image gets assigned to whichever detected region it falls inside. Nothing
here is model-specific -- it only reads the JSON contracts both phases
already produce, so it doesn't care whether Phase 2 ran PPStructure or
DocLayout-YOLO.

Usage:
    python merge_layout.py --extracted output/extracted --layout output/layout --output-dir output/merged
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas import BBox, FontInfo, ImageRef, LayoutType, TextSpan, read_json, write_json

# A span/image must have at least this fraction of its own area inside a
# region to be assigned to it. Below this, it's kept as an "unassigned" block
# rather than forced into the nearest (possibly wrong) region.
_MIN_CONTAINMENT = 0.5


@dataclass
class MergedBlock:
    id: str
    type: LayoutType
    bbox: BBox
    score: float
    raw_label: str
    spans: list[TextSpan] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "bbox": self.bbox.to_dict(),
            "score": self.score,
            "raw_label": self.raw_label,
            "content": [s.to_dict() for s in self.spans],
            "images": [i.to_dict() for i in self.images],
        }


@dataclass
class MergedPage:
    page: int
    width: float
    height: float
    blocks: list[MergedBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
        }


def _best_region(item_bbox: BBox, regions: list[MergedBlock]) -> MergedBlock | None:
    area = item_bbox.area()
    if area == 0:
        return None
    candidates = [r for r in regions if item_bbox.intersection_area(r.bbox) / area >= _MIN_CONTAINMENT]
    if not candidates:
        return None
    # Layout regions often nest (e.g. a small "formula" box for a single
    # exponent sitting inside a wider "text" box for the whole line). When an
    # item is contained well enough by more than one region, it belongs to the
    # most specific (smallest-area) one, not whichever region happened to
    # score highest during detection.
    return min(candidates, key=lambda r: r.bbox.area())


def merge_page(extraction: dict, layout: dict) -> MergedPage:
    regions = [
        MergedBlock(
            id=r["id"],
            type=LayoutType(r["type"]),
            bbox=BBox.from_dict(r["bbox"]),
            score=r["score"],
            raw_label=r["raw_label"],
        )
        for r in layout["regions"]
    ]

    spans = [
        TextSpan(text=s["text"], bbox=BBox.from_dict(s["bbox"]), font=FontInfo(**s["font"]))
        for s in extraction["spans"]
    ]

    unassigned = MergedBlock(
        id=f"p{extraction['page']}_unassigned",
        type=LayoutType.OTHER,
        bbox=BBox(0, 0, extraction["width"], extraction["height"]),
        score=0.0,
        raw_label="unassigned",
    )

    for span in spans:
        region = _best_region(span.bbox, regions)
        (region or unassigned).spans.append(span)

    for img in extraction["images"]:
        image_ref = ImageRef(
            id=img["id"], bbox=BBox.from_dict(img["bbox"]), file=img["file"], width=img["width"], height=img["height"], ext=img["ext"]
        )
        region = _best_region(image_ref.bbox, regions)
        (region or unassigned).images.append(image_ref)

    blocks = sorted(regions, key=lambda b: (b.bbox.y0, b.bbox.x0))
    if unassigned.spans or unassigned.images:
        blocks.append(unassigned)

    return MergedPage(page=extraction["page"], width=extraction["width"], height=extraction["height"], blocks=blocks)


def merge_layout(extracted_dir: Path, layout_dir: Path, output_dir: Path) -> list[MergedPage]:
    extractions = {p["page"]: p for p in read_json(extracted_dir / "extraction.json")}
    layouts = {p["page"]: p for p in read_json(layout_dir / "layout.json")}

    pages: list[MergedPage] = []
    for page_number in sorted(extractions):
        merged = merge_page(extractions[page_number], layouts.get(page_number, {"regions": []}))
        pages.append(merged)
        write_json(merged.to_dict(), output_dir / f"page{page_number}_merged.json")

    write_json([p.to_dict() for p in pages], output_dir / "merged.json")
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: assign Phase 1 spans/images to Phase 2 layout regions.")
    parser.add_argument("--extracted", type=Path, default=Path("output/extracted"), help="Phase 1 output directory")
    parser.add_argument("--layout", type=Path, default=Path("output/layout"), help="Phase 2 output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/merged"))
    args = parser.parse_args()

    pages = merge_layout(args.extracted, args.layout, args.output_dir)
    n_blocks = sum(len(p.blocks) for p in pages)
    print(f"Merged {len(pages)} pages into {n_blocks} blocks -> {args.output_dir}")


if __name__ == "__main__":
    main()
