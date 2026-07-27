"""Phase 1 — PDF Extraction.

Pulls everything that already exists inside the PDF (text spans, fonts,
coordinates, embedded images, page dimensions) using PyMuPDF only. No AI,
no OCR — this stage is pure "read what's in the file".

Usage:
    python extract_pdf.py 9709_s24_qp_12.pdf --output-dir output/extracted
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from schemas import BBox, FontInfo, ImageRef, PageExtraction, TextSpan, write_json

# span["flags"] bit values, per PyMuPDF's text extraction docs.
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4

# The "Symbol" font (a common holdover from Word/PowerPoint-generated PDFs)
# encodes Greek letters and math symbols as Latin ASCII codes shifted into
# the Private Use Area (0xF000 + ascii code) rather than their real Unicode
# code points, so raw extraction yields unprintable glyphs like ""
# instead of "α" (alpha). This is the standard Adobe Symbol encoding.
_SYMBOL_FONT_MAP = {
    "": "×",  # ×
    "": "·",  # · (used as a multiplication dot, e.g. "N·m")
    "": "±",  # ±
    **{chr(0xF000 + ord(lo)): gr for lo, gr in zip(
        "abcdefghiklmnopqrstuwxyz",
        "αβχδεφγηικλμνοπθρστυωξψζ",
    )},
    **{chr(0xF000 + ord(hi)): gr for hi, gr in zip(
        "DFGLPQSWXY",
        "ΔΦΓΛΠΘΣΩΞΨ",
    )},
}


def _remap_symbol_font(text: str, font_name: str) -> str:
    if font_name != "Symbol":
        return text
    return "".join(_SYMBOL_FONT_MAP.get(ch, ch) for ch in text)


# Exponents/unit powers (10^0) and chemical subscripts (H2O) are rendered as
# their own span at a smaller font size and shifted off the baseline, rather
# than as a single character with real superscript/subscript info attached.
# Convert those spans to the Unicode superscript/subscript block so "10", "0"
# comes back out as "10⁰" instead of two separate baseline characters.
_SIZE_RATIO_THRESHOLD = 0.85  # a span this much smaller than the line's main text is a super/subscript candidate
_BASELINE_SHIFT_PT = 1.0  # min vertical offset (points) from the main text to count as raised/lowered
# PDFs commonly typeset the minus in an exponent as an en dash (–) rather
# than an ASCII hyphen (-), so both need to map onto the same superscript minus.
_MAPPABLE_CHARS = "0123456789+-–()n"
_SUPERSCRIPT_TABLE = str.maketrans("0123456789+-–()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁽⁾ⁿ")
_SUBSCRIPT_TABLE = str.maketrans("0123456789+-–()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₍₎")


_ROW_CLUSTER_TOLERANCE_PT = 6.0  # vertical-center tolerance for grouping spans into one visual row


def _cluster_block_rows(spans: list[dict]) -> list[list[dict]]:
    """Group a block's spans into visual rows by vertical center. PyMuPDF's
    own "line" grouping sometimes splits a raised/lowered exponent into its
    own separate line object even though it visually sits inside the row
    above/below it, so super/subscript detection needs a coarser regrouping
    than the raw line structure gives us."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s["bbox"][1] + s["bbox"][3]) / 2)
    rows: list[list[dict]] = [[ordered[0]]]
    for span in ordered[1:]:
        last = rows[-1][-1]
        center = (span["bbox"][1] + span["bbox"][3]) / 2
        last_center = (last["bbox"][1] + last["bbox"][3]) / 2
        if center - last_center <= _ROW_CLUSTER_TOLERANCE_PT:
            rows[-1].append(span)
        else:
            rows.append([span])
    return rows


def _lift_super_subscripts(line_spans: list[dict]) -> None:
    """Mutate span["text"] in place, promoting small raised/lowered spans to
    Unicode superscript/subscript characters. No-op if there's no size
    variation in the line (nothing looks like an exponent) or if the small
    span contains characters outside the mappable set."""
    if len(line_spans) < 2:
        return
    ref_size = max(s["size"] for s in line_spans)
    ref_spans = [s for s in line_spans if s["size"] == ref_size]
    # Compare vertical *centers* rather than edges: ascenders/descenders on
    # individual glyphs (e.g. "x" in a multiplication sign) shift the top/
    # bottom of a bbox around even within normal baseline text, but the
    # center stays put. Use the median center across the full-size spans so
    # one odd glyph doesn't skew the reference.
    centers = sorted((s["bbox"][1] + s["bbox"][3]) / 2 for s in ref_spans)
    ref_center = centers[len(centers) // 2]

    for span in line_spans:
        if span["size"] >= ref_size * _SIZE_RATIO_THRESHOLD:
            continue
        center = (span["bbox"][1] + span["bbox"][3]) / 2
        if center < ref_center - _BASELINE_SHIFT_PT:
            table = _SUPERSCRIPT_TABLE
        elif center > ref_center + _BASELINE_SHIFT_PT:
            table = _SUBSCRIPT_TABLE
        else:
            continue
        stripped = span["text"].strip()
        if stripped and all(ch in _MAPPABLE_CHARS for ch in stripped):
            span["text"] = span["text"].replace(stripped, stripped.translate(table))


def _extract_spans(text_dict: dict) -> list[TextSpan]:
    spans: list[TextSpan] = []
    for block in text_dict["blocks"]:
        if block["type"] != 0:  # 0 = text block, 1 = image block
            continue
        block_spans = [span for line in block["lines"] for span in line["spans"]]
        for row in _cluster_block_rows(block_spans):
            _lift_super_subscripts(row)
        for line in block["lines"]:
            for span in line["spans"]:
                text = _remap_symbol_font(span["text"], span["font"])
                if not text.strip():
                    # A whitespace-only span still carries real inter-word
                    # spacing (e.g. the gap between "10⁰" and "N"); fold it
                    # into the previous span instead of dropping it, rather
                    # than persisting a separate span made of just spaces.
                    if spans:
                        spans[-1].text += text
                    continue
                spans.append(
                    TextSpan(
                        text=text,
                        bbox=BBox.from_xyxy(span["bbox"]),
                        font=FontInfo(
                            name=span["font"],
                            size=round(span["size"], 2),
                            color=span["color"],
                            bold=bool(span["flags"] & _FLAG_BOLD),
                            italic=bool(span["flags"] & _FLAG_ITALIC),
                        ),
                    )
                )
    return spans


def _extract_images(page: pymupdf.Page, doc: pymupdf.Document, images_dir: Path, page_number: int) -> list[ImageRef]:
    images: list[ImageRef] = []
    # get_image_info(xrefs=True) gives one entry per *placement* on the page
    # (a single embedded image can be placed more than once), with the bbox
    # already in page space and the xref needed to pull real pixel data.
    for idx, info in enumerate(page.get_image_info(xrefs=True)):
        xref = info.get("xref", 0)
        if not xref:
            continue
        try:
            extracted = doc.extract_image(xref)
        except Exception:
            continue

        filename = f"page{page_number}_img{idx}.{extracted['ext']}"
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / filename).write_bytes(extracted["image"])

        images.append(
            ImageRef(
                id=idx,
                bbox=BBox.from_xyxy(info["bbox"]),
                file=filename,
                width=extracted["width"],
                height=extracted["height"],
                ext=extracted["ext"],
            )
        )
    return images


_QUESTION_NUMBER_RE = re.compile(r"^(\d{1,2})[.):]?$")
_BOILERPLATE_RE = re.compile(r"©\s*UCLES|\bturn over\b|\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}", re.I)
_HEADER_ZONE = 60.0  # points from the top treated as running header
_FOOTER_ZONE = 55.0  # points from the bottom treated as running footer
_MARGIN_TOLERANCE = 2.0  # points; how close x0 must be to the detected margin bucket


@dataclass
class Question:
    """A question assembled purely from text spans: everything between one
    detected question-number marker and the next. Images/tables/formulas
    aren't attached yet -- that needs the layout regions from Phase 2."""

    number: int
    start_page: int
    end_page: int
    text: str
    spans: list[TextSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
        }


def _is_boilerplate(span: TextSpan, page_height: float) -> bool:
    if span.bbox.y0 <= _HEADER_ZONE or span.bbox.y1 >= page_height - _FOOTER_ZONE:
        return True
    return bool(_BOILERPLATE_RE.search(span.text))


def _detect_margin_x(numbered_spans: list[tuple[int, TextSpan]]) -> float:
    """The question-number column sits at one consistent x0; other bare numbers
    on the page (axis ticks, mark allocations, option labels) cluster elsewhere.
    Pick the x0 bucket (rounded to the nearest point) most spans agree on."""
    buckets = Counter(round(span.bbox.x0) for _, span in numbered_spans)
    return float(buckets.most_common(1)[0][0])


# Superscript/subscript characters (exponents, unit powers) sit a few points
# above or below their baseline text -- e.g. the "0" in "10^0" has a smaller
# y0 than the "N" it's printed next to, since PDF y increases downward and
# a raised exponent is drawn higher up. Sorting spans purely by y0 puts every
# exponent on the page before the row it belongs to. Cluster into visual rows
# within this tolerance first, then order left-to-right within each row.
_ROW_TOLERANCE_PT = 8.0


def _reading_order_spans(spans: list[tuple[int, TextSpan]]) -> list[tuple[int, TextSpan]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda ps: (ps[0], ps[1].bbox.y0))
    rows: list[list[tuple[int, TextSpan]]] = [[ordered[0]]]
    for page, span in ordered[1:]:
        last_page, last_span = rows[-1][-1]
        if page == last_page and abs(span.bbox.y0 - last_span.bbox.y0) <= _ROW_TOLERANCE_PT:
            rows[-1].append((page, span))
        else:
            rows.append([(page, span)])
    return [ps for row in rows for ps in sorted(row, key=lambda ps: ps[1].bbox.x0)]


def _join_spans_text(spans: list[TextSpan]) -> str:
    """Join span text in reading order. Real inter-word spacing is already
    embedded in each span's text (see the whitespace-folding in
    _extract_spans), so the only thing this needs to add is a break between
    visual rows that don't happen to end in whitespace themselves -- e.g. a
    line-wrapped stem that doesn't end its last span with a trailing space."""
    parts: list[str] = []
    prev: TextSpan | None = None
    for span in spans:
        text = span.text
        if parts and prev is not None:
            already_spaced = parts[-1][-1:].isspace() or text[:1].isspace()
            same_row = abs(span.bbox.y0 - prev.bbox.y0) <= _ROW_TOLERANCE_PT
            if not already_spaced and not same_row:
                parts.append(" ")
        parts.append(text)
        prev = span
    return re.sub(r"[ \t]+", " ", "".join(parts)).strip()


def group_spans_into_questions(pages: list[PageExtraction]) -> list[Question]:
    page_heights = {p.page: p.height for p in pages}

    body_spans: list[tuple[int, TextSpan]] = [
        (p.page, s) for p in pages for s in p.spans if not _is_boilerplate(s, p.height)
    ]

    numbered_candidates = [
        (page, s) for page, s in body_spans if _QUESTION_NUMBER_RE.fullmatch(s.text.strip())
    ]
    if not numbered_candidates:
        return []
    margin_x = _detect_margin_x(numbered_candidates)

    markers = [
        (page, s)
        for page, s in numbered_candidates
        if abs(s.bbox.x0 - margin_x) <= _MARGIN_TOLERANCE
    ]
    markers.sort(key=lambda ps: (ps[0], ps[1].bbox.y0))

    # Question numbers must strictly increase, but don't require an exact
    # 1, 2, 3, ... match: a genuinely missed marker (layout/OCR drops one
    # number) shouldn't cost every question after it. Requiring only
    # "greater than the last accepted number" tolerates that gap without
    # opening the door to reordering or duplicate numbers, unlike requiring
    # an exact match against a running "expected" counter (which permanently
    # breaks on the first gap and silently merges every remaining question
    # into the last one that matched).
    accepted: list[tuple[int, int, TextSpan]] = []  # (number, page, span)
    last_accepted = 0
    for page, span in markers:
        number = int(_QUESTION_NUMBER_RE.fullmatch(span.text.strip()).group(1))
        if number > last_accepted:
            accepted.append((number, page, span))
            last_accepted = number
    if not accepted:
        return []

    body_spans.sort(key=lambda ps: (ps[0], ps[1].bbox.y0))

    def marker_key(page: int, y0: float) -> tuple[int, float]:
        return (page, y0)

    # Shift each boundary up by the row tolerance: a superscript/subscript
    # that's visually part of a question's own marker row (e.g. the "-1" in
    # "N s-1 m-1") can still have a smaller y0 than the marker itself, since
    # raised text sorts earlier by raw y0 (see _ROW_TOLERANCE_PT above). Without
    # this, that span gets misattributed to the *previous* question instead.
    boundaries = [marker_key(page, span.bbox.y0 - _ROW_TOLERANCE_PT) for _, page, span in accepted]

    questions: list[Question] = []
    for idx, (number, start_page, _marker_span) in enumerate(accepted):
        start_key = boundaries[idx]
        end_key = boundaries[idx + 1] if idx + 1 < len(boundaries) else (float("inf"), float("inf"))

        q_pages_spans = [(page, s) for page, s in body_spans if start_key <= marker_key(page, s.bbox.y0) < end_key]
        end_page = max((page for page, _ in q_pages_spans), default=start_page)
        q_pages_spans = _reading_order_spans(q_pages_spans)
        q_spans = [s for _, s in q_pages_spans]
        text = _join_spans_text(q_spans)

        questions.append(Question(number=number, start_page=start_page, end_page=end_page, text=text, spans=q_spans))

    return questions


def extract_pdf(pdf_path: Path, output_dir: Path) -> tuple[list[PageExtraction], list[Question]]:
    images_dir = output_dir / "images"
    doc = pymupdf.open(pdf_path)
    pages: list[PageExtraction] = []

    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            text_dict = page.get_text("dict")
            page_extraction = PageExtraction(
                page=page_number,
                width=round(page.rect.width, 2),
                height=round(page.rect.height, 2),
                spans=_extract_spans(text_dict),
                images=_extract_images(page, doc, images_dir, page_number),
            )
            pages.append(page_extraction)
            write_json(page_extraction.to_dict(), output_dir / f"page{page_number}.json")
    finally:
        doc.close()

    write_json([p.to_dict() for p in pages], output_dir / "extraction.json")

    questions = group_spans_into_questions(pages)
    write_json([q.to_dict() for q in questions], output_dir / "questions.json")

    return pages, questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: extract text/images/coords from a PDF via PyMuPDF.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF")
    parser.add_argument("--output-dir", type=Path, default=Path("output/extracted"), help="Directory for JSON + extracted images")
    args = parser.parse_args()

    pages, questions = extract_pdf(args.pdf, args.output_dir)
    n_spans = sum(len(p.spans) for p in pages)
    n_images = sum(len(p.images) for p in pages)
    print(f"Extracted {len(pages)} pages, {n_spans} text spans, {n_images} images -> {args.output_dir}")
    print(f"Grouped into {len(questions)} questions -> {args.output_dir / 'questions.json'}")


if __name__ == "__main__":
    main()
