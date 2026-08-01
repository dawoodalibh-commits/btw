"""Phase 4 — Question Parser.

The intelligence layer: takes Phase 3's merged blocks (spans already assigned
to typed regions), strips headers/footers/boilerplate, detects question
numbers, and groups every block until the next question marker into one
question. Image/formula/table blocks are attached by region id only -- their
actual content (LaTeX, cropped files, table rows) comes from Phases 5-7 and
gets joined back in by build_questions.py using those same ids.

Usage:
    python question_parser.py --merged output/merged --output-dir output/questions
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas import LayoutType, read_json, write_json

_QUESTION_NUMBER_RE = re.compile(r"^(\d{1,2})[.):]?$")
_BOILERPLATE_RE = re.compile(r"©\s*UCLES|\bturn over\b|\d{4}/\d{2}/[A-Z]/[A-Z]/\d{2}", re.I)
# Cambridge papers end with a multi-paragraph copyright colophon after the
# last question, with no question marker to bound it. It always opens with
# this phrase, so truncate the document there rather than trying to pattern-
# match every sentence of a notice that can be reworded year to year.
_COLOPHON_START_RE = re.compile(r"permission to reproduce items", re.I)

# The right-hand "For Examiner's Use" column on pre-2010 papers. Matched on
# the two words alone so the curly vs straight apostrophe never matters.
_EXAMINER_COLUMN_RE = re.compile(r"For\s+Examiner", re.I)
# Letters that must survive removal of that phrase before a block counts as
# real content rather than the column.
_EXAMINER_REMAINDER_LETTERS = 15

# Block types whose text reads naturally as part of the question stem.
_TEXT_BEARING_TYPES = {LayoutType.TEXT, LayoutType.TITLE, LayoutType.CAPTION, LayoutType.FORMULA}

# Layout detection often carves inline math out of a text line into its own
# tiny "formula" block (e.g. "Ex" inside "...e.m.f. Ex of a cell..."), so one
# visual line ends up split across several blocks whose *block*-level bboxes
# don't reliably agree on a y0 -- a multi-line paragraph block's bbox only
# reflects its top edge, and a block trimmed by having math carved out of it
# can be drawn wider than its remaining content. Reading order is only
# reliable at the *span* level, using each span's own bbox, not its owning
# block's. Cluster spans into rows within this y-tolerance, then order
# left-to-right within each row -- same technique build_questions.py uses for
# MCQ option labels. Wide enough to chain a built-up fraction's numerator/
# label-baseline/denominator bands together (each individual gap runs ~6-8pt
# in practice) while staying well short of the ~12pt+ gap between genuinely
# separate lines of body text.
_ROW_TOLERANCE_PT = 9.0


def _reading_order_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """spans: dicts with "page" and "bbox" keys (plus whatever else the caller needs)."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s["page"], s["bbox"]["y0"]))
    rows: list[list[dict[str, Any]]] = [[ordered[0]]]
    for s in ordered[1:]:
        last = rows[-1][-1]
        if s["page"] == last["page"] and abs(s["bbox"]["y0"] - last["bbox"]["y0"]) <= _ROW_TOLERANCE_PT:
            rows[-1].append(s)
        else:
            rows.append([s])
    return [s for row in rows for s in sorted(row, key=lambda s: s["bbox"]["x0"])]


@dataclass
class RegionRef:
    id: str
    page: int
    bbox: dict

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "page": self.page, "bbox": self.bbox}


@dataclass
class ParsedQuestion:
    number: int
    page: int
    end_page: int
    text: str
    lines: list[str] = field(default_factory=list)  # one entry per text/title block, in reading order
    spans: list[dict[str, Any]] = field(default_factory=list)  # raw {"text","bbox"} spans, reading order
    images: list[RegionRef] = field(default_factory=list)
    tables: list[RegionRef] = field(default_factory=list)
    formulas: list[RegionRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.number,
            "page": self.page,
            "end_page": self.end_page,
            "lines": self.lines,
            "spans": self.spans,
            "text": self.text,
            "images": [r.to_dict() for r in self.images],
            "tables": [r.to_dict() for r in self.tables],
            "formulas": [r.to_dict() for r in self.formulas],
        }


def _is_boilerplate_block(block: dict) -> bool:
    if block["type"] in (LayoutType.HEADER.value, LayoutType.FOOTER.value):
        return True
    text = " ".join(s["text"] for s in block.get("content", []))
    if _BOILERPLATE_RE.search(text):
        return True
    # Never discard a block that opens with a question number, whatever else
    # it contains: the examiner column sometimes shares a block with the
    # marker, and dropping that doesn't just leave the label in, it deletes
    # the question boundary and folds the whole question into the previous
    # one. Keeping a stray label is much the lesser cost.
    if _leading_number(block) is not None:
        return False
    return _is_examiner_column(text)


def _is_examiner_column(text: str) -> bool:
    """True for the "For Examiner's Use" margin column, false for real content.

    Papers before ~2010 run that label down a narrow right-hand column beside
    every question, and it lands in reading order in the middle of the
    question's own text -- it was showing up in 14% of extracted questions.
    The column is almost always its own block, but not reliably enough to
    drop every block that mentions it: a handful genuinely mix the label in
    with question text, and dropping those loses the question.

    So the test is whether the block is *mostly* the label. What's left after
    removing the phrase is measured in letters, which ignores the rows of
    dotted answer lines and the mark-grid digits that otherwise look like
    content.
    """
    if not _EXAMINER_COLUMN_RE.search(text):
        return False
    remainder = _EXAMINER_COLUMN_RE.sub(" ", text)
    remainder = re.sub(r"['’]s|\bUse\b|\bTotal\b", " ", remainder)
    return len(re.findall(r"[A-Za-z]", remainder)) < _EXAMINER_REMAINDER_LETTERS


# Region types whose leading number may be a question marker. OTHER is in
# the list because it's each backend's catch-all -- DocStructBench's
# "abandon", PP-DocLayout's "number" -- and layout detection drops a real
# question marker into it often enough that excluding it cost 35 of 60
# sampled papers some of their questions, MCQ papers included (37 detected
# where the paper plainly has 40). The centred page numbers OTHER also
# carries are what `_detect_margin_x` filters out by position.
_MARKER_BEARING_TYPES = (LayoutType.TEXT.value, LayoutType.TITLE.value, LayoutType.OTHER.value)


def _leading_number(block: dict) -> int | None:
    """If a block's first (leftmost) span is a bare 1-2 digit number, treat it
    as a question marker -- this is how the question number and its stem end
    up in the same merged line block."""
    spans = block.get("content", [])
    if not spans or block["type"] not in _MARKER_BEARING_TYPES:
        return None
    first = min(spans, key=lambda s: s["bbox"]["x0"])
    m = _QUESTION_NUMBER_RE.fullmatch(first["text"].strip())
    return int(m.group(1)) if m else None


# Fraction of a cluster's markers that must equal their own page number
# before the cluster is judged to be page numbering rather than questions.
_PAGE_NUMBER_MATCH_FRAC = 0.6


def _detect_margin_x(candidates: list[tuple[int, dict, int]]) -> float:
    """The x0 that the question-number column sits at.

    Bare numbers elsewhere on the page form their own x0 clusters -- axis
    ticks, mark allocations, and above all the centred page number -- so the
    most populated cluster is not on its own a safe answer. A structured
    paper has about as many pages as it has questions, and when layout
    detection drops a couple of real markers into unclassified blocks (which
    `_leading_number` skips) the two clusters tie outright. Counter.most_common
    then breaks the tie on insertion order, handing the paper to whichever
    appears first -- normally the page numbers, since page 2 precedes the
    first question. The result is a paper whose questions are numbered by the
    pages they happen to start on.

    Page numbers give themselves away by equalling the page they sit on, so
    clusters that mostly do that are dropped before counting. Remaining ties
    go to the leftmost cluster: the question column is the leftmost thing on
    the page, and everything competing with it sits further right.
    """
    buckets: dict[int, list[tuple[int, int]]] = {}
    for page, block, num in candidates:
        x0 = round(min(block["content"], key=lambda s: s["bbox"]["x0"])["bbox"]["x0"])
        buckets.setdefault(x0, []).append((page, num))

    def is_page_numbering(entries: list[tuple[int, int]]) -> bool:
        # Needs a few entries to be worth judging: one question that happens
        # to start on its own page number proves nothing.
        if len(entries) < 3:
            return False
        return sum(1 for page, num in entries if page == num) >= len(entries) * _PAGE_NUMBER_MATCH_FRAC

    usable = {x0: e for x0, e in buckets.items() if not is_page_numbering(e)} or buckets
    return float(max(usable.items(), key=lambda kv: (len(kv[1]), -kv[0]))[0])


def parse_questions(pages: list[dict]) -> list[ParsedQuestion]:
    flat: list[tuple[int, dict]] = [
        (p["page"], b) for p in pages for b in p["blocks"] if not _is_boilerplate_block(b)
    ]
    flat.sort(key=lambda pb: (pb[0], pb[1]["bbox"]["y0"], pb[1]["bbox"]["x0"]))

    for i, (_, block) in enumerate(flat):
        if any(_COLOPHON_START_RE.search(s["text"]) for s in block.get("content", [])):
            flat = flat[:i]
            break

    candidates = [(page, block, num) for page, block in flat if (num := _leading_number(block)) is not None]
    if not candidates:
        return []
    margin_x = _detect_margin_x(candidates)
    markers = [
        (page, block, num)
        for page, block, num in candidates
        if abs(min(block["content"], key=lambda s: s["bbox"]["x0"])["bbox"]["x0"] - margin_x) <= 2.0
    ]

    # Numbers must strictly increase, but a single missed marker (layout
    # detection occasionally drops a question-number glyph into an
    # unclassified region instead of a text/title block -- see page-level
    # bugs this was written to guard against) shouldn't cost every question
    # after it: requiring an *exact* match on "expected" meant one gap
    # permanently broke the count, silently merging every remaining question
    # in the paper into the last one that matched. Requiring only "greater
    # than the last accepted number" tolerates the gap without opening the
    # door to reordering or duplicate numbers.
    accepted: list[tuple[int, dict, int]] = []
    last_accepted = 0
    for page, block, num in markers:
        if num > last_accepted:
            accepted.append((page, block, num))
            last_accepted = num
    if not accepted:
        return []

    # Shift each boundary up by a small tolerance: a block that's visually part
    # of a question's own marker row (e.g. a formula fragment carved out of
    # "...units of N s-1 m-1?") can still have a slightly smaller bbox y0 than
    # the marker block itself, since layout detection doesn't guarantee two
    # blocks on the same row agree on y0 to the point. Without this slack that
    # block sorts into the *previous* question instead.
    _BLOCK_BOUNDARY_TOLERANCE_PT = 5.0
    boundaries = [(page, block["bbox"]["y0"] - _BLOCK_BOUNDARY_TOLERANCE_PT) for page, block, _ in accepted]

    def sort_key(page: int, block: dict) -> tuple[int, float]:
        return (page, block["bbox"]["y0"])

    questions: list[ParsedQuestion] = []
    for idx, (start_page, _marker_block, number) in enumerate(accepted):
        start_key = boundaries[idx]
        end_key = boundaries[idx + 1] if idx + 1 < len(boundaries) else (float("inf"), float("inf"))

        members = [(page, block) for page, block in flat if start_key <= sort_key(page, block) < end_key]
        end_page = max((page for page, _ in members), default=start_page)

        # Flatten to individual spans (each keeps its own bbox from Phase 1,
        # not its owning block's) so reading order reflects real positions
        # even when a block was carved up by inline math or spans several
        # physical lines -- see _reading_order_spans.
        text_bearing_spans = _reading_order_spans(
            [
                {"page": page, "text": s["text"], "bbox": s["bbox"], "block_id": block["id"], "block_type": block["type"]}
                for page, block in members
                if block["type"] in {t.value for t in _TEXT_BEARING_TYPES}
                for s in block["content"]
            ]
        )
        text_parts = [s["text"].strip() for s in text_bearing_spans]

        # "lines": consecutive spans (in reading order) from the same
        # text/title block, joined back together -- this is what makes MCQ
        # option lines ("A data measured...") reliably separable later
        # without having to split on the letters A-D themselves (those
        # collide with unit symbols like A(mps), V(olts), N(ewtons) if done
        # on raw joined text).
        line_spans = [s for s in text_bearing_spans if s["block_type"] in (LayoutType.TEXT.value, LayoutType.TITLE.value)]
        lines: list[str] = []
        current_block_id, current_texts = None, []
        for s in line_spans:
            if s["block_id"] != current_block_id and current_texts:
                lines.append(" ".join(current_texts).strip())
                current_texts = []
            current_block_id = s["block_id"]
            current_texts.append(s["text"].strip())
        if current_texts:
            lines.append(" ".join(current_texts).strip())

        # Raw spans (not grouped into lines) for anything downstream that needs
        # actual span positions -- e.g. build_questions.py locates the bare
        # "A"/"B"/"C"/"D" option-label spans geometrically, which works
        # regardless of whether the paper laid options out one-per-line or
        # side-by-side in a row/grid (a single merged text block either way).
        # Includes FORMULA-block spans (unlike `lines` above): an option whose
        # entire content is math (e.g. "7 x 10^0 N") gets carved into its own
        # formula block by layout detection, and if its spans were dropped here
        # build_questions.py would find the "A"/"B" labels but nothing between
        # them -- an empty option.
        _span_block_types = {LayoutType.TEXT.value, LayoutType.TITLE.value, LayoutType.FORMULA.value}
        spans = [
            {"page": s["page"], "text": s["text"], "bbox": s["bbox"]}
            for s in text_bearing_spans
            if s["block_type"] in _span_block_types
        ]
        images = [RegionRef(id=b["id"], page=p, bbox=b["bbox"]) for p, b in members if b["type"] == LayoutType.IMAGE.value]
        tables = [RegionRef(id=b["id"], page=p, bbox=b["bbox"]) for p, b in members if b["type"] == LayoutType.TABLE.value]
        formulas = [RegionRef(id=b["id"], page=p, bbox=b["bbox"]) for p, b in members if b["type"] == LayoutType.FORMULA.value]

        questions.append(
            ParsedQuestion(
                number=number,
                page=start_page,
                end_page=end_page,
                text=" ".join(text_parts),
                lines=[line for line in lines if line],
                spans=spans,
                images=images,
                tables=tables,
                formulas=formulas,
            )
        )

    return questions


def run(merged_dir: Path, output_dir: Path) -> list[ParsedQuestion]:
    pages = read_json(merged_dir / "merged.json")
    questions = parse_questions(pages)
    write_json([q.to_dict() for q in questions], output_dir / "questions.json")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: group merged blocks into questions.")
    parser.add_argument("--merged", type=Path, default=Path("output/merged"), help="Phase 3 output directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output/questions"))
    args = parser.parse_args()

    questions = run(args.merged, args.output_dir)
    print(f"Parsed {len(questions)} questions -> {args.output_dir / 'questions.json'}")


if __name__ == "__main__":
    main()
