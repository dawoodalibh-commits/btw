"""Phase 8 — Build Final Question Object.

Joins Phase 4's parsed questions with Phase 5/6/7's formula/image/table
extraction results (matched purely by the region `id` each stage carries
forward from Phase 2) into one final object per question, plus pulls the
paper's own board code straight out of its running header/footer text
instead of guessing it from the filename.

Usage:
    python build_questions.py --extracted output/extracted --questions output/questions \
        --formulas output/formulas --images output/images --tables output/tables \
        --output-dir output/built
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from schemas import read_json, write_json

# The paper component is \d{1,2}, not \d{2}: Cambridge only moved to
# two-digit paper+variant numbers around 2009, and papers before that print
# "9702/1/M/J/02". Requiring two digits silently dropped every one of them to
# the --paper fallback, and since that fallback is one constant for a whole
# batch they then all collided on the papers table's UNIQUE(paper_code).
_PAPER_CODE_RE = re.compile(r"\b\d{4}/\d{1,2}/[A-Z]/[A-Z]/\d{2}\b")
_MARKS_RE = re.compile(r"\[(\d{1,2})\]")
_DEFAULT_MCQ_MARKS = 1  # used only when no "[n]" mark allocation appears anywhere in the question
_OPTION_LABELS = ["A", "B", "C", "D"]
# y0 within this range counts as "same visual row". Wide enough to chain a
# built-up fraction's numerator/label-baseline/denominator bands together (each
# individual gap in that stack runs ~6-8pt in practice) while staying well
# short of the ~12pt+ gap between genuinely separate lines of body text, so
# real distinct lines still don't get merged.
_ROW_TOLERANCE_PT = 9.0


def _detect_paper_code(extracted_dir: Path, fallback: str) -> str:
    for page in read_json(extracted_dir / "extraction.json"):
        for span in page["spans"]:
            m = _PAPER_CODE_RE.search(span["text"])
            if m:
                return m.group(0)
    return fallback


def _extract_marks(text: str) -> int:
    marks = [int(m) for m in _MARKS_RE.findall(text)]
    return sum(marks) if marks else _DEFAULT_MCQ_MARKS


def _reading_order(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups spans into visual rows (by y0, tolerant of small jitter) and
    orders each row left-to-right, then flattens rows top-to-bottom. This
    reads correctly whether a paper laid its 4 options out one per line, all
    in a single row, or as a 2x2 grid -- all three are common in this exam
    board's papers and a naive top-to-bottom/left-to-right global sort on
    slightly misaligned y0s can shuffle a grid's reading order."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s["page"], s["bbox"]["y0"]))
    rows: list[list[dict[str, Any]]] = [[ordered[0]]]
    for span in ordered[1:]:
        same_page = span["page"] == rows[-1][-1]["page"]
        if same_page and abs(span["bbox"]["y0"] - rows[-1][-1]["bbox"]["y0"]) <= _ROW_TOLERANCE_PT:
            rows[-1].append(span)
        else:
            rows.append([span])
    return [span for row in rows for span in sorted(row, key=lambda s: s["bbox"]["x0"])]


def _extract_options(
    spans: list[dict[str, Any]],
    formula_refs: list[dict[str, Any]],
    formulas_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    """Finds the bare "A"/"B"/"C"/"D" option-label spans geometrically (in
    reading order) rather than by splitting text, since splitting on those
    letters collides with unit symbols (A(mps), V(olts), N(ewtons)) that show
    up constantly in physics question text.

    An option whose entire content is math (e.g. "7 x 10^0 N") gets carved
    out into its own formula region by layout detection. Its resolved LaTeX
    is matched back in here by slotting a placeholder for each formula region
    into the same reading-order list as the text spans -- it lands in
    whichever [label, next_label) range its bbox geometrically falls into,
    same as any other span."""
    formula_items = [
        {
            "page": ref["page"],
            "bbox": ref["bbox"],
            "text": "",
            "_formula": {"latex": formulas_by_id[ref["id"]]["latex"], "bbox": ref["bbox"]},
        }
        for ref in formula_refs
        if ref["id"] in formulas_by_id
    ]
    ordered = _reading_order(spans + formula_items)
    label_positions = {}
    search_from = 0
    for label in _OPTION_LABELS:
        for i in range(search_from, len(ordered)):
            if ordered[i]["text"].strip() == label:
                label_positions[label] = i
                search_from = i + 1
                break
        else:
            return []  # labels must appear in strict A, B, C, D order to trust the split

    options = []
    for i, label in enumerate(_OPTION_LABELS):
        start = label_positions[label] + 1
        end = label_positions[_OPTION_LABELS[i + 1]] if i + 1 < len(_OPTION_LABELS) else len(ordered)
        segment = ordered[start:end]
        text = " ".join(s["text"].strip() for s in segment if s["text"].strip()).strip()
        option_formulas = [s["_formula"] for s in segment if "_formula" in s]
        options.append({"label": label, "text": text, "formulas": option_formulas})
    return options


def build_questions(
    extracted_dir: Path,
    questions_dir: Path,
    formulas_dir: Path,
    images_dir: Path,
    tables_dir: Path,
    question_images_dir: Path,
    output_dir: Path,
    paper_fallback: str,
) -> list[dict[str, Any]]:
    questions = read_json(questions_dir / "questions.json")
    formulas_by_id = {f["id"]: f for f in read_json(formulas_dir / "formulas.json")} if (formulas_dir / "formulas.json").exists() else {}
    images_by_id = {i["id"]: i for i in read_json(images_dir / "images.json")} if (images_dir / "images.json").exists() else {}
    tables_by_id = {t["id"]: t for t in read_json(tables_dir / "tables.json")} if (tables_dir / "tables.json").exists() else {}
    question_images_by_number = (
        {qi["question"]: qi["images"] for qi in read_json(question_images_dir / "question_images.json")}
        if (question_images_dir / "question_images.json").exists()
        else {}
    )

    paper = _detect_paper_code(extracted_dir, paper_fallback)

    built: list[dict[str, Any]] = []
    for q in questions:
        images = [
            {"file": images_by_id[ref["id"]]["file"], "bbox": ref["bbox"]}
            for ref in q["images"]
            if ref["id"] in images_by_id
        ]
        tables = [
            {"headers": tables_by_id[ref["id"]]["headers"], "rows": tables_by_id[ref["id"]]["rows"], "bbox": ref["bbox"]}
            for ref in q["tables"]
            if ref["id"] in tables_by_id
        ]
        formulas = [
            {"latex": formulas_by_id[ref["id"]]["latex"], "bbox": ref["bbox"]}
            for ref in q["formulas"]
            if ref["id"] in formulas_by_id
        ]
        # Full-page crop(s) of the whole question (Phase 7b) -- the question
        # exactly as it appeared on the page, not reassembled from the pieces
        # above. Usually one entry; two if the question spans a page break.
        question_images = question_images_by_number.get(q["question"], [])

        built.append(
            {
                "paper": paper,
                "page": q["page"],
                "question": q["question"],
                "marks": _extract_marks(q["text"]),
                "text": q["text"],
                "images": images,
                "tables": tables,
                "formulas": formulas,
                "question_images": question_images,
                "options": _extract_options(q["spans"], q["formulas"], formulas_by_id),
            }
        )

    write_json(built, output_dir / "built_questions.json")
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: merge all prior phases into final per-question objects.")
    parser.add_argument("--extracted", type=Path, default=Path("output/extracted"))
    parser.add_argument("--questions", type=Path, default=Path("output/questions"))
    parser.add_argument("--formulas", type=Path, default=Path("output/formulas"))
    parser.add_argument("--images", type=Path, default=Path("output/images"))
    parser.add_argument("--tables", type=Path, default=Path("output/tables"))
    parser.add_argument("--question-images", type=Path, default=Path("output/question_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/built"))
    parser.add_argument("--paper", default="unknown", help="Fallback paper code if none is found in the PDF's own header/footer text")
    args = parser.parse_args()

    built = build_questions(
        args.extracted, args.questions, args.formulas, args.images, args.tables, args.question_images, args.output_dir, args.paper
    )
    print(f"Built {len(built)} final question objects -> {args.output_dir / 'built_questions.json'}")


if __name__ == "__main__":
    main()
