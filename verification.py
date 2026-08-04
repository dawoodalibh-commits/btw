#!/usr/bin/env python3
"""Post-load verification — did every paper actually land in the database?

Phase 10 loads papers one at a time and a paper that failed an earlier phase
is dropped silently from the rest of the batch (see run_batch.py's `drop`),
so a database that opens fine and answers queries can still be missing a
couple of dozen papers without anything having looked wrong at the time.
This compares the papers *expected* (one output folder per paper, or a folder
of PDFs) against what's in `papers`/`questions`, and reports the gaps.

Three wrinkles make this more than a set difference:

  * Paper codes need normalizing before comparison. Phase 9 prefers the code
    printed inside the paper over the one derived from the filename, and the
    printed one zero-pads the variant -- `9702_s03_qp_1` is stored as
    9702/01/M/J/03, not the 9702/1/M/J/03 that paper_code_from_stem builds.
  * Sometimes the printed variant isn't the filename's variant at all: the
    2009 papers filed as `qp_2`/`qp_4`/`qp_5` print 21/41/51 inside. Those
    would otherwise show up twice, once as a missing paper and again as an
    unexpected one, so a missing paper is checked against the unexpected
    rows for the same sitting before being called missing.
  * `papers.paper_code` is UNIQUE, so papers whose code couldn't be
    determined at all collapse into a single "unknown" row and overwrite each
    other's questions. That row is one paper by the schema's reckoning and
    an unknown number of lost papers in reality, so it's reported separately
    rather than counted as a hit.

"Has too few questions" is deliberately not a flat threshold. Only the
multiple-choice papers (variant 1x) have a known length -- 40 -- and a paper
5 with three questions is complete, not broken, so a single --min-questions
would either miss real MCQ truncation or bury it under hundreds of healthy
structured papers.

Usage:
    ./verification.py                                    # output/questions.db vs output/*/
    ./verification.py --db output-v2.db --papers output
    ./verification.py --papers papers/ --db output/questions.db
    ./verification.py --strict                           # warnings become failures
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from schemas import _PDF_STEM_RE, paper_code_from_stem

# Zero-pads the variant so filename-derived and paper-printed codes compare
# equal. Deliberately not reusing database.py's _PAPER_CODE_RE: that one is
# for parsing metadata out of a code, this one is for canonicalizing it.
_CODE_RE = re.compile(r"^(\d{4})/(\d+)/([A-Z])/([A-Z])/(\d{2})$")

# Directory every completed paper has, used to tell "never processed" apart
# from "processed but never loaded" when a paper turns up missing.
_FINAL_PHASE_OUTPUT = Path("topics") / "classified_questions.json"


def normalize_code(code: str) -> str | None:
    m = _CODE_RE.match(code.strip().upper())
    if not m:
        return None
    subject, variant, first, second, yy = m.groups()
    return f"{subject}/{int(variant):02d}/{first}/{second}/{yy}"


def expected_stems(papers_root: Path) -> list[str]:
    """The papers that should be in the database, newest-run layout first.

    Accepts either an output tree (one subfolder per paper, named for the PDF
    stem) or the folder of source PDFs. Both name things by stem, so the two
    cases differ only in what's being listed.
    """
    if not papers_root.exists():
        sys.exit(f"No such directory: {papers_root}")
    pdfs = sorted(p.stem for p in papers_root.rglob("*.pdf"))
    if pdfs:
        return pdfs
    return sorted(d.name for d in papers_root.iterdir() if d.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every paper made it into the database.")
    parser.add_argument("--db", type=Path, default=Path("output/questions.db"), help="SQLite database to check.")
    parser.add_argument(
        "--papers",
        type=Path,
        default=None,
        help="Output tree (one folder per paper) or folder of PDFs. Defaults to the database's own directory.",
    )
    parser.add_argument(
        "--mcq-questions",
        type=int,
        default=40,
        help="Questions a multiple-choice paper (variant 1x) should have; fewer is a warning. 0 disables.",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures too.")
    parser.add_argument("--list-all", action="store_true", help="Print every offending paper, not just the first 20.")
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"No such database: {args.db}")
    papers_root = args.papers or args.db.parent

    stems = expected_stems(papers_root)
    if not stems:
        sys.exit(f"Found no papers under {papers_root} (expected PDFs or one folder per paper).")

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT p.paper_code, p.paper_id, COUNT(q.question_id) AS n_questions
           FROM papers p LEFT JOIN questions q ON q.paper_id = p.paper_id
           GROUP BY p.paper_id"""
    ).fetchall()

    # Papers keyed by normalized code. Anything whose code doesn't parse
    # (notably the "unknown" collision bucket) can't be matched to a stem and
    # is held aside instead.
    by_code: dict[str, sqlite3.Row] = {}
    unparseable: list[sqlite3.Row] = []
    # Two rows can normalize to one code if the same paper was loaded once
    # with a padded variant and once without (both spellings occur). Keeping
    # only the last would quietly under-report, so they're collected and
    # reported instead.
    duplicates: dict[str, list[str]] = {}
    for row in rows:
        code = normalize_code(row["paper_code"])
        if code is None:
            unparseable.append(row)
            continue
        if code in by_code:
            duplicates.setdefault(code, [by_code[code]["paper_code"]]).append(row["paper_code"])
            if row["n_questions"] <= by_code[code]["n_questions"]:
                continue
        by_code[code] = row

    missing: list[tuple[str, str]] = []       # (stem, why)
    empty: list[tuple[str, str]] = []         # (stem, code)
    mcq_short: list[tuple[str, str, int]] = []  # (stem, code, n)
    unmappable_stems: list[str] = []          # filename doesn't fit the convention
    matched_codes: set[str] = set()
    unmatched: list[str] = []                 # stems with no row under their own code

    for stem in stems:
        raw = paper_code_from_stem(stem)
        if raw is None:
            unmappable_stems.append(stem)
            continue
        code = normalize_code(raw)
        if code is None or code not in by_code:
            unmatched.append(stem)
            continue
        matched_codes.add(code)

    # A stem that found nothing may still be in there under a variant the
    # paper printed on itself rather than the one in its filename. Only
    # single-digit filename variants are reconciled this way, and only against
    # a row from the same sitting that nothing else claimed -- broad enough
    # for the 2009 qp_2 -> 21 case, narrow enough not to invent matches.
    orphan_codes = sorted(set(by_code) - matched_codes)
    reconciled: list[tuple[str, str]] = []  # (stem, code it was actually stored under)
    for stem in unmatched:
        code = normalize_code(paper_code_from_stem(stem) or "")
        if code is None:
            missing.append((stem, "unparseable filename"))
            continue
        subject, variant, first, second, yy = _CODE_RE.match(code).groups()
        filename_variant = _PDF_STEM_RE.match(stem).group(4)
        candidates = [
            c for c in orphan_codes
            if len(filename_variant) == 1
            and (m := _CODE_RE.match(c))
            and (m.group(1), m.group(3), m.group(4), m.group(5)) == (subject, first, second, yy)
            and m.group(2).startswith(filename_variant)
        ]
        if len(candidates) == 1:
            reconciled.append((stem, candidates[0]))
            matched_codes.add(candidates[0])
            orphan_codes.remove(candidates[0])
            continue
        processed = (papers_root / stem / _FINAL_PHASE_OUTPUT).exists()
        missing.append((stem, "reached phase 9 but never loaded" if processed else "no phase 9 output"))

    # Every paper that did land somewhere, checked for content.
    for code in sorted(matched_codes):
        row = by_code[code]
        n = row["n_questions"]
        if n == 0:
            empty.append((code, row["paper_code"]))
        elif args.mcq_questions and _CODE_RE.match(code).group(2).startswith("1") and n < args.mcq_questions:
            mcq_short.append((code, row["paper_code"], n))

    # Rows in the database that no expected paper claims. Usually a leftover
    # from an earlier run against a different set of PDFs.
    orphans = orphan_codes

    # Questions that loaded but carry no text -- they'd return empty from
    # search and give the tutor nothing to work with.
    textless = conn.execute(
        """SELECT p.paper_code, COUNT(*) AS n FROM questions q JOIN papers p ON p.paper_id = q.paper_id
           WHERE q.text IS NULL OR TRIM(q.text) = '' GROUP BY p.paper_id ORDER BY n DESC"""
    ).fetchall()
    total_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    conn.close()

    def show(items: list, render) -> None:
        limit = len(items) if args.list_all else 20
        for item in items[:limit]:
            print(f"    {render(item)}")
        if len(items) > limit:
            print(f"    ... and {len(items) - limit} more (--list-all to see them)")

    print(f"database : {args.db}")
    print(f"papers   : {papers_root} ({len(stems)} expected)")
    print(f"stored   : {len(rows)} paper rows, {total_questions} questions\n")

    failures = 0
    warnings = 0

    if missing:
        failures += len(missing)
        print(f"FAIL  {len(missing)} expected paper(s) not in the database:")
        show(missing, lambda m: f"{m[0]}  ({m[1]})")
    if empty:
        failures += len(empty)
        print(f"FAIL  {len(empty)} paper(s) stored with zero questions:")
        show(empty, lambda e: f"{e[1]}")
    if duplicates:
        failures += len(duplicates)
        print(f"FAIL  {len(duplicates)} paper(s) stored under more than one spelling of the same code:")
        show(sorted(duplicates.items()), lambda d: f"{d[0]}  as {', '.join(d[1])}")
    if unparseable:
        failures += len(unparseable)
        print(f"FAIL  {len(unparseable)} paper row(s) with an unusable paper_code:")
        print("      papers.paper_code is UNIQUE, so every paper that landed here")
        print("      overwrote the previous one's questions.")
        show(unparseable, lambda r: f"{r['paper_code']!r}  ({r['n_questions']} questions)")

    if mcq_short:
        warnings += len(mcq_short)
        print(f"WARN  {len(mcq_short)} multiple-choice paper(s) with fewer than {args.mcq_questions} questions:")
        show(mcq_short, lambda t: f"{t[1]}  {t[2]}/{args.mcq_questions} question(s)")
    if textless:
        warnings += len(textless)
        print(f"WARN  {len(textless)} paper(s) contain questions with empty text:")
        show(list(textless), lambda r: f"{r['paper_code']}  {r['n']} question(s)")
    if orphans:
        warnings += len(orphans)
        print(f"WARN  {len(orphans)} paper(s) in the database that weren't expected:")
        show(orphans, lambda c: c)
    if unmappable_stems:
        warnings += len(unmappable_stems)
        print(f"WARN  {len(unmappable_stems)} filename(s) don't fit <subject>_<season><yy>_qp_<paper>, not checked:")
        show(unmappable_stems, lambda s: s)

    if reconciled:
        print(f"NOTE  {len(reconciled)} paper(s) stored under a variant other than their filename's:")
        show(reconciled, lambda r: f"{r[0]}  stored as {r[1]}")

    if args.strict:
        failures += warnings
        warnings = 0

    ok = len(stems) - len(missing) - len(empty) - len(unmappable_stems)
    print(f"\n{ok}/{len(stems)} papers verified, {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
