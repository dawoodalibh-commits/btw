#!/usr/bin/env python3
"""Merge output folders from separate pipeline runs into one database + asset root.

Running the corpus across two machines leaves two `--output-dir` trees that each
know about some of the papers. This joins them into a single tree the web UI can
serve.

Two things have to be merged, not one. The database stores an image only by its
*filename* (`images.file`), and webapp.py turns that back into a path as
<images-root>/<pdf-stem>/images/<file> -- so a database holding every paper is
still half-blank in the UI unless the per-paper asset folders end up under one
root as well. Both are handled here.

Questions are re-loaded from each paper's phase-9 JSON rather than by copying
database rows, so the merge reuses database.py's own upsert (papers deduped by
code, questions by (paper, number), child rows replaced not appended) instead of
reimplementing primary-key remapping across seven tables and an FTS index. A
source that only has its questions.db -- no JSON -- is read straight from that
database instead; pass either kind of path.

Precedence is argument order: when the same question appears in more than one
source, the LAST source on the command line wins.

Usage:
    ./merge_outputs.py output-machine-a output-machine-b --output-dir merged
    ./merge_outputs.py output-a output-b old-run.db --output-dir merged --copy
    python3 webapp.py --db merged/questions.db --images merged --question-images merged
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterator

from database import init_db, insert_questions

# Phase outputs worth carrying into the merged tree. The intermediate phases
# (extracted/layout/merged/questions/built/topics) are inputs to a re-run, not
# things the UI reads; --all-phases opts into them for a tree you can re-run
# phases against, at several times the disk.
_SERVED_SUBDIRS = ("images", "question_images")
_ALL_SUBDIRS = _SERVED_SUBDIRS + (
    "extracted", "layout", "merged", "questions", "formulas", "tables", "built", "topics",
)

_CLASSIFIED = Path("topics") / "classified_questions.json"


def _iter_paper_dirs(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (stem, paper_dir) for every paper in an output tree.

    Handles both layouts the pipeline produces: run_batch.py and folder-mode
    run_pipeline.sh file each paper under <root>/<pdf-stem>/, while single-PDF
    run_pipeline.sh writes the phase folders directly into <root>. The latter
    has no stem of its own, so the root's own name stands in for it.
    """
    if (root / _CLASSIFIED).is_file() or (root / "images").is_dir():
        yield root.name, root
        return
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / _CLASSIFIED).is_file() or (child / "images").is_dir():
            yield child.name, child


def _questions_from_json(paper_dir: Path) -> list[dict[str, Any]] | None:
    path = paper_dir / _CLASSIFIED
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _questions_from_db(db_path: Path) -> list[dict[str, Any]]:
    """Rebuild phase-9-shaped question dicts from an existing questions.db.

    Used for a source that arrived as a bare database. The option -> formula
    link (formulas.option_id) is reconstructed rather than flattened, so a
    merged database keeps the same per-option formula attribution the original
    load produced.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def loads(value: Any) -> Any:
        return json.loads(value) if value else None

    out: list[dict[str, Any]] = []
    rows = conn.execute(
        "SELECT q.*, p.paper_code FROM questions q JOIN papers p USING(paper_id) "
        "ORDER BY p.paper_code, q.question_number"
    ).fetchall()
    for row in rows:
        qid = row["question_id"]

        options: list[dict[str, Any]] = []
        by_option: dict[int, list[dict[str, Any]]] = {}
        for f in conn.execute(
            "SELECT option_id, latex, bbox_json FROM formulas WHERE question_id = ? AND option_id IS NOT NULL", (qid,)
        ):
            by_option.setdefault(f["option_id"], []).append(
                {"latex": f["latex"], "bbox": loads(f["bbox_json"])}
            )
        for o in conn.execute("SELECT option_id, label, text FROM options WHERE question_id = ?", (qid,)):
            options.append(
                {"label": o["label"], "text": o["text"], "formulas": by_option.get(o["option_id"], [])}
            )

        out.append(
            {
                "paper": row["paper_code"],
                "page": row["page"],
                "question": row["question_number"],
                "marks": row["marks"],
                "text": row["text"],
                "images": [
                    {"file": r["file"], "bbox": loads(r["bbox_json"])}
                    for r in conn.execute("SELECT file, bbox_json FROM images WHERE question_id = ?", (qid,))
                ],
                "question_images": [
                    {"page": r["page"], "file": r["file"], "bbox": loads(r["bbox_json"])}
                    for r in conn.execute(
                        "SELECT page, file, bbox_json FROM question_images WHERE question_id = ?", (qid,)
                    )
                ],
                "tables": [
                    {"headers": loads(r["headers_json"]), "rows": loads(r["rows_json"]), "bbox": loads(r["bbox_json"])}
                    for r in conn.execute(
                        "SELECT headers_json, rows_json, bbox_json FROM tables_ WHERE question_id = ?", (qid,)
                    )
                ],
                # Only the unattached formulas: the per-option ones are carried
                # on their option above, and database.py re-derives the rest.
                "formulas": [
                    {"latex": r["latex"], "bbox": loads(r["bbox_json"])}
                    for r in conn.execute(
                        "SELECT latex, bbox_json FROM formulas WHERE question_id = ? AND option_id IS NULL", (qid,)
                    )
                ],
                "options": options,
                "topics": [
                    r["name"]
                    for r in conn.execute(
                        "SELECT t.name FROM topics t JOIN question_topics qt ON qt.topic_id = t.topic_id "
                        "WHERE qt.question_id = ?",
                        (qid,),
                    )
                ],
            }
        )
    conn.close()
    return out


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Place one file, preferring a hardlink so a merged corpus of PNGs doesn't
    cost a second full copy of itself. Hardlinks can't cross filesystems (and
    don't exist on some of them), so a failure falls back to copying rather
    than aborting the merge."""
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _merge_assets(paper_dir: Path, target: Path, subdirs: tuple[str, ...], mode: str, overwrite: bool) -> tuple[int, int]:
    """Bring one paper's asset folders into the merged tree. Returns (placed, skipped)."""
    placed = skipped = 0
    for sub in subdirs:
        source = paper_dir / sub
        if not source.is_dir():
            continue
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            dest = target / sub / item.relative_to(source)
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()
            _link_or_copy(item, dest, mode)
            placed += 1
    return placed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two or more pipeline output folders into one database + asset tree.",
        epilog="Precedence is argument order: for a question present in several sources, the last one wins.",
    )
    parser.add_argument("sources", type=Path, nargs="+", help="Output folders and/or questions.db files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the merged tree")
    parser.add_argument("--db", type=Path, default=None, help="Merged database (default: <output-dir>/questions.db)")
    parser.add_argument(
        "--copy",
        action="store_const", const="copy", dest="mode", default="link",
        help="Copy asset files instead of hardlinking them (needed across filesystems; uses full disk)",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Also carry intermediate phase output, for a tree you can re-run later phases against",
    )
    parser.add_argument(
        "--overwrite-assets",
        action="store_true",
        help="Let a later source replace an asset file an earlier one already placed",
    )
    parser.add_argument("--no-assets", action="store_true", help="Merge the database only, leave files alone")
    args = parser.parse_args()

    for source in args.sources:
        if not source.exists():
            sys.exit(f"No such file or directory: {source}")

    target_root = args.output_dir
    db_path = args.db or target_root / "questions.db"
    if db_path.exists():
        # Re-loading into a populated database would take the upsert path, and
        # the FTS index is fed by an AFTER INSERT trigger only -- updated rows
        # would keep their old search text. Rebuilding from empty keeps the
        # index honest, and re-running the merge stays cheap either way.
        sys.exit(
            f"{db_path} already exists. Delete it first (or pass a different --db) -- "
            "this tool builds the merged database from scratch so its search index stays consistent."
        )
    target_root.mkdir(parents=True, exist_ok=True)
    subdirs = _ALL_SUBDIRS if args.all_phases else _SERVED_SUBDIRS

    conn = init_db(db_path)
    seen_questions: dict[tuple[str, int], str] = {}
    overridden = 0
    total_placed = total_skipped = 0

    for source in args.sources:
        label = str(source)
        if source.is_file():
            questions = _questions_from_db(source)
            insert_questions(conn, questions)
            for q in questions:
                key = (q["paper"], q["question"])
                if key in seen_questions:
                    overridden += 1
                seen_questions[key] = label
            print(f"=== {label}: {len(questions)} questions from database (no assets to merge) ===")
            continue

        papers = list(_iter_paper_dirs(source))
        if not papers:
            print(f"!!! {label}: no paper folders found, skipping", file=sys.stderr)
            continue

        loaded = placed = skipped = 0
        fallback = _questions_from_db(source / "questions.db") if (source / "questions.db").is_file() else None
        by_paper: dict[str, list[dict[str, Any]]] = {}
        if fallback:
            for q in fallback:
                by_paper.setdefault(q["paper"], []).append(q)

        for stem, paper_dir in papers:
            questions = _questions_from_json(paper_dir)
            if questions is None:
                # No phase-9 JSON for this paper; fall back to whatever the
                # source's own database recorded for it, matched by stem.
                questions = next(
                    (qs for code, qs in by_paper.items() if _stem_matches(stem, code)), []
                )
                if not questions:
                    print(f"    {stem}: no classified_questions.json and nothing in the database, skipping",
                          file=sys.stderr)
                    continue

            insert_questions(conn, questions)
            loaded += len(questions)
            for q in questions:
                key = (q["paper"], q["question"])
                if key in seen_questions and seen_questions[key] != label:
                    overridden += 1
                seen_questions[key] = label

            if not args.no_assets:
                p, s = _merge_assets(paper_dir, target_root / stem, subdirs, args.mode, args.overwrite_assets)
                placed += p
                skipped += s

        total_placed += placed
        total_skipped += skipped
        note = "" if args.no_assets else f", {placed} asset files placed, {skipped} already present"
        print(f"=== {label}: {len(papers)} papers, {loaded} questions{note} ===")

    # The FTS index is populated by an AFTER INSERT trigger, so it is already
    # correct for a from-scratch build. Rebuilding anyway costs a second on this
    # corpus and makes the result independent of how the rows happened to land.
    conn.execute("INSERT INTO questions_fts(questions_fts) VALUES('rebuild')")
    conn.commit()

    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    conn.close()

    print()
    print(f"=== Merged {n_questions} questions across {n_papers} papers -> {db_path} ===")
    if overridden:
        print(f"=== {overridden} questions were present in more than one source; the later source won ===")
    if not args.no_assets:
        print(f"=== {total_placed} asset files in {target_root} ({total_skipped} already present) ===")
    print()
    print("Serve it with:")
    print(f"  python3 webapp.py --db {db_path} --images {target_root} --question-images {target_root}")


def _stem_matches(stem: str, paper_code: str) -> bool:
    """True if a pdf stem (9702_w25_qp_12) and a board code (9702/12/O/N/25) name
    the same paper. Only needed on the database-fallback path, where the stem is
    all there is to match a paper folder against."""
    from schemas import paper_code_from_stem

    derived = paper_code_from_stem(stem)
    if derived == paper_code:
        return True
    # Pre-2009 papers print an unpadded number the filename pads, and vice
    # versa; compare on the numeric fields with padding removed.
    if not derived:
        return False
    d = derived.split("/")
    p = paper_code.split("/")
    return len(d) == len(p) == 5 and d[0] == p[0] and d[2:] == p[2:] and d[1].lstrip("0") == p[1].lstrip("0")


if __name__ == "__main__":
    main()
