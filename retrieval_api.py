"""Phase 11 — Search & Retrieval API.

Plain query functions over the Phase 10 SQLite database -- no web framework,
just functions a chatbot backend (or Phase 12's tutor) can import directly.
Every function returns fully hydrated question dicts (images/formulas/
tables/options/topics attached), matching the shape build_questions.py
produces, so callers don't need to know the schema underneath.

Usage:
    python retrieval_api.py --db output/questions.db topic Momentum
    python retrieval_api.py --db output/questions.db marks 1 2
    python retrieval_api.py --db output/questions.db ref 9702/12/O/N/25 7
    python retrieval_api.py --db output/questions.db search differentiation
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _row_to_question(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    qid = row["question_id"]
    paper = conn.execute("SELECT paper_code FROM papers WHERE paper_id = ?", (row["paper_id"],)).fetchone()

    def rows(sql: str) -> list[sqlite3.Row]:
        return conn.execute(sql, (qid,)).fetchall()

    images = [{"file": r["file"], "bbox": json.loads(r["bbox_json"])} for r in rows("SELECT * FROM images WHERE question_id = ?")]
    question_images = [
        {"page": r["page"], "file": r["file"], "bbox": json.loads(r["bbox_json"])}
        for r in rows("SELECT * FROM question_images WHERE question_id = ?")
    ]
    formulas = [{"latex": r["latex"], "bbox": json.loads(r["bbox_json"])} for r in rows("SELECT * FROM formulas WHERE question_id = ?")]
    tables = [
        {"headers": json.loads(r["headers_json"]), "rows": json.loads(r["rows_json"]), "bbox": json.loads(r["bbox_json"])}
        for r in rows("SELECT * FROM tables_ WHERE question_id = ?")
    ]
    options = [{"label": r["label"], "text": r["text"]} for r in rows("SELECT * FROM options WHERE question_id = ?")]
    topics = [
        r["name"]
        for r in conn.execute(
            "SELECT t.name FROM topics t JOIN question_topics qt ON qt.topic_id = t.topic_id WHERE qt.question_id = ?", (qid,)
        ).fetchall()
    ]

    return {
        "paper": paper["paper_code"],
        "page": row["page"],
        "question": row["question_number"],
        "marks": row["marks"],
        "text": row["text"],
        "images": images,
        "question_images": question_images,
        "tables": tables,
        "formulas": formulas,
        "options": options,
        "topics": topics,
    }


def _connect(db_path: Path) -> sqlite3.Connection:
    # Opened read-write-but-must-exist rather than plain connect(): sqlite
    # treats a missing path as "create an empty database here", so a typo in
    # --db silently yields a valid connection with no tables in it, and the
    # first query fails with "no such table: questions" pointing at the schema
    # instead of at the path.
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=rw", uri=True)
    except sqlite3.OperationalError as exc:
        raise sqlite3.OperationalError(f"Cannot open database {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    return conn


def _row_limit(limit: int | None) -> int:
    """SQLite reads a negative LIMIT as no limit, so callers passing None keep
    the original unbounded behaviour without a second query variant."""
    return -1 if limit is None else limit


def get_by_topic(db_path: Path, topic: str, limit: int | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT q.* FROM questions q
           JOIN question_topics qt ON qt.question_id = q.question_id
           JOIN topics t ON t.topic_id = qt.topic_id
           WHERE t.name = ? COLLATE NOCASE
           ORDER BY q.paper_id, q.question_number
           LIMIT ?""",
        (topic, _row_limit(limit)),
    ).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def get_by_marks_range(db_path: Path, min_marks: int, max_marks: int, limit: int | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM questions WHERE marks BETWEEN ? AND ? ORDER BY paper_id, question_number LIMIT ?",
        (min_marks, max_marks, _row_limit(limit)),
    ).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def get_by_reference(db_path: Path, paper_code: str, question_number: int) -> dict[str, Any] | None:
    conn = _connect(db_path)
    row = conn.execute(
        """SELECT q.* FROM questions q JOIN papers p ON p.paper_id = q.paper_id
           WHERE p.paper_code = ? AND q.question_number = ?""",
        (paper_code, question_number),
    ).fetchone()
    return _row_to_question(conn, row) if row else None


def get_random(
    db_path: Path,
    subject: str | None = None,
    variant: str | None = None,
    topic: str | None = None,
    min_marks: int | None = None,
    max_marks: int | None = None,
    count: int = 1,
) -> list[dict[str, Any]]:
    """Picks `count` random questions matching whichever filters are supplied.

    Built for the tutor's "give me a question" path, where fetching every
    match just to discard all but one is the dominant cost. ORDER BY RANDOM()
    scans the *filtered* set, so passing a subject or topic keeps it cheap;
    unfiltered over a full multi-subject corpus it's a whole-table scan.
    """
    conn = _connect(db_path)
    joins = ["JOIN papers p ON p.paper_id = q.paper_id"]
    where: list[str] = []
    params: list[Any] = []

    if subject is not None:
        where.append("p.subject = ?")
        params.append(subject)
    if variant is not None:
        where.append("p.variant = ?")
        params.append(variant)
    if topic is not None:
        joins.append("JOIN question_topics qt ON qt.question_id = q.question_id")
        joins.append("JOIN topics t ON t.topic_id = qt.topic_id")
        where.append("t.name = ? COLLATE NOCASE")
        params.append(topic)
    if min_marks is not None:
        where.append("q.marks >= ?")
        params.append(min_marks)
    if max_marks is not None:
        where.append("q.marks <= ?")
        params.append(max_marks)

    sql = f"SELECT q.* FROM questions q {' '.join(joins)}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY RANDOM() LIMIT ?"
    params.append(count)

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def search_text(db_path: Path, query: str, limit: int = 20) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT q.* FROM questions q
           JOIN questions_fts f ON f.rowid = q.question_id
           WHERE questions_fts MATCH ?
           ORDER BY rank LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11: query the question database.")
    parser.add_argument("--db", type=Path, default=Path("output/questions.db"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("topic", help='e.g. topic Momentum')
    p.add_argument("name")
    p.add_argument("--limit", type=int, default=None, help="Cap results (default: all)")

    p = sub.add_parser("marks", help="e.g. marks 1 2")
    p.add_argument("min_marks", type=int)
    p.add_argument("max_marks", type=int)
    p.add_argument("--limit", type=int, default=None, help="Cap results (default: all)")

    p = sub.add_parser("ref", help="e.g. ref 9702/12/O/N/25 7")
    p.add_argument("paper_code")
    p.add_argument("question_number", type=int)

    p = sub.add_parser("search", help='e.g. search differentiation')
    p.add_argument("query")

    p = sub.add_parser("random", help="e.g. random --subject 9702 --variant 12 --topic Momentum")
    p.add_argument("--subject")
    p.add_argument("--variant")
    p.add_argument("--topic")
    p.add_argument("--min-marks", type=int)
    p.add_argument("--max-marks", type=int)
    p.add_argument("--count", type=int, default=1)

    args = parser.parse_args()

    if args.command == "topic":
        results = get_by_topic(args.db, args.name, args.limit)
    elif args.command == "marks":
        results = get_by_marks_range(args.db, args.min_marks, args.max_marks, args.limit)
    elif args.command == "random":
        results = get_random(
            args.db,
            subject=args.subject,
            variant=args.variant,
            topic=args.topic,
            min_marks=args.min_marks,
            max_marks=args.max_marks,
            count=args.count,
        )
    elif args.command == "ref":
        result = get_by_reference(args.db, args.paper_code, args.question_number)
        results = [result] if result else []
    elif args.command == "search":
        results = search_text(args.db, args.query)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
