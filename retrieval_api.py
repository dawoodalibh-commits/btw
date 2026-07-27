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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_by_topic(db_path: Path, topic: str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT q.* FROM questions q
           JOIN question_topics qt ON qt.question_id = q.question_id
           JOIN topics t ON t.topic_id = qt.topic_id
           WHERE t.name = ? COLLATE NOCASE
           ORDER BY q.paper_id, q.question_number""",
        (topic,),
    ).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def get_by_marks_range(db_path: Path, min_marks: int, max_marks: int) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM questions WHERE marks BETWEEN ? AND ? ORDER BY paper_id, question_number",
        (min_marks, max_marks),
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

    p = sub.add_parser("marks", help="e.g. marks 1 2")
    p.add_argument("min_marks", type=int)
    p.add_argument("max_marks", type=int)

    p = sub.add_parser("ref", help="e.g. ref 9702/12/O/N/25 7")
    p.add_argument("paper_code")
    p.add_argument("question_number", type=int)

    p = sub.add_parser("search", help='e.g. search differentiation')
    p.add_argument("query")

    args = parser.parse_args()

    if args.command == "topic":
        results = get_by_topic(args.db, args.name)
    elif args.command == "marks":
        results = get_by_marks_range(args.db, args.min_marks, args.max_marks)
    elif args.command == "ref":
        result = get_by_reference(args.db, args.paper_code, args.question_number)
        results = [result] if result else []
    elif args.command == "search":
        results = search_text(args.db, args.query)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
