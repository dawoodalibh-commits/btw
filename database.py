"""Phase 10 — Store in Database.

A normalized SQLite schema: one row per paper, one per question, and
separate child tables for images/formulas/tables/options/topics so a
question can have any number of each without duplicating question data.
Topics get their own table (rather than a flat column) specifically so
"all questions on topic X" is a plain join, not a text search.

Usage:
    python database.py --classified output/topics/classified_questions.json --db output/questions.db
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_code TEXT UNIQUE NOT NULL,
    subject TEXT,
    variant TEXT,
    session TEXT,
    year INTEGER
);

CREATE TABLE IF NOT EXISTS questions (
    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(paper_id),
    question_number INTEGER NOT NULL,
    page INTEGER,
    text TEXT,
    marks INTEGER,
    UNIQUE (paper_id, question_number)
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS question_topics (
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    topic_id INTEGER NOT NULL REFERENCES topics(topic_id),
    PRIMARY KEY (question_id, topic_id)
);

CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    file TEXT,
    bbox_json TEXT
);

-- Phase 7b's full-page crop(s) of the whole question, as opposed to `images`
-- above (isolated diagram/photo regions only). Usually one row per question,
-- two if it spans a page break -- hence `page` instead of assuming one.
CREATE TABLE IF NOT EXISTS question_images (
    question_image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    page INTEGER,
    file TEXT,
    bbox_json TEXT
);

CREATE TABLE IF NOT EXISTS formulas (
    formula_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    option_id INTEGER REFERENCES options(option_id),
    latex TEXT,
    bbox_json TEXT
);

CREATE TABLE IF NOT EXISTS tables_ (
    table_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    headers_json TEXT,
    rows_json TEXT,
    bbox_json TEXT
);

CREATE TABLE IF NOT EXISTS options (
    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(question_id),
    label TEXT,
    text TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
    text, content='questions', content_rowid='question_id'
);
CREATE TRIGGER IF NOT EXISTS questions_ai AFTER INSERT ON questions BEGIN
    INSERT INTO questions_fts(rowid, text) VALUES (new.question_id, new.text);
END;
"""

_PAPER_CODE_RE = re.compile(r"^(\d{4})/(\d+)/([A-Z])/([A-Z])/(\d{2})$")
_SESSION_NAMES = {"F/M": "February/March", "M/J": "May/June", "O/N": "October/November"}


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _parse_paper_code(paper_code: str) -> dict[str, Any]:
    m = _PAPER_CODE_RE.match(paper_code)
    if not m:
        return {"subject": None, "variant": None, "session": None, "year": None}
    subject, variant, m1, m2, yy = m.groups()
    return {
        "subject": subject,
        "variant": variant,
        "session": _SESSION_NAMES.get(f"{m1}/{m2}", f"{m1}/{m2}"),
        "year": 2000 + int(yy),
    }


def _get_or_create(conn: sqlite3.Connection, table: str, id_col: str, name_col: str, value: str) -> int:
    row = conn.execute(f"SELECT {id_col} FROM {table} WHERE {name_col} = ?", (value,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute(f"INSERT INTO {table} ({name_col}) VALUES (?)", (value,))
    return cur.lastrowid


def _get_or_create_paper(conn: sqlite3.Connection, paper_code: str) -> int:
    row = conn.execute("SELECT paper_id FROM papers WHERE paper_code = ?", (paper_code,)).fetchone()
    if row:
        return row[0]
    meta = _parse_paper_code(paper_code)
    cur = conn.execute(
        "INSERT INTO papers (paper_code, subject, variant, session, year) VALUES (?, ?, ?, ?, ?)",
        (paper_code, meta["subject"], meta["variant"], meta["session"], meta["year"]),
    )
    return cur.lastrowid


def insert_questions(conn: sqlite3.Connection, questions: list[dict[str, Any]]) -> None:
    for q in questions:
        paper_id = _get_or_create_paper(conn, q["paper"])
        cur = conn.execute(
            """INSERT INTO questions (paper_id, question_number, page, text, marks) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (paper_id, question_number) DO UPDATE SET
                   page=excluded.page, text=excluded.text, marks=excluded.marks
               RETURNING question_id""",
            (paper_id, q["question"], q["page"], q["text"], q["marks"]),
        )
        question_id = cur.fetchone()[0]

        # Clear old child rows so re-running an insert doesn't duplicate them.
        for table in ("images", "question_images", "formulas", "tables_", "options", "question_topics"):
            conn.execute(f"DELETE FROM {table} WHERE question_id = ?", (question_id,))

        for img in q.get("images", []):
            conn.execute(
                "INSERT INTO images (question_id, file, bbox_json) VALUES (?, ?, ?)",
                (question_id, img.get("file"), json.dumps(img.get("bbox"))),
            )
        for qimg in q.get("question_images", []):
            conn.execute(
                "INSERT INTO question_images (question_id, page, file, bbox_json) VALUES (?, ?, ?, ?)",
                (question_id, qimg.get("page"), qimg.get("file"), json.dumps(qimg.get("bbox"))),
            )
        for t in q.get("tables", []):
            conn.execute(
                "INSERT INTO tables_ (question_id, headers_json, rows_json, bbox_json) VALUES (?, ?, ?, ?)",
                (question_id, json.dumps(t.get("headers")), json.dumps(t.get("rows")), json.dumps(t.get("bbox"))),
            )

        # Options are inserted before formulas so each option's own matched
        # formulas (see build_questions.py's per-option formula matching) can
        # reference the right option_id, instead of every formula on the
        # question being linked to every option.
        matched_bboxes: set[str] = set()
        for opt in q.get("options", []):
            cur = conn.execute(
                "INSERT INTO options (question_id, label, text) VALUES (?, ?, ?)",
                (question_id, opt.get("label"), opt.get("text")),
            )
            option_id = cur.lastrowid
            for f in opt.get("formulas", []):
                bbox_json = json.dumps(f.get("bbox"), sort_keys=True)
                matched_bboxes.add(bbox_json)
                conn.execute(
                    "INSERT INTO formulas (question_id, option_id, latex, bbox_json) VALUES (?, ?, ?, ?)",
                    (question_id, option_id, f.get("latex"), bbox_json),
                )
        for f in q.get("formulas", []):
            bbox_json = json.dumps(f.get("bbox"), sort_keys=True)
            if bbox_json in matched_bboxes:
                continue  # already inserted above, linked to its option
            conn.execute(
                "INSERT INTO formulas (question_id, option_id, latex, bbox_json) VALUES (?, ?, ?, ?)",
                (question_id, None, f.get("latex"), bbox_json),
            )
        for topic in q.get("topics", []):
            topic_id = _get_or_create(conn, "topics", "topic_id", "name", topic)
            conn.execute(
                "INSERT OR IGNORE INTO question_topics (question_id, topic_id) VALUES (?, ?)",
                (question_id, topic_id),
            )

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10: load classified questions into a SQLite database.")
    parser.add_argument("--classified", type=Path, default=Path("output/topics/classified_questions.json"))
    parser.add_argument("--db", type=Path, default=Path("output/questions.db"))
    args = parser.parse_args()

    with open(args.classified) as f:
        questions = json.load(f)

    conn = init_db(args.db)
    insert_questions(conn, questions)
    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    conn.close()
    print(f"Loaded {n_questions} questions across {n_papers} paper(s) -> {args.db}")


if __name__ == "__main__":
    main()
