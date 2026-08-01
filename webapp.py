"""Phase 13 — Web UI.

A minimal Flask front end over the Phase 10 database (via retrieval_api) and
Phase 12 tutor (explain/grade/chat). One file, inline templates, no build
step, no JS framework -- a search page, a question page (explain/grade
panels), and a free-form ChatGPT-style chat page.

Usage:
    python webapp.py --db output/questions.db --images output/images \
        --question-images output/question_images --port 5000

Note on images: papers processed via run_pipeline.sh's batch mode each get
their own output/<pdf-stem>/images/ (and .../question_images/) folder, not
the shared --images / --question-images dir. _guess_asset_dirs() below
reconstructs that per-paper folder name from the paper code as a best-effort
fallback; if neither location has the file, the image just won't render.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from retrieval_api import _connect, _row_to_question, get_by_marks_range, get_by_reference, get_by_topic, search_text
from tutor import _DEFAULT_MODELS, _question_context_text, chat_reply, explain_question, grade_answer

# Lightweight reference detection for the free-form chat page: chat_reply()
# is deliberately not tied to any question, so "explain Q7 in 9702/12/O/N/25"
# gets nothing to work with unless we spot the reference ourselves and splice
# in that question's actual content before calling the model. Simple keyword
# heuristic, not real NLU -- covers "question 7" / "q7" / "q 7" in either
# order relative to the paper code.
_PAPER_CODE_IN_TEXT_RE = re.compile(r"\b(\d{4}/\d{1,2}/[A-Za-z]/[A-Za-z]/\d{2})\b")
_QUESTION_NUM_IN_TEXT_RE = re.compile(r"\b(?:question|q)\.?\s*#?\s*(\d{1,2})\b", re.IGNORECASE)


def _detect_question_reference(text: str) -> tuple[str, int] | None:
    paper_match = _PAPER_CODE_IN_TEXT_RE.search(text)
    question_match = _QUESTION_NUM_IN_TEXT_RE.search(text)
    if paper_match and question_match:
        return paper_match.group(1).upper(), int(question_match.group(1))
    return None

app = Flask(__name__)

_PAPER_CODE_RE = re.compile(r"^(\d{4})/(\d+)/([A-Z])/([A-Z])/(\d{2})$")
_SEASON_CODE = {"F/M": "m", "M/J": "s", "O/N": "w"}


def _guess_asset_dirs(root: Path, paper_code: str, subfolder: str) -> list[Path]:
    """Where this paper's assets might live, best guess first.

    Batch runs file assets per paper under <batch-root>/<pdf-stem>/<subfolder>,
    which the paper's own code can be turned back into. The awkward part is
    that --images may point at either level: at the batch root ("output"), or
    at a single paper's own folder ("output/images"), which is what the
    single-PDF pipeline produces and what the flag's default still assumes.
    Both are tried rather than making the caller know which convention the run
    used -- guessing wrong just means blank image panels with nothing in the
    log to say why.
    """
    dirs = [root]
    m = _PAPER_CODE_RE.match(paper_code)
    if m:
        subject, variant, m1, m2, yy = m.groups()
        season = _SEASON_CODE.get(f"{m1}/{m2}")
        if season:
            # Every spelling of the paper number this code might have been
            # filed under, because the code a paper prints and the name its
            # file arrived with disagree in two different eras:
            #   * older papers print it zero-padded ("9702/05/O/N/03") while
            #     the file is named unpadded (9702_w03_qp_5), and
            #   * Cambridge split papers into variants around 2009 (paper 2
            #     became 21/22/23), but downloads from that year are still
            #     commonly filed under the pre-split number, so a paper
            #     printing "9702/21/O/N/09" lives in 9702_w09_qp_2.
            # Extra candidates cost a stat() and nothing else; a missing one
            # costs a silently blank image panel.
            variants = [variant]
            unpadded = variant.lstrip("0")
            if unpadded and unpadded not in variants:
                variants.append(unpadded)
            if len(variant) == 2 and not variant.startswith("0") and variant[0] not in variants:
                variants.append(variant[0])
            for base in (root, root.parent):
                for number in variants:
                    candidate = base / f"{subject}_{season}{yy}_qp_{number}" / subfolder
                    if candidate not in dirs:
                        dirs.append(candidate)
    return dirs


def _list_questions(db_path: Path, limit: int = 30) -> list[dict]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM questions ORDER BY paper_id, question_number LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_question(conn, r) for r in rows]


def _topics(db_path: Path) -> list[str]:
    conn = _connect(db_path)
    return [r["name"] for r in conn.execute("SELECT name FROM topics ORDER BY name").fetchall()]


# --------------------------------------------------------------------------
# Design system
#
# One token block + one stylesheet shared by all three pages, so the search
# page, the question page and the chat page look like the same product.
# Light only, white canvas; the accent is a blueprint blue and the utility
# face is monospace, borrowed from the exam-paper vernacular (paper codes,
# mark totals, marking points).
# --------------------------------------------------------------------------
STYLES = r"""
<style>
  :root {
    color-scheme: light;
    --bg: #ffffff;
    --surface: #f7f8fa;
    --surface-hover: #eff2f6;
    --border: #e4e7ec;
    --border-strong: #d3d9e1;
    --text: #14181d;
    --text-muted: #5f6b78;
    --accent: #1b4f8f;
    --accent-hover: #163f73;
    --accent-soft: #eef3fa;
    --good: #12734a;
    --good-soft: #e8f5ee;
    --bad: #b42318;
    --bad-soft: #fdeceb;
    --radius-sm: 8px;
    --radius: 12px;
    --radius-lg: 18px;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --shadow-sm: 0 1px 2px rgba(16, 24, 40, .04), 0 1px 3px rgba(16, 24, 40, .06);
    --shadow-md: 0 4px 16px rgba(16, 24, 40, .08);
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }

  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  :focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
  }

  /* ---- top bar ---------------------------------------------------- */
  .topbar {
    position: sticky;
    top: 0;
    z-index: 20;
    background: rgba(255, 255, 255, .88);
    backdrop-filter: saturate(180%) blur(12px);
    border-bottom: 1px solid var(--border);
  }
  .topbar-inner {
    max-width: 860px;
    margin: 0 auto;
    padding: .7rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -.01em;
  }
  .brand:hover { text-decoration: none; }
  .brand-mark {
    width: 26px;
    height: 26px;
    display: grid;
    place-items: center;
    border-radius: 7px;
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-size: .82rem;
    line-height: 1;
  }
  .nav { margin-left: auto; display: flex; align-items: center; gap: .35rem; }
  .nav-link {
    color: var(--text-muted);
    font-size: .9rem;
    padding: .35rem .7rem;
    border-radius: var(--radius-sm);
  }
  .nav-link:hover { background: var(--surface-hover); color: var(--text); text-decoration: none; }

  /* ---- page ------------------------------------------------------- */
  .page { max-width: 860px; margin: 0 auto; padding: 2rem 1.25rem 5rem; }
  h1.title { font-size: 1.4rem; font-weight: 650; letter-spacing: -.02em; margin: 0 0 .35rem; }
  .subtitle { color: var(--text-muted); font-size: .92rem; margin: 0 0 1.75rem; }
  .backlink { display: inline-block; color: var(--text-muted); font-size: .88rem; margin-bottom: 1.1rem; }
  .backlink:hover { color: var(--text); text-decoration: none; }

  /* ---- controls --------------------------------------------------- */
  input[type=text], select, textarea {
    font: inherit;
    color: inherit;
    background: var(--bg);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: .5rem .7rem;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type=text]:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  input::placeholder, textarea::placeholder { color: #9aa4b0; }
  select { cursor: pointer; }
  textarea { width: 100%; resize: vertical; min-height: 5.5rem; line-height: 1.5; }

  .btn {
    font: inherit;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    padding: .5rem .95rem;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-strong);
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
    transition: background .15s, border-color .15s;
  }
  .btn:hover { background: var(--surface-hover); }
  .btn:disabled { opacity: .5; cursor: default; }
  .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn-primary:hover { background: var(--accent-hover); }

  .searchbar { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .searchfield { position: relative; flex: 1; min-width: 220px; }
  .searchfield input { width: 100%; padding-left: 2.2rem; }
  .searchfield svg { position: absolute; left: .7rem; top: 50%; transform: translateY(-50%); color: #98a2b0; }

  .toolbar-note { color: var(--text-muted); font-size: .85rem; margin: 0 0 .9rem; }

  /* ---- cards ------------------------------------------------------ */
  .card {
    display: block;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .95rem 1.1rem;
    margin-bottom: .7rem;
    background: var(--bg);
    color: inherit;
    transition: border-color .15s, box-shadow .15s, transform .15s;
  }
  a.card:hover {
    text-decoration: none;
    border-color: var(--border-strong);
    box-shadow: var(--shadow-sm);
  }
  .card-meta { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; margin-bottom: .5rem; }
  .card-text { color: var(--text); }
  .card-topics { color: var(--text-muted); font-size: .82rem; }

  .chip {
    font-family: var(--mono);
    font-size: .74rem;
    letter-spacing: .04em;
    padding: .17rem .45rem;
    border-radius: 5px;
    background: var(--accent-soft);
    color: var(--accent);
    white-space: nowrap;
  }
  .chip-plain { background: var(--surface); color: var(--text-muted); }
  .dot { color: var(--border-strong); }

  .empty {
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius);
    padding: 2.5rem 1.5rem;
    text-align: center;
    color: var(--text-muted);
  }

  /* ---- question detail -------------------------------------------- */
  .q-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .8rem; }
  .q-body { font-size: 1rem; }
  .paper-card {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    background: var(--bg);
    box-shadow: var(--shadow-sm);
  }
  img.q-image { max-width: 100%; display: block; margin: .9rem 0; border-radius: var(--radius-sm); border: 1px solid var(--border); }
  .options { list-style: none; padding: 0; margin: 1rem 0 0; }
  .options li { display: flex; gap: .65rem; padding: .35rem 0; }
  .options .label {
    flex: none;
    width: 1.55rem; height: 1.55rem;
    display: grid; place-items: center;
    border: 1px solid var(--border-strong);
    border-radius: 50%;
    font-family: var(--mono);
    font-size: .78rem;
  }
  .formula { display: block; margin: .6rem 0; overflow-x: auto; }
  table.data { border-collapse: collapse; margin: 1rem 0 0; font-size: .92rem; }
  table.data th, table.data td { border: 1px solid var(--border); padding: .4rem .7rem; text-align: left; }
  table.data th { background: var(--surface); font-weight: 600; }

  /* ---- panels ----------------------------------------------------- */
  .panel {
    margin-top: 1.75rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .panel-head {
    display: flex;
    align-items: baseline;
    gap: .6rem;
    padding: .8rem 1.1rem;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .panel-head h2 { font-size: .95rem; font-weight: 600; margin: 0; }
  .panel-head .hint { color: var(--text-muted); font-size: .82rem; }
  .panel-body { padding: 1.1rem; }
  .field-label { display: block; font-size: .85rem; font-weight: 500; margin: 0 0 .35rem; }
  .field + .field { margin-top: .9rem; }
  .controls { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; margin-top: 1rem; }
  .controls .model { width: 12rem; }

  .output { margin-top: 1.1rem; }
  .output:empty { margin-top: 0; }
  .status { display: flex; align-items: center; gap: .55rem; color: var(--text-muted); font-size: .9rem; }
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid var(--border-strong);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .alert {
    border: 1px solid #f3c9c5;
    background: var(--bad-soft);
    color: var(--bad);
    border-radius: var(--radius-sm);
    padding: .65rem .8rem;
    font-size: .9rem;
  }

  /* ---- grading ---------------------------------------------------- */
  .score { display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem; }
  .score-value { font-family: var(--mono); font-size: 1.3rem; font-weight: 600; letter-spacing: -.02em; }
  .score-track { flex: 1; height: 6px; background: var(--surface-hover); border-radius: 999px; overflow: hidden; }
  .score-fill { height: 100%; background: var(--good); border-radius: 999px; transition: width .4s ease; }
  .point { display: flex; gap: .6rem; padding: .55rem 0; border-top: 1px solid var(--border); }
  .point:first-of-type { border-top: none; }
  .badge {
    flex: none;
    width: 1.3rem; height: 1.3rem;
    margin-top: .12rem;
    display: grid; place-items: center;
    border-radius: 50%;
    font-size: .72rem;
    font-weight: 700;
  }
  .point.earned .badge { background: var(--good-soft); color: var(--good); }
  .point.missed .badge { background: var(--bad-soft); color: var(--bad); }
  .point-text { font-size: .93rem; }
  .point-text .pt { font-weight: 500; }
  .point-text .why { color: var(--text-muted); }
  .feedback {
    margin-top: 1rem;
    padding: .85rem 1rem;
    background: var(--surface);
    border-radius: var(--radius-sm);
    font-size: .93rem;
  }

  /* ---- rendered markdown (LLM output) ----------------------------- */
  .rendered-md > :first-child { margin-top: 0; }
  .rendered-md > :last-child { margin-bottom: 0; }
  .rendered-md :is(h1, h2, h3, h4, h5, h6) { margin: 1.1rem 0 .45rem; line-height: 1.3; font-weight: 600; }
  .rendered-md h1 { font-size: 1.15rem; }
  .rendered-md h2 { font-size: 1.05rem; }
  .rendered-md h3, .rendered-md h4 { font-size: .97rem; }
  .rendered-md p { margin: .7rem 0; }
  .rendered-md ul, .rendered-md ol { margin: .7rem 0; padding-left: 1.3rem; }
  .rendered-md li { margin: .25rem 0; }
  .rendered-md li::marker { color: var(--text-muted); }
  .rendered-md table { border-collapse: collapse; margin: .8rem 0; font-size: .92rem; }
  .rendered-md th, .rendered-md td { border: 1px solid var(--border); padding: .4rem .7rem; text-align: left; }
  .rendered-md th { background: var(--surface); }
  .rendered-md code { font-family: var(--mono); background: var(--surface); border: 1px solid var(--border); padding: .05rem .3rem; border-radius: 5px; font-size: .88em; }
  .rendered-md pre { background: var(--surface); border: 1px solid var(--border); padding: .8rem 1rem; border-radius: var(--radius-sm); overflow-x: auto; }
  .rendered-md pre code { background: none; border: none; padding: 0; }
  .rendered-md blockquote { border-left: 3px solid var(--border-strong); margin: .8rem 0; padding-left: .9rem; color: var(--text-muted); }
  .rendered-md hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
  .rendered-md.inline, .rendered-md.inline p { display: inline; margin: 0; }

  @media (max-width: 600px) {
    .page { padding: 1.5rem 1rem 4rem; }
    .controls .model { width: 100%; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  }
</style>
"""

TOPBAR = """
<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="/"><span class="brand-mark">&#937;</span> Question Bank</a>
    <nav class="nav">
      <a class="nav-link" href="/">Search</a>
      <a class="nav-link" href="/chat">Tutor chat</a>
    </nav>
  </div>
</header>
"""

MATH_ASSETS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
"""

# Shared markdown + KaTeX renderer used by the question page and the chat page.
RENDER_MD_JS = r"""
<script>
const MATH_DELIMITERS = [
  {left: "$$", right: "$$", display: true},
  {left: "\\[", right: "\\]", display: true},
  {left: "\\(", right: "\\)", display: false},
  {left: "$", right: "$", display: false}
];

function renderMD(el, text, inline) {
  const html = inline ? marked.parseInline(text || "") : marked.parse(text || "");
  el.innerHTML = DOMPurify.sanitize(html);
  if (window.renderMathInElement) {
    try {
      renderMathInElement(el, {delimiters: MATH_DELIMITERS, throwOnError: false});
    } catch (e) { /* math is a nicety; plain text still reads fine */ }
  }
}
</script>
"""


SEARCH_PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question Bank</title>
"""
    + STYLES
    + """</head>
<body>
"""
    + TOPBAR
    + """
<main class="page">
  <h1 class="title">Search the question bank</h1>
  <p class="subtitle">Find a past-paper question by its wording or topic, then open it to get an explanation or have your answer marked.</p>

  <form class="searchbar" method="get" action="/">
    <div class="searchfield">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <circle cx="11" cy="11" r="7"></circle><path d="M20 20l-3.5-3.5"></path>
      </svg>
      <input type="text" name="q" placeholder="Search question text&hellip;" value="{{ q }}" autocomplete="off">
    </div>
    <select name="topic" onchange="this.form.submit()" aria-label="Filter by topic">
      <option value="">All topics</option>
      {% for t in topics %}
        <option value="{{ t }}" {{ "selected" if t == topic else "" }}>{{ t }}</option>
      {% endfor %}
    </select>
    <button class="btn btn-primary" type="submit">Search</button>
  </form>

  {% if results %}
    <p class="toolbar-note">
      {{ results|length }} question{{ "" if results|length == 1 else "s" }}
      {%- if q %} matching &ldquo;{{ q }}&rdquo;{% elif topic %} in {{ topic }}{% else %} in the bank{% endif -%}
    </p>
  {% endif %}

  {% for r in results %}
    <a class="card" href="/question?paper={{ r.paper }}&amp;question={{ r.question }}">
      <div class="card-meta">
        <span class="chip">{{ r.paper }}</span>
        <span class="chip chip-plain">Q{{ r.question }}</span>
        <span class="chip chip-plain">{{ r.marks }} mark{{ "" if r.marks == 1 else "s" }}</span>
        {% if r.topics %}<span class="card-topics">{{ r.topics|join(", ") }}</span>{% endif %}
      </div>
      <div class="card-text">{{ r.text[:220] }}{{ "&hellip;"|safe if r.text|length > 220 else "" }}</div>
    </a>
  {% else %}
    <div class="empty">
      <p>No questions match that search.</p>
      <p>Try a shorter phrase, or pick a topic from the filter.</p>
    </div>
  {% endfor %}
</main>
</body>
</html>
"""
)


QUESTION_PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ q.paper }} &middot; Q{{ q.question }}</title>
"""
    + STYLES
    + MATH_ASSETS
    + """</head>
<body>
"""
    + TOPBAR
    + """
<main class="page">
  <a class="backlink" href="/">&larr; All questions</a>

  <div class="q-head">
    <span class="chip">{{ q.paper }}</span>
    <h1 class="title" style="margin:0">Question {{ q.question }}</h1>
    <span class="chip chip-plain">{{ q.marks }} mark{{ "" if q.marks == 1 else "s" }}</span>
  </div>

  <article class="paper-card">
    <div class="q-body">{{ q.text }}</div>

    {% for img in q.images %}
      <img class="q-image" src="/images?paper={{ q.paper }}&amp;file={{ img.file }}" alt="Question diagram">
    {% endfor %}

    {% if q.options %}
      <ul class="options">
        {% for o in q.options %}
          <li><span class="label">{{ o.label }}</span><span>{{ o.text }}</span></li>
        {% endfor %}
      </ul>
    {% endif %}

    {% if q.formulas %}
      <div>
        {% for f in q.formulas %}
          <span class="formula" data-latex="{{ f.latex }}"></span>
        {% endfor %}
      </div>
    {% endif %}

    {% if q.tables %}
      {% for t in q.tables %}
        <table class="data">
          <tr>{% for h in t.headers %}<th>{{ h }}</th>{% endfor %}</tr>
          {% for row in t.rows %}<tr>{% for c in row %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}
        </table>
      {% endfor %}
    {% endif %}
  </article>

  <section class="panel">
    <div class="panel-head">
      <h2>Explanation</h2>
      <span class="hint">Worked through step by step</span>
    </div>
    <div class="panel-body">
      <div class="controls" style="margin-top:0">
        <select id="explain-provider" aria-label="Model provider">
          {% for p in providers %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
        </select>
        <input class="model" type="text" id="explain-model" placeholder="Model override (optional)">
        <button class="btn btn-primary" id="explain-btn" onclick="doExplain()">Explain this question</button>
      </div>
      <div class="output rendered-md" id="explain-result"></div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Mark my answer</h2>
      <span class="hint">Checked against the marking points you give</span>
    </div>
    <div class="panel-body">
      <div class="field">
        <label class="field-label" for="grade-answer">Your answer</label>
        <textarea id="grade-answer" placeholder="Write your answer as you would in the exam&hellip;" style="min-height:4.5rem"></textarea>
      </div>
      <div class="field">
        <label class="field-label" for="grade-points">Marking points</label>
        <textarea id="grade-points" placeholder="One marking point per line"></textarea>
      </div>
      <div class="controls">
        <select id="grade-provider" aria-label="Model provider">
          {% for p in providers %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
        </select>
        <input class="model" type="text" id="grade-model" placeholder="Model override (optional)">
        <button class="btn btn-primary" id="grade-btn" onclick="doGrade()">Mark answer</button>
      </div>
      <div class="output" id="grade-result"></div>
    </div>
  </section>
</main>
"""
    + RENDER_MD_JS
    + r"""
<script>
const PAPER = {{ q.paper|tojson }};
const QUESTION = {{ q.question|tojson }};
const QUESTION_IMAGES = {{ q.question_images|tojson }};

window.addEventListener("load", () => {
  document.querySelectorAll(".formula").forEach(el => {
    if (window.katex) katex.render(el.dataset.latex, el, {throwOnError: false});
  });
});

function busy(el, label) {
  el.innerHTML = "";
  const status = document.createElement("div");
  status.className = "status";
  const dot = document.createElement("span");
  dot.className = "spinner";
  const text = document.createElement("span");
  text.textContent = label;
  status.append(dot, text);
  el.appendChild(status);
  return status;
}

function showError(node, message) {
  const box = document.createElement("div");
  box.className = "alert";
  box.textContent = message;
  node.replaceWith(box);
}

async function doExplain() {
  const out = document.getElementById("explain-result");
  const btn = document.getElementById("explain-btn");
  out.innerHTML = "";
  for (const qi of QUESTION_IMAGES) {
    const img = document.createElement("img");
    img.className = "q-image";
    img.alt = "Question " + QUESTION + " as printed";
    img.src = "/question-images?paper=" + encodeURIComponent(PAPER) + "&file=" + encodeURIComponent(qi.file);
    out.appendChild(img);
  }
  const status = document.createElement("div");
  status.className = "status";
  status.innerHTML = '<span class="spinner"></span><span>Working through the question&hellip;</span>';
  out.appendChild(status);
  btn.disabled = true;
  try {
    const res = await fetch("/api/explain", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        paper: PAPER, question: QUESTION,
        provider: document.getElementById("explain-provider").value,
        model: document.getElementById("explain-model").value,
      }),
    });
    const data = await res.json();
    if (res.ok) {
      const textDiv = document.createElement("div");
      renderMD(textDiv, data.text);
      status.replaceWith(textDiv);
    } else {
      showError(status, "Couldn't explain this question: " + data.error);
    }
  } catch (e) {
    showError(status, "Couldn't reach the server: " + e);
  } finally {
    btn.disabled = false;
  }
}

async function doGrade() {
  const out = document.getElementById("grade-result");
  const btn = document.getElementById("grade-btn");
  const answer = document.getElementById("grade-answer").value.trim();
  const points = document.getElementById("grade-points").value.split("\n").map(s => s.trim()).filter(Boolean);

  if (!answer) {
    out.innerHTML = '<div class="alert">Write an answer first, then mark it.</div>';
    return;
  }
  if (!points.length) {
    out.innerHTML = '<div class="alert">Add at least one marking point, one per line.</div>';
    return;
  }

  const status = busy(out, "Marking against " + points.length + " point" + (points.length === 1 ? "" : "s") + "\u2026");
  btn.disabled = true;
  try {
    const res = await fetch("/api/grade", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        paper: PAPER, question: QUESTION,
        answer: answer,
        marking_points: points,
        provider: document.getElementById("grade-provider").value,
        model: document.getElementById("grade-model").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) { showError(status, "Couldn't mark this answer: " + data.error); return; }

    out.innerHTML = "";

    const score = document.createElement("div");
    score.className = "score";
    const value = document.createElement("span");
    value.className = "score-value";
    value.textContent = data.marks_awarded + " / " + data.max_marks;
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = (data.max_marks ? (100 * data.marks_awarded / data.max_marks) : 0) + "%";
    track.appendChild(fill);
    score.append(value, track);
    out.appendChild(score);

    for (const p of data.point_results) {
      const row = document.createElement("div");
      row.className = "point " + (p.earned ? "earned" : "missed");
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = p.earned ? "\u2713" : "\u2715";
      const body = document.createElement("div");
      body.className = "point-text";
      const pt = document.createElement("span");
      pt.className = "pt";
      pt.textContent = p.point + " \u2014 ";
      const why = document.createElement("span");
      why.className = "why rendered-md inline";
      renderMD(why, p.justification, true);
      body.append(pt, why);
      row.append(badge, body);
      out.appendChild(row);
    }

    if (data.feedback) {
      const feedback = document.createElement("div");
      feedback.className = "feedback rendered-md";
      renderMD(feedback, data.feedback);
      out.appendChild(feedback);
    }
  } catch (e) {
    showError(status, "Couldn't reach the server: " + e);
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""
)


CHAT_PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Physics Tutor</title>
"""
    + STYLES
    + MATH_ASSETS
    + """
<style>
  html, body { height: 100%; }
  body { display: flex; flex-direction: column; overflow: hidden; }

  .chat-topbar .topbar-inner { max-width: 100%; gap: .5rem; }
  .chat-topbar select, .chat-topbar input { font-size: .85rem; padding: .3rem .5rem; }
  .chat-topbar .model { width: 10rem; }
  .chat-tools { margin-left: auto; display: flex; align-items: center; gap: .45rem; }

  #messages { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
  .thread { max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem 1rem; display: flex; flex-direction: column; gap: 1.4rem; }

  .msg { line-height: 1.6; max-width: 100%; }
  .msg.user {
    align-self: flex-end;
    max-width: 78%;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: .6rem .95rem;
    border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
    white-space: pre-wrap;
  }
  .msg.assistant { align-self: stretch; display: flex; gap: .75rem; }
  .avatar {
    flex: none;
    width: 26px; height: 26px;
    margin-top: .15rem;
    display: grid; place-items: center;
    border-radius: 7px;
    background: var(--accent);
    color: #fff;
    font-family: var(--mono);
    font-size: .78rem;
  }
  .msg.assistant .content { min-width: 0; flex: 1; }
  .msg.error .content, .msg.error { color: var(--bad); white-space: pre-wrap; }
  .msg img.q-image { max-width: 100%; display: block; border-radius: var(--radius-sm); border: 1px solid var(--border); margin-bottom: .8rem; }

  .typing { display: inline-flex; gap: 4px; padding: .45rem 0; }
  .typing span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--border-strong);
    animation: blink 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2) { animation-delay: .18s; }
  .typing span:nth-child(3) { animation-delay: .36s; }
  @keyframes blink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }

  .copy {
    margin-top: .5rem;
    font: inherit;
    font-size: .8rem;
    color: var(--text-muted);
    background: none;
    border: none;
    padding: .2rem .35rem;
    border-radius: 5px;
    cursor: pointer;
    opacity: 0;
    transition: opacity .15s, background .15s;
  }
  .msg.assistant:hover .copy, .copy:focus-visible { opacity: 1; }
  .copy:hover { background: var(--surface-hover); color: var(--text); }

  /* empty state -- faint graph paper, the one flourish on the page */
  .welcome { margin: auto 0; text-align: center; padding: 3rem 0 2rem; }
  .welcome-grid {
    width: 76px; height: 76px;
    margin: 0 auto 1.1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 12px 12px;
    display: grid; place-items: center;
    font-family: var(--mono);
    color: var(--accent);
    font-size: 1.3rem;
  }
  .welcome h2 { font-size: 1.15rem; font-weight: 600; margin: 0 0 .3rem; letter-spacing: -.01em; }
  .welcome p { color: var(--text-muted); margin: 0 0 1.4rem; font-size: .92rem; }
  .suggestions { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; }
  .suggestion {
    font: inherit;
    font-size: .87rem;
    text-align: left;
    padding: .55rem .8rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--bg);
    color: var(--text-muted);
    cursor: pointer;
    transition: background .15s, color .15s, border-color .15s;
  }
  .suggestion:hover { background: var(--surface); color: var(--text); border-color: var(--border-strong); }

  /* composer */
  #inputbar { border-top: 1px solid var(--border); background: var(--bg); padding: .9rem 1.25rem 1.1rem; }
  .inputwrap {
    max-width: 720px;
    margin: 0 auto;
    display: flex;
    gap: .5rem;
    align-items: flex-end;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: .45rem .45rem .45rem .9rem;
    background: var(--bg);
    transition: border-color .15s, box-shadow .15s;
  }
  .inputwrap:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  #composer {
    flex: 1;
    border: none;
    background: none;
    padding: .45rem 0;
    max-height: 10rem;
    min-height: auto;
    resize: none;
  }
  #composer:focus { outline: none; box-shadow: none; }
  #send {
    flex: none;
    width: 34px; height: 34px;
    display: grid; place-items: center;
    border: none;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    cursor: pointer;
    transition: background .15s, opacity .15s;
  }
  #send:hover:not(:disabled) { background: var(--accent-hover); }
  #send:disabled { background: var(--border-strong); cursor: default; }
  .composer-hint { max-width: 720px; margin: .5rem auto 0; color: var(--text-muted); font-size: .78rem; text-align: center; }

  @media (max-width: 640px) {
    .chat-topbar .model { display: none; }
    .thread { padding: 1.5rem 1rem 1rem; }
    #inputbar { padding: .7rem 1rem .9rem; }
  }
</style>
</head>
<body>
<header class="topbar chat-topbar">
  <div class="topbar-inner">
    <a class="brand" href="/"><span class="brand-mark">&#937;</span> Physics Tutor</a>
    <div class="chat-tools">
      <a class="nav-link" href="/">Search</a>
      <select id="provider" aria-label="Model provider">
        {% for p in providers %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
      </select>
      <input class="model" type="text" id="model" placeholder="Model override">
      <button class="btn" id="reset">New chat</button>
    </div>
  </div>
</header>

<div id="messages"><div class="thread" id="thread"></div></div>

<div id="inputbar">
  <div class="inputwrap">
    <textarea id="composer" rows="1" placeholder="Ask about a topic, or name a question &mdash; e.g. Q7 in 9702/12/O/N/25"></textarea>
    <button id="send" disabled aria-label="Send message">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 19V5"></path><path d="M5 12l7-7 7 7"></path>
      </svg>
    </button>
  </div>
  <p class="composer-hint">Enter to send &middot; Shift + Enter for a new line</p>
</div>
"""
    + RENDER_MD_JS
    + r"""
<script>
let history = [];
const thread = document.getElementById("thread");
const messages = document.getElementById("messages");
const composer = document.getElementById("composer");
const sendBtn = document.getElementById("send");

const SUGGESTIONS = [
  "Explain Q7 in 9702/12/O/N/25",
  "Why does terminal velocity happen?",
  "Derive the centripetal acceleration formula",
];

function scrollToEnd() {
  messages.scrollTop = messages.scrollHeight;
}

function showWelcome() {
  thread.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "welcome";
  wrap.innerHTML =
    '<div class="welcome-grid">\u03a9</div>' +
    "<h2>What are we working on?</h2>" +
    "<p>Ask a physics question, or name a past-paper question and I'll pull it up.</p>";
  const chips = document.createElement("div");
  chips.className = "suggestions";
  for (const s of SUGGESTIONS) {
    const b = document.createElement("button");
    b.className = "suggestion";
    b.type = "button";
    b.textContent = s;
    b.addEventListener("click", () => {
      composer.value = s;
      autosize();
      send();
    });
    chips.appendChild(b);
  }
  wrap.appendChild(chips);
  thread.appendChild(wrap);
}

function clearWelcome() {
  const w = thread.querySelector(".welcome");
  if (w) w.remove();
}

function addUser(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  thread.appendChild(div);
  scrollToEnd();
  return div;
}

function addAssistant() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "\u03a9";
  const content = document.createElement("div");
  content.className = "content";
  content.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  row.append(avatar, content);
  thread.appendChild(row);
  scrollToEnd();
  return {row, content};
}

function addCopyButton(content, text) {
  const btn = document.createElement("button");
  btn.className = "copy";
  btn.type = "button";
  btn.textContent = "Copy";
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = "Copy"; }, 1400);
    } catch (e) {
      btn.textContent = "Copy failed";
    }
  });
  content.appendChild(btn);
}

function autosize() {
  composer.style.height = "auto";
  composer.style.height = Math.min(composer.scrollHeight, 160) + "px";
  sendBtn.disabled = !composer.value.trim();
}

composer.addEventListener("input", autosize);

composer.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

sendBtn.addEventListener("click", send);

document.getElementById("reset").addEventListener("click", () => {
  history = [];
  showWelcome();
  composer.focus();
});

async function send() {
  const text = composer.value.trim();
  if (!text || sendBtn.disabled) return;
  clearWelcome();
  composer.value = "";
  autosize();
  addUser(text);
  history.push({role: "user", content: text});
  sendBtn.disabled = true;
  const {row, content} = addAssistant();
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        messages: history,
        provider: document.getElementById("provider").value,
        model: document.getElementById("model").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      row.className = "msg assistant error";
      content.textContent = "That didn't go through: " + data.error;
      return;
    }
    content.innerHTML = "";
    for (const qi of (data.question_images || [])) {
      const img = document.createElement("img");
      img.className = "q-image";
      img.alt = "Question as printed";
      img.src = "/question-images?paper=" + encodeURIComponent(data.paper) + "&file=" + encodeURIComponent(qi.file);
      content.appendChild(img);
    }
    const textDiv = document.createElement("div");
    textDiv.className = "rendered-md";
    renderMD(textDiv, data.reply);
    content.appendChild(textDiv);
    addCopyButton(content, data.reply);
    history.push({role: "assistant", content: data.reply});
  } catch (e) {
    row.className = "msg assistant error";
    content.textContent = "Couldn't reach the server: " + e;
  } finally {
    autosize();
    scrollToEnd();
    composer.focus();
  }
}

showWelcome();
composer.focus();
</script>
</body>
</html>
"""
)


@app.route("/")
def index():
    db = Path(app.config["DB_PATH"])
    q = request.args.get("q", "").strip()
    topic = request.args.get("topic", "").strip()
    if q:
        results = search_text(db, q)
    elif topic:
        results = get_by_topic(db, topic)
    else:
        results = _list_questions(db)
    return render_template_string(SEARCH_PAGE, results=results, topics=_topics(db), q=q, topic=topic)


@app.route("/question")
def question_detail():
    db = Path(app.config["DB_PATH"])
    paper_code = request.args.get("paper", "")
    question_number = request.args.get("question", type=int)
    q = get_by_reference(db, paper_code, question_number) if question_number is not None else None
    if q is None:
        return "Question not found", 404
    return render_template_string(QUESTION_PAGE, q=q, providers=sorted(_DEFAULT_MODELS))


@app.route("/images")
def serve_image():
    filename = request.args.get("file", "")
    paper_code = request.args.get("paper", "")
    for d in _guess_asset_dirs(Path(app.config["IMAGES_DIR"]), paper_code, "images"):
        if (d / filename).is_file():
            return send_from_directory(d, filename)
    return "", 404


@app.route("/question-images")
def serve_question_image():
    filename = request.args.get("file", "")
    paper_code = request.args.get("paper", "")
    for d in _guess_asset_dirs(Path(app.config["QUESTION_IMAGES_DIR"]), paper_code, "question_images"):
        if (d / filename).is_file():
            return send_from_directory(d, filename)
    return "", 404


@app.route("/api/explain", methods=["POST"])
def api_explain():
    data = request.get_json(force=True)
    try:
        text = explain_question(
            Path(app.config["DB_PATH"]),
            Path(app.config["IMAGES_DIR"]),
            data["paper"],
            int(data["question"]),
            provider=data.get("provider") or "anthropic",
            model=(data.get("model") or None),
        )
        return jsonify({"text": text})
    except Exception as e:  # surfaced to the UI as an error message, not a 500 page
        return jsonify({"error": str(e)}), 400


@app.route("/api/grade", methods=["POST"])
def api_grade():
    data = request.get_json(force=True)
    try:
        result = grade_answer(
            Path(app.config["DB_PATH"]),
            data["paper"],
            int(data["question"]),
            data.get("answer", ""),
            data.get("marking_points") or [],
            provider=data.get("provider") or "anthropic",
            model=(data.get("model") or None),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/chat")
def chat_page():
    return render_template_string(CHAT_PAGE, providers=sorted(_DEFAULT_MODELS))


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    messages = list(data.get("messages") or [])

    # If the latest user turn references a specific question, splice its real
    # content in before calling the model -- see _detect_question_reference.
    # The matched question's whole-page image(s), if any, ride along in the
    # response so the UI can show the question before the explanation text.
    question_images: list[dict] = []
    ref_paper = None
    if messages and messages[-1].get("role") == "user":
        ref = _detect_question_reference(messages[-1]["content"])
        if ref:
            paper_code, question_number = ref
            q = get_by_reference(Path(app.config["DB_PATH"]), paper_code, question_number)
            if q is not None:
                context = _question_context_text(q)
                question_images = q.get("question_images", [])
                ref_paper = paper_code
            else:
                context = f"(No question {question_number} found in the database for paper {paper_code}.)"
            messages = messages[:-1] + [{"role": "user", "content": f"{context}\n\n{messages[-1]['content']}"}]

    try:
        reply = chat_reply(
            messages,
            provider=data.get("provider") or "anthropic",
            model=(data.get("model") or None),
        )
        return jsonify({"reply": reply, "question_images": question_images, "paper": ref_paper})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _check_db(db_path: Path) -> None:
    """Refuse to start against a database that has nothing to serve.

    Every page here opens the DB lazily per request, so an unusable --db used
    to surface as a 500 on the first page load with a SQL error naming the
    table rather than the file. Checking once at startup puts the path -- the
    thing that's actually wrong -- in front of the person who typed it.
    """
    if not db_path.exists():
        sys.exit(
            f"No database at {db_path.resolve()}\n"
            "Pass --db pointing at the questions.db your batch produced "
            "(run_batch.py writes it to <output-dir>/questions.db)."
        )
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        count = conn.execute("SELECT count(*) FROM questions").fetchone()[0]
        papers = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        conn.close()
    except sqlite3.DatabaseError as exc:
        size = db_path.stat().st_size
        hint = " It is 0 bytes -- sqlite creates an empty file for a path that doesn't exist, so this is\nmost likely the wrong path." if size == 0 else ""
        sys.exit(f"{db_path.resolve()} is not a usable question database ({exc}).{hint}")
    if not count:
        sys.exit(f"{db_path.resolve()} has no questions in it. Did phase 11 run?")
    print(f"Serving {count} questions from {papers} papers out of {db_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 13: minimal web UI over the question database.")
    parser.add_argument("--db", type=Path, default=Path("output/questions.db"))
    parser.add_argument("--images", type=Path, default=Path("output/images"))
    parser.add_argument("--question-images", type=Path, default=Path("output/question_images"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    _check_db(args.db)

    app.config["DB_PATH"] = str(args.db)
    app.config["IMAGES_DIR"] = str(args.images)
    app.config["QUESTION_IMAGES_DIR"] = str(args.question_images)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()