"""Phase 12 — LLM Tutoring & Grading.

Uses the database (Phase 10) via retrieval_api (Phase 11) to do the two
things a student actually wants: "explain this question" and "grade my
answer." Supports four backends -- Anthropic (Claude), Groq, OpenRouter, and
Gemini -- selected per call with --provider. Groq and OpenRouter both speak
the OpenAI-compatible chat-completions API, so they share one code path;
Gemini and Anthropic each get their own since their SDKs/wire formats differ
(notably: only Anthropic gets a JSON-schema-enforced response for grading --
the others get the schema spelled out in the prompt and a plain JSON-object
response, since not every model behind those providers supports strict
schema enforcement).

Mark schemes aren't ingested by this pipeline -- Phases 1-9 all operate on
a paper's *_qp_*.pdf (the question paper), not its *_ms_*.pdf (the mark
scheme). grade_answer() takes marking points as an explicit argument for
now; pointing the same extract_pdf.py -> question_parser.py pipeline at a
mark scheme PDF to derive them automatically is a natural next phase, not
built here.

Usage:
    python tutor.py explain --db output/questions.db --paper 9702/12/O/N/25 --question 7
    python tutor.py explain --db output/questions.db --paper 9702/12/O/N/25 --question 7 \
        --provider groq --model llama-3.3-70b-versatile
    python tutor.py grade --db output/questions.db --paper 9702/12/O/N/25 --question 4 \
        --answer "8%" --marking-points "correct method shown" "correct final answer" \
        --provider gemini

Required credentials (only for the provider(s) you actually use):
    anthropic  -> ANTHROPIC_API_KEY (or `ant auth login`)
    groq       -> GROQ_API_KEY
    openrouter -> OPENROUTER_API_KEY
    gemini     -> GEMINI_API_KEY (or GOOGLE_API_KEY)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import anthropic

from retrieval_api import get_by_reference

_TUTOR_SYSTEM = (
    "You are an A-level tutor. Explain exam questions clearly, at the level "
    "of a student studying for this exact paper. Reference the specific "
    "numbers, diagrams, and options given -- don't give a generic explanation "
    "of the topic. This output is printed to a plain terminal, not rendered -- "
    "format all math as plain text: no LaTeX, no $, no \\frac{}{} or \\sqrt{}, "
    "no \\text{}. Use / for division, ^ for exponents, sqrt() for square roots, "
    "and plain unit abbreviations (e.g. 'm s^-1', not '\\text{m s}^{-1}')."
)

# Reasonable per-provider defaults, overridable with --model. OpenRouter in
# particular fronts hundreds of models under this one provider -- its default
# here is just a safe, widely-available starting point, not a recommendation.
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "google/gemma-3-27b-it",
    "gemini": "gemini-2.5-pro",
}

# Groq and OpenRouter both expose an OpenAI-compatible /chat/completions API,
# so one client shape covers both -- only the base URL and API key differ.
_OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_OPENAI_COMPATIBLE_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _openai_compatible_client(provider: str):
    from openai import OpenAI  # local import: only required if this provider is used

    env_var = _OPENAI_COMPATIBLE_API_KEY_ENV[provider]
    if not os.environ.get(env_var):
        raise RuntimeError(f"--provider {provider} requires the {env_var} environment variable")
    return OpenAI(api_key=os.environ[env_var], base_url=_OPENAI_COMPATIBLE_BASE_URLS[provider])


def _image_content_blocks(images_dir: Path, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped image content blocks (base64 source)."""
    blocks = []
    for img in images:
        path = images_dir / img["file"]
        if not path.exists():
            continue
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
    return blocks


def _images_as_data_urls(images_dir: Path, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-chat-completions-shaped image content blocks (data: URL)."""
    blocks = []
    for img in images:
        path = images_dir / img["file"]
        if not path.exists():
            continue
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}})
    return blocks


def _images_as_gemini_parts(images_dir: Path, images: list[dict[str, Any]]) -> list[Any]:
    from google.genai import types  # local import: only required if this provider is used

    parts = []
    for img in images:
        path = images_dir / img["file"]
        if not path.exists():
            continue
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=media_type))
    return parts


def _question_context_text(q: dict[str, Any]) -> str:
    parts = [f"Paper {q['paper']}, Question {q['question']} ({q['marks']} mark(s)):", q["text"]]
    if q["options"]:
        parts.append("Options:")
        parts.extend(f"  {o['label']}. {o['text']}" for o in q["options"])
    if q["formulas"]:
        parts.append("Formulas/expressions in this question (OCR'd from the page):")
        parts.extend(f"  {f['latex']}" for f in q["formulas"] if f["latex"])
    if q["tables"]:
        parts.append("Tables in this question:")
        for t in q["tables"]:
            parts.append(f"  headers: {t['headers']}")
            parts.extend(f"  row: {row}" for row in t["rows"])
    return "\n".join(parts)


def _explain_via_anthropic(model: str, images_dir: Path, q: dict[str, Any], client: anthropic.Anthropic | None = None) -> str:
    client = client or anthropic.Anthropic()
    content = _image_content_blocks(images_dir, q["images"])
    content.append(
        {"type": "text", "text": _question_context_text(q) + "\n\nExplain this question and how to answer it."}
    )
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_TUTOR_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return next(block.text for block in response.content if block.type == "text")


def _explain_via_openai_compatible(provider: str, model: str, images_dir: Path, q: dict[str, Any]) -> str:
    client = _openai_compatible_client(provider)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": _question_context_text(q) + "\n\nExplain this question and how to answer it."}
    ]
    content.extend(_images_as_data_urls(images_dir, q["images"]))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TUTOR_SYSTEM},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content


def _explain_via_gemini(model: str, images_dir: Path, q: dict[str, Any]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment
    contents = _images_as_gemini_parts(images_dir, q["images"])
    contents.append(_question_context_text(q) + "\n\nExplain this question and how to answer it.")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=_TUTOR_SYSTEM),
    )
    return response.text


def explain_question(
    db_path: Path,
    images_dir: Path,
    paper_code: str,
    question_number: int,
    provider: str = "anthropic",
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> str:
    q = get_by_reference(db_path, paper_code, question_number)
    if q is None:
        raise ValueError(f"No question {question_number} found for paper {paper_code}")
    model = model or _DEFAULT_MODELS[provider]

    if provider == "anthropic":
        return _explain_via_anthropic(model, images_dir, q, client=client)
    if provider in _OPENAI_COMPATIBLE_BASE_URLS:
        return _explain_via_openai_compatible(provider, model, images_dir, q)
    if provider == "gemini":
        return _explain_via_gemini(model, images_dir, q)
    raise ValueError(f"Unknown provider: {provider!r}. Choose from {sorted(_DEFAULT_MODELS)}")


_GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "marks_awarded": {"type": "integer"},
        "max_marks": {"type": "integer"},
        "point_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "earned": {"type": "boolean"},
                    "justification": {"type": "string"},
                },
                "required": ["point", "earned", "justification"],
                "additionalProperties": False,
            },
        },
        "feedback": {"type": "string"},
    },
    "required": ["marks_awarded", "max_marks", "point_results", "feedback"],
    "additionalProperties": False,
}

# Only Anthropic's request gets the schema enforced server-side
# (output_config.format). Not every model behind Groq/OpenRouter/Gemini
# supports strict schema enforcement, so those three get the shape spelled
# out in the prompt instead and are asked to return a plain JSON object.
_GRADE_SCHEMA_INSTRUCTIONS = (
    "Respond with ONLY a JSON object (no markdown fences, no commentary) matching exactly this shape:\n"
    '{"marks_awarded": <int>, "max_marks": <int>, '
    '"point_results": [{"point": <string>, "earned": <bool>, "justification": <string>}, ...], '
    '"feedback": <string>}'
)


def _grade_prompt(q: dict[str, Any], student_answer: str, marking_points: list[str]) -> str:
    return (
        f"{_question_context_text(q)}\n\n"
        "Marking points:\n" + "\n".join(f"- {p}" for p in marking_points) + "\n\n"
        f"Student's answer:\n{student_answer}\n\n"
        "Grade the answer against each marking point independently."
    )


def _grade_via_anthropic(
    model: str, q: dict[str, Any], student_answer: str, marking_points: list[str], client: anthropic.Anthropic | None = None
) -> dict[str, Any]:
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_TUTOR_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _GRADE_SCHEMA}},
        messages=[{"role": "user", "content": _grade_prompt(q, student_answer, marking_points)}],
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def _grade_via_openai_compatible(
    provider: str, model: str, q: dict[str, Any], student_answer: str, marking_points: list[str]
) -> dict[str, Any]:
    client = _openai_compatible_client(provider)
    prompt = _grade_prompt(q, student_answer, marking_points) + "\n\n" + _GRADE_SCHEMA_INSTRUCTIONS
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _TUTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _grade_via_gemini(model: str, q: dict[str, Any], student_answer: str, marking_points: list[str]) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = _grade_prompt(q, student_answer, marking_points) + "\n\n" + _GRADE_SCHEMA_INSTRUCTIONS
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=_TUTOR_SYSTEM, response_mime_type="application/json"),
    )
    return json.loads(response.text)


def grade_answer(
    db_path: Path,
    paper_code: str,
    question_number: int,
    student_answer: str,
    marking_points: list[str],
    provider: str = "anthropic",
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    q = get_by_reference(db_path, paper_code, question_number)
    if q is None:
        raise ValueError(f"No question {question_number} found for paper {paper_code}")
    model = model or _DEFAULT_MODELS[provider]

    if provider == "anthropic":
        return _grade_via_anthropic(model, q, student_answer, marking_points, client=client)
    if provider in _OPENAI_COMPATIBLE_BASE_URLS:
        return _grade_via_openai_compatible(provider, model, q, student_answer, marking_points)
    if provider == "gemini":
        return _grade_via_gemini(model, q, student_answer, marking_points)
    raise ValueError(f"Unknown provider: {provider!r}. Choose from {sorted(_DEFAULT_MODELS)}")


_CHAT_SYSTEM = (
    "You are an A-level physics tutor having a conversation with a student. "
    "Answer clearly and at the right level for someone studying for this "
    "exam. Format all math as plain text: no LaTeX, no $, no \\frac{}{} or "
    "\\sqrt{}, no \\text{}. Use / for division, ^ for exponents, sqrt() for "
    "square roots, and plain unit abbreviations (e.g. 'm s^-1', not "
    "'\\text{m s}^{-1}')."
)


def chat_reply(
    messages: list[dict[str, str]],
    provider: str = "anthropic",
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...], starting with a user turn.
    Free-form multi-turn chat, not tied to a specific database question --
    for that, use explain_question / grade_answer instead."""
    model = model or _DEFAULT_MODELS[provider]

    if provider == "anthropic":
        c = client or anthropic.Anthropic()
        response = c.messages.create(
            model=model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=_CHAT_SYSTEM,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return next(block.text for block in response.content if block.type == "text")

    if provider in _OPENAI_COMPATIBLE_BASE_URLS:
        c = _openai_compatible_client(provider)
        response = c.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _CHAT_SYSTEM}]
            + [{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return response.choices[0].message.content

    if provider == "gemini":
        from google import genai
        from google.genai import types

        c = genai.Client()
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        response = c.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=_CHAT_SYSTEM),
        )
        return response.text

    raise ValueError(f"Unknown provider: {provider!r}. Choose from {sorted(_DEFAULT_MODELS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 12: explain and grade questions via an LLM.")
    parser.add_argument("--db", type=Path, default=Path("output/questions.db"))
    parser.add_argument("--images", type=Path, default=Path("output/images"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("explain")
    p.add_argument("--paper", required=True)
    p.add_argument("--question", type=int, required=True)
    p.add_argument("--provider", choices=sorted(_DEFAULT_MODELS), default="anthropic")
    p.add_argument("--model", help="Override the provider's default model")

    p = sub.add_parser("grade")
    p.add_argument("--paper", required=True)
    p.add_argument("--question", type=int, required=True)
    p.add_argument("--answer", required=True)
    p.add_argument("--marking-points", nargs="+", required=True)
    p.add_argument("--provider", choices=sorted(_DEFAULT_MODELS), default="anthropic")
    p.add_argument("--model", help="Override the provider's default model")

    args = parser.parse_args()

    if args.command == "explain":
        print(explain_question(args.db, args.images, args.paper, args.question, provider=args.provider, model=args.model))
    elif args.command == "grade":
        result = grade_answer(
            args.db, args.paper, args.question, args.answer, args.marking_points, provider=args.provider, model=args.model
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
