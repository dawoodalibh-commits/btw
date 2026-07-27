#!/usr/bin/env python3
"""Phase-major batch driver — the throughput-oriented alternative to run_pipeline.sh.

run_pipeline.sh is paper-major: it runs all 11 phases for one paper, then all
11 for the next. Every paper therefore reloads DocLayout-YOLO, Pix2Tex and
PaddleOCR from scratch, which across a few hundred papers costs far more than
the actual inference does.

This driver flips the loop: it runs one phase across *all* papers before
moving to the next. That buys two things.

  * The three model phases (2 layout, 5 formulas, 7 tables) run as a single
    process each, loading their model once for the whole batch and keeping
    the GPU fed instead of idling through repeated startups.
  * The eight CPU-bound phases fan out across cores with a process pool,
    since they're independent per paper and mostly PyMuPDF rasterization.

Phase 11 (DB load) stays serial so the papers land in one shared SQLite file
rather than one DB per paper.

A paper that fails any phase is dropped from later phases and reported at the
end; the rest of the batch continues.

Usage:
    ./run_batch.py papers/ --output-dir output --device cuda --backend doclayout_yolo
    ./run_batch.py papers/ --jobs 8 --dpi 150
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import-time settings for every child process. Albumentations otherwise makes
# a blocking "is there a new version?" HTTPS call on each import, which on a
# throttled box stalls for its full timeout once per phase per paper; the
# thread caps stop N concurrent workers each spawning N BLAS threads and
# thrashing the cores they're supposed to be sharing.
_CHILD_ENV = {
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def find_pdfs(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() == ".pdf")
    return [target]


def run(cmd: list[str], log_prefix: str) -> tuple[bool, str]:
    """Runs one phase invocation, returning (ok, combined output)."""
    env = {**os.environ, **_CHILD_ENV}
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    ok = proc.returncode == 0
    if not ok:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        print(f"!!! FAILED {log_prefix}\n{tail}", file=sys.stderr)
    return ok, proc.stdout


def run_parallel(name: str, cmds: dict[Path, list[str]], jobs: int) -> set[Path]:
    """Runs one CPU phase across papers in a pool. Returns the papers that failed."""
    start = time.monotonic()
    failed: set[Path] = set()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run, cmd, f"{name} for {pdf.name}"): pdf for pdf, cmd in cmds.items()}
        for future in futures:
            ok, _ = future.result()
            if not ok:
                failed.add(futures[future])
    elapsed = time.monotonic() - start
    print(f"=== {name}: {len(cmds) - len(failed)}/{len(cmds)} ok in {elapsed:.1f}s ===")
    return failed


def run_batched(name: str, cmd: list[str], papers: list[Path]) -> set[Path]:
    """Runs one model phase as a single process over every paper at once.

    The child keeps going after a per-paper error and reports which PDF broke,
    so failures are recovered by scanning its output rather than by exit code.
    """
    start = time.monotonic()
    ok, _ = run(cmd, f"{name} (batch of {len(papers)})")
    elapsed = time.monotonic() - start
    if not ok:
        print(f"=== {name}: batch FAILED after {elapsed:.1f}s ===")
        return set(papers)
    print(f"=== {name}: {len(papers)} papers in {elapsed:.1f}s (model loaded once) ===")
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-major batch runner for many PDFs.")
    parser.add_argument("input", type=Path, help="A PDF or a folder of PDFs")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--paper", default="unknown", help="Fallback paper code when auto-detection fails")
    parser.add_argument("--backend", default="doclayout_yolo", choices=("doclayout_yolo", "ppstructure"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dpi", type=int, default=None, help="Override rasterization DPI for every phase")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel workers for the CPU phases (default: half the cores)",
    )
    args = parser.parse_args()

    pdfs = find_pdfs(args.input)
    if not pdfs:
        sys.exit(f"No PDFs found in {args.input}")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "questions.db"
    dpi = ["--dpi", str(args.dpi)] if args.dpi else []

    live = list(pdfs)  # papers still healthy; failures drop out as we go
    failed: dict[Path, str] = {}

    def out(pdf: Path, phase: str) -> str:
        return str(root / pdf.stem / phase)

    def drop(newly_failed: set[Path], phase: str) -> None:
        for pdf in newly_failed:
            failed.setdefault(pdf, phase)
        live[:] = [p for p in live if p not in newly_failed]

    print(f"=== {len(pdfs)} papers | {args.jobs} parallel workers | backend={args.backend} device={args.device} ===")
    started = time.monotonic()

    # Phase 1 — PDF extraction (CPU, parallel)
    drop(run_parallel("phase 1 extract_pdf", {
        p: ["extract_pdf.py", str(p), "--output-dir", out(p, "extracted")] for p in live
    }, args.jobs), "extract_pdf")

    # Phase 2 — layout detection (GPU, one model load for the whole batch)
    if live:
        drop(run_batched("phase 2 layout_detection", [
            "layout_detection.py", *[str(p) for p in live],
            "--output-root", str(root), "--backend", args.backend, "--device", args.device, *dpi,
        ], live), "layout_detection")

    # Phase 3 — merge (CPU, parallel)
    drop(run_parallel("phase 3 merge_layout", {
        p: ["merge_layout.py", "--extracted", out(p, "extracted"),
            "--layout", out(p, "layout"), "--output-dir", out(p, "merged")] for p in live
    }, args.jobs), "merge_layout")

    # Phase 4 — question parsing (CPU, parallel)
    drop(run_parallel("phase 4 question_parser", {
        p: ["question_parser.py", "--merged", out(p, "merged"), "--output-dir", out(p, "questions")] for p in live
    }, args.jobs), "question_parser")

    # Phase 5 — formulas (GPU, one model load for the whole batch)
    if live:
        drop(run_batched("phase 5 formula_extractor", [
            "formula_extractor.py", *[str(p) for p in live],
            "--output-root", str(root), "--device", args.device, *dpi,
        ], live), "formula_extractor")

    # Phase 6 — image export (CPU, parallel)
    drop(run_parallel("phase 6 image_exporter", {
        p: ["image_exporter.py", str(p), "--merged", out(p, "merged"),
            "--extracted", out(p, "extracted"), "--output-dir", out(p, "images"), *dpi] for p in live
    }, args.jobs), "image_exporter")

    # Phase 7 — tables (GPU with paddlepaddle-gpu, one model load for the batch)
    if live:
        drop(run_batched("phase 7 table_extractor", [
            "table_extractor.py", *[str(p) for p in live],
            "--output-root", str(root), "--device", args.device, *dpi,
        ], live), "table_extractor")

    # Phase 8 — question images (CPU, parallel)
    drop(run_parallel("phase 8 question_image_exporter", {
        p: ["question_image_exporter.py", str(p), "--merged", out(p, "merged"),
            "--questions", out(p, "questions"), "--output-dir", out(p, "question_images"), *dpi] for p in live
    }, args.jobs), "question_image_exporter")

    # Phase 9 — build (CPU, parallel)
    drop(run_parallel("phase 9 build_questions", {
        p: ["build_questions.py", "--extracted", out(p, "extracted"), "--questions", out(p, "questions"),
            "--formulas", out(p, "formulas"), "--images", out(p, "images"), "--tables", out(p, "tables"),
            "--question-images", out(p, "question_images"), "--output-dir", out(p, "built"),
            "--paper", args.paper] for p in live
    }, args.jobs), "build_questions")

    # Phase 10 — topic classification (CPU, parallel)
    drop(run_parallel("phase 10 topic_classifier", {
        p: ["topic_classifier.py", "--built", out(p, "built"), "--output-dir", out(p, "topics")] for p in live
    }, args.jobs), "topic_classifier")

    # Phase 11 — DB load. Serial and single-writer: every paper lands in one
    # shared SQLite file, and concurrent writers to it would just contend.
    db_failed: set[Path] = set()
    for pdf in live:
        ok, _ = run(
            ["database.py", "--classified", f"{out(pdf, 'topics')}/classified_questions.json", "--db", str(db_path)],
            f"phase 11 database for {pdf.name}",
        )
        if not ok:
            db_failed.add(pdf)
    drop(db_failed, "database")
    print(f"=== phase 11 database: {len(live)} papers -> {db_path} ===")

    elapsed = time.monotonic() - started
    print(f"\n=== Batch complete: {len(live)}/{len(pdfs)} papers in {elapsed:.1f}s -> {root} ===")
    if failed:
        print("=== Failed papers ===", file=sys.stderr)
        for pdf, phase in sorted(failed.items()):
            print(f"  {pdf.name} (at {phase})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
