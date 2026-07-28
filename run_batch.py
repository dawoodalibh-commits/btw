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
  * Phases 5-8 all hang off phase 4 and none of them depends on another, so
    the two GPU phases and the two CPU phases run as two concurrent lanes:
    the cores crop images while the card does formulas and tables, instead of
    each waiting its turn to leave the other idle.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from schemas import PAPER_FAILED_PREFIX

# Import-time settings for every child process. Albumentations otherwise makes
# a blocking "is there a new version?" HTTPS call on each import, which on a
# throttled box stalls for its full timeout once per phase per paper.
# PYTHONUNBUFFERED matters because a child writing to a pipe (rather than a
# terminal) block-buffers its stdout: without it the streamed model phases
# would arrive in 8 KB lumps, which for a phase that runs for hours means no
# visible progress at all.
_CHILD_ENV = {
    "NO_ALBUMENTATIONS_UPDATE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

# Extra caps for the pooled CPU phases only: without them N concurrent
# workers each spawn N BLAS threads and thrash the cores they're supposed to
# be sharing. The GPU phases are one process each and want every core they
# can get for rasterization and pre/post-processing, so they don't get these.
_POOL_CHILD_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

SCRIPT_DIR = Path(__file__).resolve().parent


def find_pdfs(target: Path) -> list[Path]:
    """Resolves a PDF or a folder of PDFs to the list of papers to process.

    The walk is recursive because download_papers.sh --all files papers one
    subfolder per subject (<out-dir>/<code>/*.pdf), so a top-level-only scan
    would come back empty on exactly the layout the batch runner is for.
    """
    if not target.exists():
        sys.exit(f"No such file or directory: {target}")
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() == ".pdf")
    if target.suffix.lower() != ".pdf":
        sys.exit(f"Not a PDF: {target}")
    return [target]


def run(cmd: list[str], log_prefix: str, extra_env: dict[str, str] | None = None, stream: bool = False) -> tuple[bool, str]:
    """Runs one phase invocation, returning (ok, combined output).

    `stream` echoes the child's output as it arrives, for the model phases:
    those run for as long as the whole batch takes, and buffering them means
    a multi-hour phase shows nothing at all until it's over. Pooled phases
    stay buffered, since a dozen workers writing at once is unreadable.
    """
    env = {**os.environ, **_CHILD_ENV, **(extra_env or {})}
    proc = subprocess.Popen(
        [sys.executable, *cmd],
        cwd=SCRIPT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # one stream: keeps the child's own ordering intact
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        if stream:
            sys.stdout.write(line)
    proc.wait()
    output = "".join(lines)
    ok = proc.returncode == 0
    if not ok and not stream:
        tail = "\n".join(output.strip().splitlines()[-15:])
        print(f"!!! FAILED {log_prefix}\n{tail}", file=sys.stderr)
    return ok, output


def parse_paper_failures(output: str) -> set[Path]:
    """Pull the per-paper failures a batched phase reported on its stdout."""
    failed: set[Path] = set()
    for line in output.splitlines():
        if line.startswith(PAPER_FAILED_PREFIX):
            parts = line.split("\t")
            if len(parts) == 3:
                failed.add(Path(parts[2]))
    return failed


def run_parallel(name: str, cmds: dict[Path, list[str]], jobs: int) -> set[Path]:
    """Runs one CPU phase across papers in a pool. Returns the papers that failed."""
    start = time.monotonic()
    failed: set[Path] = set()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run, cmd, f"{name} for {pdf.name}", _POOL_CHILD_ENV): pdf for pdf, cmd in cmds.items()
        }
        for future in as_completed(futures):
            ok, _ = future.result()
            if not ok:
                failed.add(futures[future])
    elapsed = time.monotonic() - start
    print(f"=== {name}: {len(cmds) - len(failed)}/{len(cmds)} ok in {elapsed:.1f}s ===")
    return failed


def run_batched(name: str, cmd: list[str], papers: list[Path]) -> set[Path]:
    """Runs one model phase as a single process over every paper at once.

    The child keeps going after a per-paper error and marks which PDF broke
    on its stdout, because its exit code can't: it's 0 whenever *some* paper
    survived, so failures have to be recovered by scanning the output.
    """
    start = time.monotonic()
    ok, output = run(cmd, f"{name} (batch of {len(papers)})", stream=True)
    elapsed = time.monotonic() - start
    if not ok:
        print(f"=== {name}: batch FAILED after {elapsed:.1f}s ===")
        return set(papers)
    failed = parse_paper_failures(output) & set(papers)
    print(
        f"=== {name}: {len(papers) - len(failed)}/{len(papers)} papers in {elapsed:.1f}s "
        f"(model loaded once) ==="
    )
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-major batch runner for many PDFs.")
    parser.add_argument("input", type=Path, help="A PDF or a folder of PDFs")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--paper", default="unknown", help="Fallback paper code when auto-detection fails")
    # ppstructure by default because it's the only backend that populates
    # phase 5: DocStructBench's "isolate_formula" label means a display
    # equation set on its own line, and exam papers are mostly inline math and
    # small fragments, so DocLayout-YOLO tags none of it. On one 9702 paper
    # ppstructure found 59 formula regions across 8 pages and DocLayout-YOLO
    # found none, leaving every question's formula list empty.
    parser.add_argument("--backend", default="ppstructure", choices=("doclayout_yolo", "ppstructure"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dpi", type=int, default=None, help="Override rasterization DPI for every phase")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel workers for the CPU phases (default: half the cores)",
    )
    parser.add_argument(
        "--layout-batch-size",
        type=int,
        default=None,
        help="Pages per phase-2 detector call. Higher keeps the GPU busier but uses more VRAM.",
    )
    parser.add_argument(
        "--formula-batch-size",
        type=int,
        default=None,
        help="Crops per phase-5 decode call. Higher keeps the GPU busier but uses more VRAM.",
    )
    parser.add_argument(
        "--rec-batch-size",
        type=int,
        default=None,
        help="Text lines per phase-7 recognition pass. Higher keeps the GPU busier but uses more VRAM.",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Run phases 5-8 strictly in order instead of overlapping the GPU and CPU lanes.",
    )
    args = parser.parse_args()

    pdfs = find_pdfs(args.input)
    if not pdfs:
        sys.exit(f"No PDFs found in {args.input}")

    # Output lives at <root>/<stem>/<phase>, so two same-named PDFs in
    # different subfolders would write over each other's phase output.
    by_stem: dict[str, list[Path]] = {}
    for pdf in pdfs:
        by_stem.setdefault(pdf.stem, []).append(pdf)
    clashes = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if clashes:
        detail = "\n".join(f"  {stem}: " + ", ".join(str(p) for p in paths) for stem, paths in sorted(clashes.items()))
        sys.exit(f"Duplicate PDF names would share one output folder:\n{detail}")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "questions.db"
    dpi = ["--dpi", str(args.dpi)] if args.dpi else []
    layout_batch = ["--batch-size", str(args.layout_batch_size)] if args.layout_batch_size else []
    formula_batch = ["--batch-size", str(args.formula_batch_size)] if args.formula_batch_size else []
    rec_batch = ["--rec-batch-size", str(args.rec_batch_size)] if args.rec_batch_size else []

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
            "--output-root", str(root), "--backend", args.backend, "--device", args.device,
            *dpi, *layout_batch,
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

    # Phases 5-8 all read phase 3/4 output and none of them reads another's,
    # so they split into a GPU lane and a CPU lane that run at the same time.
    # Within the GPU lane the two phases stay serial: torch and paddle
    # otherwise size their allocators against a card they each think is empty.
    # `papers` is a snapshot because both lanes read it while `drop` is still
    # waiting on the join.
    def gpu_lane(papers: list[Path]) -> list[tuple[set[Path], str]]:
        if not papers:
            return []
        return [
            (run_batched("phase 5 formula_extractor", [
                "formula_extractor.py", *[str(p) for p in papers],
                "--output-root", str(root), "--device", args.device, *dpi, *formula_batch,
            ], papers), "formula_extractor"),
            (run_batched("phase 7 table_extractor", [
                "table_extractor.py", *[str(p) for p in papers],
                "--output-root", str(root), "--device", args.device, *dpi, *rec_batch,
            ], papers), "table_extractor"),
        ]

    def cpu_lane(papers: list[Path]) -> list[tuple[set[Path], str]]:
        if not papers:
            return []
        return [
            (run_parallel("phase 6 image_exporter", {
                p: ["image_exporter.py", str(p), "--merged", out(p, "merged"),
                    "--extracted", out(p, "extracted"), "--output-dir", out(p, "images"), *dpi] for p in papers
            }, args.jobs), "image_exporter"),
            (run_parallel("phase 8 question_image_exporter", {
                p: ["question_image_exporter.py", str(p), "--merged", out(p, "merged"),
                    "--questions", out(p, "questions"), "--output-dir", out(p, "question_images"), *dpi]
                for p in papers
            }, args.jobs), "question_image_exporter"),
        ]

    snapshot = list(live)
    if args.no_overlap:
        outcomes = gpu_lane(snapshot) + cpu_lane(snapshot)
    else:
        with ThreadPoolExecutor(max_workers=2) as lanes:
            futures = [lanes.submit(gpu_lane, snapshot), lanes.submit(cpu_lane, snapshot)]
            outcomes = [outcome for future in futures for outcome in future.result()]
    for newly_failed, phase in outcomes:
        drop(newly_failed, phase)

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
