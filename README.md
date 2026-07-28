# A-Level Question Extraction Pipeline

Turns an exam paper PDF into a structured, queryable question database, then
answers "explain this" / "grade my answer" via the Claude API.

Each phase is an independent script. A phase reads plain JSON files written
by the phase(s) before it and writes its own JSON — nothing is hardcoded
between them, so any phase's internals (which OCR model, which layout
detector) can be swapped without touching the others.

## Setup (one-time)

Dependencies are already installed in this environment. If setting up fresh:

```bash
pip install -r requirements.txt
```

## Running the pipeline

Run these from this directory, in order. Replace `9702_w25_qp_12.pdf` with
your own PDF's filename if different — every command takes the PDF as its
first argument.

```bash
PDF=9702_w25_qp_12.pdf
```

### Phase 1 — Extract text, fonts, coordinates, embedded images

```bash
python3 extract_pdf.py "$PDF" --output-dir output/extracted
```
Output: `output/extracted/extraction.json` (all pages), `output/extracted/questions.json`
(spans grouped into questions by number), `output/extracted/images/` (embedded raster images).

### Phase 2 — Detect page layout (text/image/formula/table/header/footer regions)

```bash
python3 layout_detection.py "$PDF" --output-dir output/layout --backend ppstructure
```
Swap `--backend doclayout_yolo` to use the other detector — same output shape either way.
Output: `output/layout/layout.json`.

### Phase 3 — Merge: assign spans/images to layout regions

```bash
python3 merge_layout.py --extracted output/extracted --layout output/layout --output-dir output/merged
```
Output: `output/merged/merged.json`.

### Phase 4 — Parse merged blocks into questions

```bash
python3 question_parser.py --merged output/merged --output-dir output/questions
```
Output: `output/questions/questions.json` (text, page range, and references to
each question's image/table/formula regions).

### Phase 5 — OCR formula regions to LaTeX

```bash
python3 formula_extractor.py "$PDF" --merged output/merged --output-dir output/formulas
```
Output: `output/formulas/formulas.json`, `output/formulas/crops/` (cropped images).

### Phase 6 — Export image regions as files

```bash
python3 image_exporter.py "$PDF" --merged output/merged --extracted output/extracted --output-dir output/images
```
Output: `output/images/images.json`, plus the actual `.png` files.

### Phase 7 — Extract tables into headers/rows

```bash
python3 table_extractor.py "$PDF" --merged output/merged --output-dir output/tables
```
Output: `output/tables/tables.json`.

### Phase 8 — Build final per-question object

```bash
python3 build_questions.py \
  --extracted output/extracted --questions output/questions \
  --formulas output/formulas --images output/images --tables output/tables \
  --output-dir output/built
```
Output: `output/built/built_questions.json` — one object per question with
paper code, page, marks, text, images, tables, formulas, and MCQ options.

### Phase 9 — Tag topics

```bash
python3 topic_classifier.py --built output/built --output-dir output/topics
```
Output: `output/topics/classified_questions.json`.

### Phase 10 — Load into the database

```bash
python3 database.py --classified output/topics/classified_questions.json --db output/questions.db
```
Output: `output/questions.db` (SQLite).

### Phase 11 — Query the database

```bash
python3 retrieval_api.py --db output/questions.db topic Momentum
python3 retrieval_api.py --db output/questions.db marks 1 2
python3 retrieval_api.py --db output/questions.db ref 9702/12/O/N/25 7
python3 retrieval_api.py --db output/questions.db search "momentum collision"
```

### Phase 12 — Explain / grade via the Claude API

Requires `ANTHROPIC_API_KEY` set (not configured in this environment).

```bash
python3 tutor.py explain --db output/questions.db --paper 9702/12/O/N/25 --question 7

python3 tutor.py grade --db output/questions.db --paper 9702/12/O/N/25 --question 4 \
  --answer "8%" --marking-points "correct method shown" "correct final answer"
```
`--marking-points` must be supplied manually — mark scheme PDF extraction
isn't built yet (only question papers are ingested).

## Run it all as one script

```bash
#!/usr/bin/env bash
set -e
PDF=9702_w25_qp_12.pdf

python3 extract_pdf.py "$PDF" --output-dir output/extracted
python3 layout_detection.py "$PDF" --output-dir output/layout --backend ppstructure
python3 merge_layout.py --extracted output/extracted --layout output/layout --output-dir output/merged
python3 question_parser.py --merged output/merged --output-dir output/questions
python3 formula_extractor.py "$PDF" --merged output/merged --output-dir output/formulas
python3 image_exporter.py "$PDF" --merged output/merged --extracted output/extracted --output-dir output/images
python3 table_extractor.py "$PDF" --merged output/merged --output-dir output/tables
python3 build_questions.py --extracted output/extracted --questions output/questions \
  --formulas output/formulas --images output/images --tables output/tables --output-dir output/built
python3 topic_classifier.py --built output/built --output-dir output/topics
python3 database.py --classified output/topics/classified_questions.json --db output/questions.db
```

Save that as `run_pipeline.sh`, `chmod +x run_pipeline.sh`, then `./run_pipeline.sh`.

## Running many papers on a GPU box

`run_pipeline.sh` is paper-major: all phases for one paper, then all phases
for the next, reloading three models every time. `run_batch.py` flips the
loop and runs one phase across every paper before moving to the next, which
is what you want for anything past a handful of PDFs.

```bash
./setup_vm.sh                      # CUDA 12.6; --cuda 128 for Blackwell
./run_batch.py papers/ --output-dir output --device cuda --jobs $(nproc)
```

Three of the eleven phases run on the GPU — 2 (layout), 5 (formula OCR) and
7 (table OCR). The rest are PDF rasterization and JSON joining, which have no
GPU work in them, so they fan out across cores instead.

### Which layout backend

Use `ppstructure` (the default). It's the only one that gives phase 5
anything to do: DocStructBench's `isolate_formula` label means a display
equation set on its own line, and exam papers are mostly inline math and
short fragments, so DocLayout-YOLO tags none of it. Measured on one 9702
paper, same 40 questions out the far end:

| | formula regions | questions with formulas | images | tables |
| --- | --- | --- | --- | --- |
| `ppstructure` | 59 across 8 pages | 12 | 14 | 8 |
| `doclayout_yolo` | 0 | 0 | 14 | 8 |

Both need Torch *and* Paddle installed either way, so there's no dependency
saving in picking one — with `ppstructure` phases 2 and 7 are Paddle and
phase 5 is Torch; with `doclayout_yolo` it's 2 and 5 on Torch, 7 on Paddle.
`--backend doclayout_yolo` is still there and is the faster detector per
page; it's the right choice only if you don't care about formulas.

What keeps the card busy:

- **Batched inference.** Each GPU phase infers a group at a time rather than
  one page or crop per call. Phase 5 matters most here: Pix2Tex is
  autoregressive, so at batch 1 every generated token is its own kernel
  launch and the run is bound by launch latency, not arithmetic. Its crops
  are pooled *across papers* to fill those batches.
- **Overlap.** Rasterization for the next batch runs on a background thread
  while the current one is on the GPU, and phases 5–8 run as two concurrent
  lanes (GPU: 5 and 7, CPU: 6 and 8) since none of the four depends on
  another.
- **fp16** on CUDA for phases 2 and 5.

Tuning knobs, all of which trade VRAM for throughput:

| Flag | Phase | Default |
| --- | --- | --- |
| `--layout-batch-size` | 2 | 8 pages per call |
| `--formula-batch-size` | 5 | 16 crops per decode |
| `--rec-batch-size` | 7 | 16 text lines per pass |
| `--jobs` | CPU phases | half the cores |

Raise them until `nvidia-smi` shows the card saturated or you hit an
out-of-memory error; phase 5 retries a batch that doesn't fit one crop at a
time, so overshooting there costs throughput rather than results.
`--no-overlap` runs phases 5–8 strictly in order,
which is worth trying if the CPU lane is starving the GPU lane of cores.
`--device cuda` fails loudly when CUDA isn't reachable rather than quietly
finishing the batch on CPU.

## Known limitations

- Formula OCR (Pix2Tex) is accurate on question content, noisier on dense
  physics-constants tables.
- MCQ option extraction works when the A/B/C/D labels appear as text; a
  handful of questions use diagram-only or table-only options with no
  extractable text, so `options` comes back empty for those (correctly —
  there's nothing to parse, not a bug).
- Phase 12 needs `ANTHROPIC_API_KEY` and hasn't been run live in this session.
