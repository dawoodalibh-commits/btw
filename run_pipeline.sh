#!/usr/bin/env bash
# Runs all phases (PDF extraction -> layout -> merge -> question parsing ->
# formula/image/table/full-question-image extraction -> build -> topic
# classification -> DB load) for a single PDF, or for every PDF in a folder,
# wiring each phase's output into the next one's input.
#
# Single file: outputs go straight into --output-dir (e.g. output/extracted, ...).
# Folder:      each PDF gets its own subfolder (output/<pdf-stem>/extracted, ...)
#              so papers don't clobber each other, but all of them load into
#              the same shared database at --output-dir/questions.db.
#
# Usage:
#   ./run_pipeline.sh <pdf-or-folder> [--output-dir DIR] [--paper CODE] [--backend BACKEND] [--dpi N]
set -eo pipefail

usage() {
    echo "Usage: $0 <pdf-or-folder> [--output-dir DIR] [--paper CODE] [--backend BACKEND] [--dpi N]" >&2
    exit 1
}

[ $# -ge 1 ] || usage
INPUT="$1"
shift

OUTPUT_DIR="output"
PAPER="unknown"
BACKEND="ppstructure"
DPI_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir) [ $# -ge 2 ] || usage; OUTPUT_DIR="$2"; shift 2 ;;
        --paper) [ $# -ge 2 ] || usage; PAPER="$2"; shift 2 ;;
        --backend) [ $# -ge 2 ] || usage; BACKEND="$2"; shift 2 ;;
        --dpi) [ $# -ge 2 ] || usage; DPI_ARGS=(--dpi "$2"); shift 2 ;;
        *) usage ;;
    esac
done

[ -e "$INPUT" ] || { echo "Not found: $INPUT" >&2; exit 1; }

# Resolve INPUT and OUTPUT_DIR to absolute paths before cd'ing to the script
# dir, so relative paths given from another working directory keep working.
abspath() {
    if [ -d "$1" ]; then
        (cd "$1" && pwd)
    else
        printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"
    fi
}
INPUT="$(abspath "$INPUT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(abspath "$OUTPUT_DIR")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run() {
    echo "=== $* ==="
    python3 "$@"
}

# Runs phases 1-9 into $2, then loads the result into the shared DB at $3.
# Phases are chained with && so a failure stops the chain and is reported to
# the caller even when run_one is invoked inside an `if` (where set -e is off).
run_one() {
    local pdf="$1" out="$2" paper="$3" db="$4"
    run extract_pdf.py "$pdf" --output-dir "$out/extracted" &&
    run layout_detection.py "$pdf" --output-dir "$out/layout" --backend "$BACKEND" "${DPI_ARGS[@]}" &&
    run merge_layout.py --extracted "$out/extracted" --layout "$out/layout" --output-dir "$out/merged" &&
    run question_parser.py --merged "$out/merged" --output-dir "$out/questions" &&
    run formula_extractor.py "$pdf" --merged "$out/merged" --output-dir "$out/formulas" "${DPI_ARGS[@]}" &&
    run image_exporter.py "$pdf" --merged "$out/merged" --extracted "$out/extracted" --output-dir "$out/images" "${DPI_ARGS[@]}" &&
    run table_extractor.py "$pdf" --merged "$out/merged" --output-dir "$out/tables" "${DPI_ARGS[@]}" &&
    run question_image_exporter.py "$pdf" --merged "$out/merged" --questions "$out/questions" --output-dir "$out/question_images" "${DPI_ARGS[@]}" &&
    run build_questions.py \
        --extracted "$out/extracted" \
        --questions "$out/questions" \
        --formulas "$out/formulas" \
        --images "$out/images" \
        --tables "$out/tables" \
        --question-images "$out/question_images" \
        --output-dir "$out/built" \
        --paper "$paper" &&
    run topic_classifier.py --built "$out/built" --output-dir "$out/topics" &&
    run database.py --classified "$out/topics/classified_questions.json" --db "$db"
}

DB_PATH="$OUTPUT_DIR/questions.db"

if [ -d "$INPUT" ]; then
    shopt -s nullglob
    pdfs=("$INPUT"/*.pdf "$INPUT"/*.PDF)
    [ ${#pdfs[@]} -gt 0 ] || { echo "No PDFs found in $INPUT" >&2; exit 1; }

    count=0
    failed=()
    for pdf in "${pdfs[@]}"; do
        stem="$(basename "$pdf")"
        stem="${stem%.*}"
        echo
        echo "########## [$((count + 1))/${#pdfs[@]}] $stem ##########"
        # build_questions.py auto-detects each paper's real code from its own
        # header/footer text, so --paper here is only the fallback if that fails.
        # A failing paper is skipped (and reported below) instead of killing
        # the rest of the batch.
        if ! run_one "$pdf" "$OUTPUT_DIR/$stem" "$PAPER" "$DB_PATH"; then
            echo "!!! FAILED: $stem (continuing with remaining papers)" >&2
            failed+=("$stem")
        fi
        count=$((count + 1))
    done

    echo
    echo "=== Batch complete: $((count - ${#failed[@]}))/$count papers -> $OUTPUT_DIR (shared DB: $DB_PATH) ==="
    if [ ${#failed[@]} -gt 0 ]; then
        echo "=== Failed papers: ${failed[*]} ===" >&2
        exit 1
    fi
else
    run_one "$INPUT" "$OUTPUT_DIR" "$PAPER" "$DB_PATH"
    echo "=== Pipeline complete -> $OUTPUT_DIR ==="
fi
