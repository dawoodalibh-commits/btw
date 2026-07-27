#!/usr/bin/env bash
# Bulk-downloads Cambridge past papers, trying every valid {season}{code}
# combination. Primary source is dynamicpapers.com; if a given paper isn't
# found there, falls back through papacambridge.com, xtrapapers.co, and
# pastpapers.co (in that order) before giving up on it.
#
# Filename pattern: {subject}_{season}{yy}_qp_{code}.pdf
#   season: s (summer), w (winter), m (march)
#   code:   two digits, {paper number}{variant} -- march sittings only ever
#           run as variant 2 (12, 22, 32, ...)
#
# Papers are fetched concurrently (--jobs, default 8). If a source starts
# rate-limiting you, lower --jobs and/or raise --delay.
#
# Usage:
#   ./download_papers.sh --start-year 15 --end-year 24 --out-dir papers
#   ./download_papers.sh --start-year 20 --end-year 24 --subject 9702 --codes 11,12,13 --seasons s,w,m
#   ./download_papers.sh --start-year 15 --end-year 24 --all --out-dir all-papers
#
# --all mode: iterates every A-Level subject in the ALL_SUBJECTS table below
# (scraped from pastpapers.co's own A-Level directory, 2026-07), downloading
# into one subfolder per subject code (<out-dir>/<code>/...). Unless --codes
# is given, it tries papers 1-6 x variants 1-3 for every subject; components
# a subject never ran simply come back MISSING (written to <out-dir>/
# missing.txt rather than dumped on the terminal).
set -eo pipefail

SUBJECT="9702"
START_YEAR=""
END_YEAR=""
CODES=""
SEASONS="s,w,m"
OUT_DIR="papers"
UPLOAD_PATH="2015/09"
DELAY=0
JOBS=8
ALL_MODE=0
BASE_URL="https://dynamicpapers.com/wp-content/uploads"
UA="Mozilla/5.0"

# Backup sources, tried in order after the primary fails:
#
#   1. papacambridge -- a flat file store keyed by filename alone
#      (.../CAIE-pastpapers/upload/<filename>), so it needs no subject slug
#      and works for every subject out of the box.
#   2. xtrapapers -- https://xtrapapers.co/papers/caie/as-and-a-level/
#      <slug-lowercase>/<yyyy>-<jun|nov|mar>/<filename>.pdf/download
#   3. pastpapers.co -- https://pastpapers.co/caie/A-Level/
#      <Slug-Capitalized>/<yyyy>-<May-June|Oct-Nov|March>/<filename>
#      (this direct path serves the raw PDF; the site's /api/file/ endpoint
#      is broken -- it 301s to localhost:3000 -- and missing files come back
#      as HTTP 200 text/html, which the PDF mime check below filters out).
#
# Both slug-based sources derive from one canonical slug in pastpapers.co's
# capitalized form ("Physics-9702"): xtrapapers gets it lowercased. Slugs are
# looked up from ALL_SUBJECTS by subject code; --backup-slug overrides.
PAPACAMBRIDGE_BASE_URL="https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload"
XTRAPAPERS_BASE_URL="https://xtrapapers.co/papers/caie/as-and-a-level"
PASTPAPERSCO_BASE_URL="https://pastpapers.co/caie/A-Level"
BACKUP_SLUG=""

# Every A-Level subject directory listed on pastpapers.co (code is the last
# 4 digits of each slug). Computer-Science-9618 is missing from their listing
# and added by hand -- it still resolves via the flat-store sources.
# Language-and-Literature-8695 was dropped as a duplicate of English-8695.
ALL_SUBJECTS="
Accounting-9706 Afrikaans-8679 Afrikaans-8779 Afrikaans-9679 AICT-9713
Arabic-8680 Arabic-9680 Art-and-Design-9479 Art-and-Design-9704 Biology-9184
Biology-9700 Business-9609 Business-Studies-9707 Chemistry-9185
Chemistry-9701 Chineese-8238 Chinese-8669 Chinese-8681 Chinese-9715
Chinese-9868 Classical-Studies-9274 Computer-Science-9608
Computer-Science-9618 Computing-9691 Design-and-Technology-9705
Design-and-Textiles-9631 Economics-9275 Economics-9708 English-8274
English-8287 English-8695 English-9093 English-9276 English-9695
English-General-Paper-8021 English-Language-8693
Environmental-Management-8291 Food-Studies-9336 French-8276 French-8670
French-9281 French-9716 French-Language-8682 General-Paper-8001
General-Paper-8004 Geography-9278 Geography-9696 German-8683 German-9717
Global-Perspectives-8275 Global-Perspectives-8987
Global-Perspectives-and-Research-9239 Hindi-8675 Hindi-9687
Hindi-Language-8687 Hinduism-8058 Hinduism-9014 Hinduism-9487 History-9279
History-9389 History-9489 History-9697 Information-Technology-9626
Islamic-Studies-9013 Japanese-Language-8281 Law-9084 Marathi-9688
Marathi-Language-8688 Marine-Science-9693 Mathematics-9280 Mathematics-9709
Mathematics-Further-9231 Media-Studies-9607 Music-9385 Music-9483 Music-9703
Physical-Education-9396 Physical-Science-8780 Physics-9277 Physics-9702
Portuguese-8684 Portuguese-9718 Portuguese-Literature-8672 Psychology-9698
Psychology-9990 Sociology-9699 Spanish-8022 Spanish-8278 Spanish-8279
Spanish-8665 Spanish-8673 Spanish-8685 Spanish-9282 Spanish-9719
Spanish-9844 Tamil-8689 Tamil-9689 Telugu-9690 Telugu-Language-8690
Thinking-Skills-9694 Urdu-8686 Urdu-9676 Urdu-9686
"

# Season -> backup-source folder names. Case statements rather than
# associative arrays: macOS ships bash 3.2, which doesn't support `declare -A`.
_season_folder() {
    case "$1" in
        s) echo "jun" ;;
        w) echo "nov" ;;
        m) echo "mar" ;;
    esac
}

_season_folder_ppco() {
    case "$1" in
        s) echo "May-June" ;;
        w) echo "Oct-Nov" ;;
        m) echo "March" ;;
    esac
}

_slug_for_subject() {
    local code="$1" entry
    for entry in $ALL_SUBJECTS; do
        if [ "${entry##*-}" = "$code" ]; then
            echo "$entry"
            return 0
        fi
    done
}

usage() {
    echo "Usage: $0 --start-year YY --end-year YY [--subject CODE | --all] [--codes 11,12,13] [--seasons s,w,m] [--out-dir DIR] [--jobs N] [--delay SECONDS] [--backup-slug Subject-Slug]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --start-year) START_YEAR="$2"; shift 2 ;;
        --end-year) END_YEAR="$2"; shift 2 ;;
        --subject) SUBJECT="$2"; shift 2 ;;
        --all|-all) ALL_MODE=1; shift ;;
        --codes) CODES="$2"; shift 2 ;;
        --seasons) SEASONS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --upload-path) UPLOAD_PATH="$2"; shift 2 ;;
        --delay) DELAY="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --backup-slug) BACKUP_SLUG="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[ -n "$START_YEAR" ] && [ -n "$END_YEAR" ] || usage

# Default codes: paper 1 variants for a single subject; every paper 1-6 x
# variant 1-3 when sweeping all subjects (nonexistent components just turn
# up MISSING).
if [ -z "$CODES" ]; then
    if [ "$ALL_MODE" = "1" ]; then
        CODES="11,12,13,21,22,23,31,32,33,41,42,43,51,52,53,61,62,63"
    else
        CODES="11,12,13"
    fi
fi

if [ "$ALL_MODE" != "1" ] && [ -z "$BACKUP_SLUG" ]; then
    BACKUP_SLUG="$(_slug_for_subject "$SUBJECT")"
fi

mkdir -p "$OUT_DIR"
IFS=',' read -ra CODE_LIST <<< "$CODES"
IFS=',' read -ra SEASON_LIST <<< "$SEASONS"

# Per-line status records from the parallel workers land here; the summary is
# tallied from it after all jobs finish. Appends of short lines with O_APPEND
# are atomic, so concurrent workers don't interleave.
RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

# Downloads $url into $out_path if it resolves to a real PDF.
_try_download() {
    local url="$1" out_path="$2"
    local tmp http_code content_type
    tmp="$(mktemp)"
    http_code=$(curl -s -L -A "$UA" --connect-timeout 10 --max-time 120 \
        -o "$tmp" -w "%{http_code}" "$url" || echo "000")
    content_type=$(file -b --mime-type "$tmp" 2>/dev/null || echo "")
    if [ "$http_code" = "200" ] && [ "$content_type" = "application/pdf" ]; then
        mv "$tmp" "$out_path"
        return 0
    fi
    rm -f "$tmp"
    return 1
}

# Handles one paper end-to-end (primary source, then backups). Runs in a
# worker subshell under xargs, so it reports via RESULTS_FILE rather than
# shared shell variables. Takes one packed task: subject|slug|season|yy|code
# (slug may be empty -- then only the slug-free papacambridge backup runs).
_process_one() {
    local subject slug season yy code
    IFS='|' read -r subject slug season yy code <<< "$1"

    local fname="${subject}_${season}${yy}_qp_${code}.pdf"
    local dir="$OUT_DIR"
    local label="$fname"
    if [ "$ALL_MODE" = "1" ]; then
        dir="$OUT_DIR/$subject"
        label="$subject/$fname"
    fi
    local out_path="$dir/$fname"

    if [ -f "$out_path" ]; then
        echo "SKIP (exists)   $label"
        echo "SKIP $label" >> "$RESULTS_FILE"
        return 0
    fi

    if _try_download "$BASE_URL/$UPLOAD_PATH/$fname" "$out_path"; then
        echo "OK              $label"
        echo "OK $label" >> "$RESULTS_FILE"
        return 0
    fi
    [ "$DELAY" != "0" ] && sleep "$DELAY"

    local slug_lc entry source_name backup_url
    local backup_urls=("papacambridge|$PAPACAMBRIDGE_BASE_URL/${fname}")
    if [ -n "$slug" ]; then
        slug_lc="$(echo "$slug" | tr '[:upper:]' '[:lower:]')"
        backup_urls+=("xtrapapers|$XTRAPAPERS_BASE_URL/$slug_lc/20${yy}-$(_season_folder "$season")/${fname}/download")
        backup_urls+=("pastpapers.co|$PASTPAPERSCO_BASE_URL/$slug/20${yy}-$(_season_folder_ppco "$season")/${fname}")
    fi

    for entry in "${backup_urls[@]}"; do
        source_name="${entry%%|*}"
        backup_url="${entry#*|}"
        if _try_download "$backup_url" "$out_path"; then
            echo "OK ($source_name)  $label"
            echo "OK $label" >> "$RESULTS_FILE"
            return 0
        fi
        [ "$DELAY" != "0" ] && sleep "$DELAY"
    done

    echo "MISSING         $label"
    echo "MISSING $label" >> "$RESULTS_FILE"
    return 0
}

export OUT_DIR UPLOAD_PATH DELAY BASE_URL UA ALL_MODE \
    PAPACAMBRIDGE_BASE_URL XTRAPAPERS_BASE_URL PASTPAPERSCO_BASE_URL \
    RESULTS_FILE
export -f _try_download _process_one _season_folder _season_folder_ppco

# One subject in normal mode, the whole table in --all mode.
if [ "$ALL_MODE" = "1" ]; then
    TARGETS=""
    for entry in $ALL_SUBJECTS; do
        TARGETS="$TARGETS ${entry##*-}|$entry"
        mkdir -p "$OUT_DIR/${entry##*-}"
    done
else
    TARGETS="$SUBJECT|$BACKUP_SLUG"
fi

# Emit one packed "subject|slug|season|yy|code" task per paper and fan them
# out across $JOBS concurrent workers.
for target in $TARGETS; do
    subject="${target%%|*}"
    slug="${target#*|}"
    for yy in $(seq "$START_YEAR" "$END_YEAR"); do
        yy=$(printf "%02d" "$yy")
        for season in "${SEASON_LIST[@]}"; do
            for code in "${CODE_LIST[@]}"; do
                # March sittings only ever run as variant 2.
                if [ "$season" = "m" ]; then
                    case "$code" in *2) ;; *) continue ;; esac
                fi
                printf '%s|%s|%s|%s|%s\n' "$subject" "$slug" "$season" "$yy" "$code"
            done
        done
    done
done | xargs -P "$JOBS" -n 1 bash -c '_process_one "$@"' _

downloaded=$(grep -c '^OK ' "$RESULTS_FILE" || true)
skipped=$(grep -c '^SKIP ' "$RESULTS_FILE" || true)
missing_count=$(grep -c '^MISSING ' "$RESULTS_FILE" || true)

echo
echo "=== Done: $downloaded downloaded, $skipped already present, $missing_count missing -> $OUT_DIR ==="
if [ "$ALL_MODE" != "1" ] && [ -z "$BACKUP_SLUG" ]; then
    echo "(No slug known for subject $SUBJECT -- only the slug-free papacambridge backup was tried. Pass --backup-slug <Name>-$SUBJECT to also enable xtrapapers and pastpapers.co.)"
fi
if [ "$missing_count" -gt 0 ]; then
    # An --all sweep probes components many subjects never ran, so the
    # missing list is long and expected -- file it instead of printing it.
    if [ "$ALL_MODE" = "1" ] || [ "$missing_count" -gt 40 ]; then
        grep '^MISSING ' "$RESULTS_FILE" | sed 's/^MISSING //' | sort > "$OUT_DIR/missing.txt"
        echo "Missing list written to $OUT_DIR/missing.txt"
    else
        echo "Missing:"
        grep '^MISSING ' "$RESULTS_FILE" | sed 's/^MISSING /  /'
    fi
fi
