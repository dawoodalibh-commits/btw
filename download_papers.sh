#!/usr/bin/env bash
# Bulk-downloads Cambridge past papers for a subject over a range of years,
# trying every valid {season}{code} combination. Primary source is
# dynamicpapers.com; if a given paper isn't found there, falls back to
# xtrapapers.co before giving up on it.
#
# Filename pattern: {subject}_{season}{yy}_qp_{code}.pdf
#   season: s (summer), w (winter), m (march)
#   code:   11, 12, 13 -- march sittings only ever use code 12
#
# Usage:
#   ./download_papers.sh --start-year 15 --end-year 24 --out-dir papers
#   ./download_papers.sh --start-year 20 --end-year 24 --subject 9702 --codes 11,12,13 --seasons s,w,m
set -eo pipefail

SUBJECT="9702"
START_YEAR=""
END_YEAR=""
CODES="11,12,13"
SEASONS="s,w,m"
OUT_DIR="papers"
UPLOAD_PATH="2015/09"
DELAY=1
BASE_URL="https://dynamicpapers.com/wp-content/uploads"
UA="Mozilla/5.0"

# Backup source. Its URL is https://xtrapapers.co/papers/caie/as-and-a-level/
# <subject-slug>/<yyyy>-<season-folder>/<filename>.pdf/download -- the
# "as-and-a-level" prefix and trailing "/download" are fixed, but the
# subject-name slug ("physics-9702", "chemistry-9701", ...) isn't derivable
# from the numeric subject code alone, so it must be passed explicitly for
# any subject other than the 9702 default.
BACKUP_BASE_URL="https://xtrapapers.co/papers/caie/as-and-a-level"
BACKUP_SLUG=""

# Season -> backup-source folder name. A case statement rather than an
# associative array: macOS ships bash 3.2, which doesn't support `declare -A`.
_season_folder() {
    case "$1" in
        s) echo "jun" ;;
        w) echo "nov" ;;
        m) echo "mar" ;;
    esac
}

usage() {
    echo "Usage: $0 --start-year YY --end-year YY [--subject CODE] [--codes 11,12,13] [--seasons s,w,m] [--out-dir DIR] [--delay SECONDS] [--backup-slug SUBJECT-SLUG]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --start-year) START_YEAR="$2"; shift 2 ;;
        --end-year) END_YEAR="$2"; shift 2 ;;
        --subject) SUBJECT="$2"; shift 2 ;;
        --codes) CODES="$2"; shift 2 ;;
        --seasons) SEASONS="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --upload-path) UPLOAD_PATH="$2"; shift 2 ;;
        --delay) DELAY="$2"; shift 2 ;;
        --backup-slug) BACKUP_SLUG="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[ -n "$START_YEAR" ] && [ -n "$END_YEAR" ] || usage

# Known subject-code -> slug mappings for the backup source. Add more here as
# you confirm them; --backup-slug always overrides this table.
if [ -z "$BACKUP_SLUG" ] && [ "$SUBJECT" = "9702" ]; then
    BACKUP_SLUG="physics-9702"
fi

mkdir -p "$OUT_DIR"
IFS=',' read -ra CODE_LIST <<< "$CODES"
IFS=',' read -ra SEASON_LIST <<< "$SEASONS"

# Downloads $url into $out_path if it resolves to a real PDF. Echoes 0/1 so
# the caller can tell success from failure without parsing stdout.
_try_download() {
    local url="$1" out_path="$2"
    local tmp http_code content_type
    tmp="$(mktemp)"
    http_code=$(curl -s -L -A "$UA" -o "$tmp" -w "%{http_code}" "$url" || echo "000")
    content_type=$(file -b --mime-type "$tmp" 2>/dev/null || echo "")
    if [ "$http_code" = "200" ] && [ "$content_type" = "application/pdf" ]; then
        mv "$tmp" "$out_path"
        return 0
    fi
    rm -f "$tmp"
    return 1
}

downloaded=0
skipped=0
missing=()

for yy in $(seq "$START_YEAR" "$END_YEAR"); do
    yy=$(printf "%02d" "$yy")
    for season in "${SEASON_LIST[@]}"; do
        for code in "${CODE_LIST[@]}"; do
            # March sittings only ever run as variant 12.
            if [ "$season" = "m" ] && [ "$code" != "12" ]; then
                continue
            fi

            fname="${SUBJECT}_${season}${yy}_qp_${code}.pdf"
            out_path="$OUT_DIR/$fname"

            if [ -f "$out_path" ]; then
                echo "SKIP (exists)   $fname"
                skipped=$((skipped + 1))
                continue
            fi

            primary_url="$BASE_URL/$UPLOAD_PATH/$fname"
            if _try_download "$primary_url" "$out_path"; then
                echo "OK              $fname"
                downloaded=$((downloaded + 1))
                sleep "$DELAY"
                continue
            fi
            sleep "$DELAY"

            found=false
            if [ -n "$BACKUP_SLUG" ]; then
                season_folder="$(_season_folder "$season")"
                backup_url="$BACKUP_BASE_URL/$BACKUP_SLUG/20${yy}-${season_folder}/${fname}/download"
                if _try_download "$backup_url" "$out_path"; then
                    echo "OK (backup)     $fname"
                    downloaded=$((downloaded + 1))
                    found=true
                fi
                sleep "$DELAY"
            fi

            if [ "$found" = false ]; then
                echo "MISSING         $fname"
                missing+=("$fname")
            fi
        done
    done
done

echo
echo "=== Done: $downloaded downloaded, $skipped already present, ${#missing[@]} missing -> $OUT_DIR ==="
if [ -z "$BACKUP_SLUG" ]; then
    echo "(No --backup-slug set for subject $SUBJECT -- backup source was skipped. Pass --backup-slug <name>-$SUBJECT to enable it.)"
fi
if [ "${#missing[@]}" -gt 0 ]; then
    echo "Missing:"
    printf '  %s\n' "${missing[@]}"
fi
