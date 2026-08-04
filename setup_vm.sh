#!/usr/bin/env bash
# One-shot environment bring-up for a fresh CUDA VM (vast.ai, RunPod, or a
# bare Ubuntu box with an NVIDIA card).
#
# This exists instead of copying a venv between machines because the two GPU
# frameworks the pipeline needs -- Torch (phase 2 doclayout_yolo, phase 5
# pix2tex) and PaddlePaddle (phase 7 tables, phase 2 ppstructure) -- share the
# nvidia-* wheel namespace and quietly break each other when installed in the
# wrong order. The three rules that matter, all encoded below:
#
#   1. paddlepaddle and paddlepaddle-gpu are separate distributions that
#      install the same paddle/ directory, and pip reports no conflict. Only
#      ever have one installed, and uninstall the CPU build *before* the GPU
#      build lands -- uninstalling it afterwards deletes files the GPU build
#      is now relying on, leaving a half-populated tree that pip still
#      believes is healthy.
#   2. Torch and Paddle must sit on the same CUDA generation. The cu12 and
#      cu13 nvidia-* wheels install to identical paths (nvidia/nccl/lib/
#      libnccl.so.2 and friends), so mixing generations overwrites libraries
#      in place and surfaces as "undefined symbol: ncclCommResume" on
#      `import torch` -- a symptom no amount of LD_LIBRARY_PATH tweaking fixes,
#      because the path was never wrong, the file at it was.
#   3. Torch goes in last. Its nvidia-* pins are the stricter of the two, and
#      Paddle tolerates newer CUDA runtime libs better than Torch tolerates
#      older ones.
#
# Safe to re-run: every step is idempotent, and the verification at the end
# exits non-zero if either framework fails to reach the GPU.
#
# Usage:
#   ./setup_vm.sh                 # CUDA 12.6, into the active venv (or ./.venv)
#   ./setup_vm.sh --cuda 128      # Blackwell (sm_100 / sm_120) needs 12.8+
#   ./setup_vm.sh --clean         # wipe torch/paddle/nvidia wheels first
#   ./setup_vm.sh --skip-models   # don't prefetch model weights
#   ./setup_vm.sh --skip-apt      # no root, or system deps already present
set -eo pipefail

CUDA="126"
CLEAN=0
SKIP_MODELS=0
SKIP_APT=0

usage() {
    echo "Usage: $0 [--cuda 126|128] [--clean] [--skip-models] [--skip-apt] [--venv DIR]" >&2
    exit 1
}

VENV_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda) [ $# -ge 2 ] || usage; CUDA="$2"; shift 2 ;;
        --venv) [ $# -ge 2 ] || usage; VENV_DIR="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --skip-apt) SKIP_APT=1; shift ;;
        *) usage ;;
    esac
done

case "$CUDA" in
    126|128) ;;
    *) echo "Unsupported --cuda $CUDA (expected 126 or 128)" >&2; exit 1 ;;
esac

TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA}"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

step() { echo; echo "=== $* ==="; }

# cu12 and cu13 nvidia-* wheels are separate distributions that install to
# identical paths, so pip reports no conflict while their files overwrite each
# other. Every --cuda value here is a cu12 one, so any cu13 wheel left behind
# by an earlier run has to go before Torch's own pins land on top of it.
purge_foreign_cuda_wheels() {
    local stale
    stale="$($PIP list --format=freeze 2>/dev/null | grep -iE '^nvidia[-_].*[-_]cu13' | cut -d= -f1 || true)"
    if [ -n "$stale" ]; then
        echo "Removing cu13 wheels that would overwrite the cu${CUDA} ones:"
        echo "$stale" | sed 's/^/  /'
        # shellcheck disable=SC2086
        $PIP uninstall -y -q $stale 2>/dev/null || true
    fi
}

# --- GPU / driver sanity -----------------------------------------------------
step "Checking GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found -- this script targets CUDA boxes. Install the" >&2
    echo "NVIDIA driver, or use the CPU install path in requirements.txt." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

# CUDA 12.6 wheels only carry kernels up to sm_90 (Hopper). A Blackwell card
# imports fine and then dies at first inference with "no kernel image is
# available for execution on the device", so catch it here instead.
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if [ -n "$COMPUTE_CAP" ] && [ "$CUDA" = "126" ]; then
    if awk -v c="$COMPUTE_CAP" 'BEGIN { exit !(c >= 10.0) }'; then
        echo "Compute capability $COMPUTE_CAP needs CUDA 12.8+; re-run with --cuda 128." >&2
        exit 1
    fi
fi

# --- System packages ---------------------------------------------------------
# OpenCV comes in via both PaddleOCR and DocLayout-YOLO and dlopens libGL at
# import time, which headless images don't ship. `file` is what
# download_papers.sh uses for its PDF mime check.
if [ "$SKIP_APT" = "0" ] && command -v apt-get >/dev/null 2>&1; then
    step "Installing system packages"
    # -E so the noninteractive settings below survive into the sudo
    # environment; already-root needs no sudo and keeps them by inheritance.
    SUDO=""
    [ "$(id -u)" = "0" ] || SUDO="sudo -E"
    # -y answers apt's own prompts but not debconf's. Without these two a
    # minimal image can stop dead on tzdata's timezone menu, or on 22.04's
    # needrestart "which services should be restarted?" dialog -- and -qq
    # hides the prompt, so it presents as a hang with no output rather than
    # as a question. Neither is recoverable over a detached SSH session.
    export DEBIAN_FRONTEND=noninteractive
    export NEEDRESTART_MODE=a
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq python3-venv python3-pip git curl file libglib2.0-0
    # Renamed between Ubuntu releases; whichever exists is the one we want.
    $SUDO apt-get install -y -qq libgl1 || $SUDO apt-get install -y -qq libgl1-mesa-glx
fi

# --- Virtualenv --------------------------------------------------------------
# Rented GPU images usually boot with a venv already active (vast.ai puts one
# at /venv/main); reuse it rather than nesting another one inside it.
step "Preparing Python environment"
if [ -n "$VENV_DIR" ]; then
    [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using already-active venv: $VIRTUAL_ENV"
else
    [ -d .venv ] || python3 -m venv .venv
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi

PIP="python -m pip"
# Two of the wheels below are ~1-2 GB and one of them comes from Baidu object
# storage in Beijing, which from a rented box in the US or EU routinely drops
# to a crawl or stalls outright mid-transfer. pip's stock 15s timeout / 5
# retries gives up on a big slow download that would have finished; these
# survive the stall instead. Set as env vars so every pip call inherits them.
export PIP_DEFAULT_TIMEOUT=60
export PIP_RETRIES=10
$PIP install -q -U pip wheel
python -c "import sys; print('python', sys.version.split()[0], 'at', sys.prefix)"

# --- Optional clean slate ----------------------------------------------------
# For recovering a venv that already has both CUDA generations in it. The
# rmtree matters: once cu12 and cu13 wheels have overwritten each other,
# orphaned files are in no distribution's RECORD and no pip uninstall will
# ever remove them, so a plain reinstall just layers over the debris.
if [ "$CLEAN" = "1" ]; then
    step "Removing existing torch/paddle/CUDA wheels"
    $PIP uninstall -y -q torch torchvision paddlepaddle paddlepaddle-gpu 2>/dev/null || true
    NVIDIA_PKGS="$($PIP list --format=freeze 2>/dev/null | grep -iE '^nvidia[-_]' | cut -d= -f1 || true)"
    if [ -n "$NVIDIA_PKGS" ]; then
        # shellcheck disable=SC2086
        $PIP uninstall -y -q $NVIDIA_PKGS 2>/dev/null || true
    fi
    python - <<'PY'
import shutil, site, os
for root in site.getsitepackages():
    for leftover in ("paddle", "nvidia"):
        path = os.path.join(root, leftover)
        if os.path.isdir(path):
            print("removing orphaned", path)
            shutil.rmtree(path, ignore_errors=True)
PY
fi

# --- Python dependencies -----------------------------------------------------
step "Installing requirements.txt"
$PIP install -r requirements.txt

# requirements.txt deliberately omits Paddle, but an older checkout may still
# pin the CPU build. Drop it here -- while it still solely owns paddle/ -- so
# the GPU wheel below lands on clean ground.
if $PIP show paddlepaddle >/dev/null 2>&1; then
    step "Removing CPU paddlepaddle before installing the GPU build"
    $PIP uninstall -y -q paddlepaddle
fi

# Deliberately not -q. This wheel is ~2 GB and is served from Baidu object
# storage in Beijing; at the 1-5 MB/s a rented box outside China typically
# gets, it is 7-30 minutes of work. Quiet, that is half an hour of a script
# that looks wedged, and the reflex is to Ctrl-C it -- so let pip draw its
# progress bar and be visibly alive.
step "Installing paddlepaddle-gpu (cu${CUDA})"
echo "~2 GB from paddlepaddle.org.cn -- expect 7-30 min depending on distance."
$PIP install --progress-bar on paddlepaddle-gpu -i "$PADDLE_INDEX"

# Last, so its stricter nvidia-* pins win the resolution.
#
# The uninstall is not optional. requirements.txt pulls torch in as a pix2tex
# dependency, from default PyPI, whose Linux wheel bundles whatever CUDA
# generation upstream currently ships. A plain `pip install torch` against
# this index then reports "Requirement already satisfied" and leaves that
# build in place -- while still installing torchvision from the index, because
# nothing has pulled *that* in yet. The result is the two halves of Torch on
# two CUDA generations, which is rule 2 at the top of this file breaking, and
# it looks from the outside like a clean install.
step "Installing torch (cu${CUDA})"
$PIP uninstall -y -q torch torchvision 2>/dev/null || true
purge_foreign_cuda_wheels
$PIP install torch torchvision --index-url "$TORCH_INDEX"

cat <<'EOF'

Note: pip has probably just complained that paddlepaddle-gpu requires older
nvidia-cudnn / nvidia-nccl / nvidia-cusparselt / nvidia-cuda-nvrtc than the
ones Torch pulled in. That is this step working as intended -- both frameworks
pin those wheels exactly, their pins disagree, and rule 3 above is the
decision that Torch wins. Paddle loads those libraries by soname
(libcudnn.so.9, libnccl.so.2, libcusparseLt.so.0), which the version bumps
don't change, so it keeps working against the newer ones.

Do not "fix" it by reinstalling Paddle afterwards -- that drags the older
wheels back over Torch's, which is the direction that actually breaks. The
GPU check at the end of this script is what decides whether this held.
EOF

# --- Model weights -----------------------------------------------------------
# Prefetched serially so the first batch doesn't stall on three downloads, and
# so parallel workers can't race on a half-written cache entry.
if [ "$SKIP_MODELS" = "0" ]; then
    step "Prefetching model weights"
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    python - <<'PY'
from huggingface_hub import hf_hub_download
print("doclayout-yolo:", hf_hub_download(
    "juliozhao/DocLayout-YOLO-DocStructBench",
    "doclayout_yolo_docstructbench_imgsz1024.pt"))
PY
    python -c "from pix2tex.cli import LatexOCR; LatexOCR(); print('pix2tex: ok')"
    python - <<'PY'
from paddleocr import PaddleOCR, LayoutDetection
# Constructed exactly as table_extractor.py does: the orientation and
# unwarping sub-models are switched off there, and prefetching the bare
# defaults would download three models the pipeline never loads.
PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
LayoutDetection()
print("paddleocr: ok")
PY
fi

# --- Verification ------------------------------------------------------------
# pip list will happily report a healthy GPU install that cannot import, and an
# import that succeeds still says nothing about whether a kernel will launch:
# a build whose kernels don't cover this card imports fine, reports
# cuda=True, and only dies at the first real launch with "no kernel image is
# available for execution on the device". So each framework actually runs
# something on the GPU here.
step "Verifying GPU access"
python - <<'PY'
import sys

# Which interpreter this all landed in. Worth printing because everything
# below passes here and then fails in your shell if the two disagree --
# which is what happens when this script isn't run from the venv you later
# work in (a sudo/non-login invocation drops VIRTUAL_ENV, so the venv block
# above quietly builds ./.venv instead of reusing the active one).
print(f"python {sys.version.split()[0]:<16} at {sys.prefix}\n")

ok = True

try:
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "-"
    if cuda:
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    print(f"torch  {torch.__version__:<16} cuda={cuda}  {name}")
    ok &= cuda
except Exception as exc:
    print(f"torch  FAILED: {exc}")
    ok = False

try:
    import paddle
    device = paddle.device.get_device()
    on_gpu = device.startswith("gpu")
    if on_gpu:
        float(paddle.matmul(paddle.randn([64, 64]), paddle.randn([64, 64])).sum())
    print(f"paddle {paddle.__version__:<16} device={device}  compiled_with_cuda={paddle.is_compiled_with_cuda()}")
    ok &= on_gpu
except Exception as exc:
    print(f"paddle FAILED: {exc}")
    ok = False

if not ok:
    print("\nOne or both frameworks are not on the GPU.")
    print("'undefined symbol': the cu12/cu13 wheels have overwritten each")
    print("  other -- re-run with --clean.")
    print("'no kernel image is available': this build has no kernels for this")
    print("  card -- re-run with --cuda 128.")
    sys.exit(1)

print("\nBoth frameworks are on the GPU.")
PY

step "Setup complete"
cat <<EOF
Everything above went into: $(python -c 'import sys; print(sys.prefix)')
If that is not the environment your shell is in, activate it first --
otherwise the pipeline reports "No module named 'paddle'" despite this
script having just verified paddle on the GPU:
  source $(python -c 'import sys; print(sys.prefix)')/bin/activate

Smoke test one paper:
  ./run_batch.py 9702_w25_qp_12.pdf --output-dir output-smoke --device cuda --backend doclayout_yolo

Then fetch papers and run the batch (--jobs = physical cores, not threads):
  ./download_papers.sh --start-year 15 --end-year 24 --subject 9702 --out-dir papers
  ./run_batch.py papers/ --output-dir output --device cuda --backend doclayout_yolo --jobs \$(nproc)

The GPU phases (2 layout, 5 formulas, 7 tables) batch their work; if
\`nvidia-smi\` shows the card idling during a run, raise the batch sizes:
  --layout-batch-size 16 --formula-batch-size 32 --rec-batch-size 32
EOF
