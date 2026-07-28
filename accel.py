"""Shared accelerator plumbing for the GPU-bound phases (2 layout, 5 formulas, 7 tables).

Those three phases each need the same three things, and used to answer them
three slightly different ways:

  * turn a `--device` string into whatever their framework calls that device,
  * fail loudly when CUDA was asked for and isn't actually reachable, rather
    than quietly running the batch on CPU,
  * keep the GPU fed while the CPU rasterizes the next batch out of the PDF.

The third is the one that decides throughput. Every GPU phase alternates
between PyMuPDF rasterization (CPU, and the part that releases the GIL) and
inference. Done inline those two take turns idling each other; run one batch
ahead on a background thread and the GPU has its next batch waiting the
moment it finishes the current one.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable, Iterator, TypeVar

# Accepted --device values across every phase. "auto" picks the best
# accelerator actually present; the rest force a specific one.
DEVICES = ("auto", "cpu", "cuda", "mps")

_T = TypeVar("_T")


def _check_known(requested: str) -> None:
    if requested not in DEVICES:
        raise ValueError(f"Unknown device {requested!r}. Choose from {list(DEVICES)}")


def resolve_torch_device(requested: str) -> str:
    """Resolve a `--device` value to a concrete torch device string.

    An explicitly requested accelerator that isn't available is an error, not
    a silent fall back to CPU. On a rented GPU box that fall back is the
    difference between a twenty-minute batch and an overnight one, and it
    shows up as nothing louder than a slow run -- the whole reason to pass
    `--device cuda` is that CPU is not an acceptable answer.
    """
    import torch

    _check_known(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda, but torch.cuda.is_available() is False. Install the CUDA "
            "build of torch (see setup_vm.sh) or pass --device cpu."
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps, but torch reports no MPS backend. Pass --device cpu.")
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_paddle_device(requested: str) -> str:
    """Resolve a `--device` value to a paddle device string ("gpu" or "cpu").

    Paddle spells CUDA "gpu" and has no Metal backend, so mps can only mean
    cpu here. Whether a GPU is reachable at all is decided by which
    *distribution* is installed (paddlepaddle vs paddlepaddle-gpu) rather
    than by any runtime flag, and the CPU build accepts device="gpu" and then
    runs on CPU anyway -- so, as with torch, an explicit --device cuda against
    a CPU-only build is an error rather than a slow success.
    """
    import paddle

    _check_known(requested)
    has_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if requested == "cuda":
        if not has_gpu:
            raise RuntimeError(
                "--device cuda, but this paddle build cannot reach a GPU "
                f"(compiled_with_cuda={paddle.is_compiled_with_cuda()}). Install paddlepaddle-gpu "
                "(see setup_vm.sh) or pass --device cpu."
            )
        return "gpu"
    if requested in ("cpu", "mps"):
        return "cpu"
    return "gpu" if has_gpu else "cpu"


def describe_torch_device(device: str) -> str:
    """Human-readable device label for the one-line banner each phase prints."""
    if not device.startswith("cuda"):
        return device
    try:
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_gib = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"{device} ({name}, {total_gib:.1f} GiB)"
    except Exception:
        return device


def prefetch(produce: Callable[[], Iterable[_T]], depth: int = 2) -> Iterator[_T]:
    """Yield `produce()`'s items, running it on a background thread.

    `depth` bounds how far ahead the producer may run, so rasterizing a long
    document can't buffer the whole thing into memory. Exceptions raised by
    the producer surface on the consumer side, at the point the failed item
    would have been yielded.
    """
    channel: queue.Queue = queue.Queue(maxsize=max(1, depth))
    done = object()

    def worker() -> None:
        try:
            for item in produce():
                channel.put((item, None))
        except BaseException as exc:  # re-raised on the consumer's thread below
            channel.put((done, exc))
            return
        channel.put((done, None))

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item, exc = channel.get()
        if exc is not None:
            raise exc
        if item is done:
            return
        yield item


def chunked(items: list[_T], size: int) -> Iterator[list[_T]]:
    """Split `items` into consecutive chunks of at most `size`."""
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]
