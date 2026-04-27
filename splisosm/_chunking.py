"""Shared response-column chunking helpers for SPLISOSM tests."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

_DEFAULT_CHUNK_BUDGET = 2 * (1 << 30)
_DEFAULT_RESPONSE_COLUMN_CAP = 32


def _resolve_n_jobs(n_jobs: int) -> int:
    """Return a positive job count from a joblib-style ``n_jobs`` value."""
    if n_jobs == 0:
        raise ValueError("`n_jobs` must not be 0.")
    n_cpus = os.cpu_count() or 1
    if n_jobs < 0:
        return max(1, n_cpus + 1 + n_jobs)
    return max(1, int(n_jobs))


def auto_chunk_size(
    n_observations: int,
    *,
    backend: Literal["np", "fft"] = "np",
    n_jobs: int = 1,
    dtype_bytes: int = 8,
    memory_budget: int = _DEFAULT_CHUNK_BUDGET,
    max_columns: int = _DEFAULT_RESPONSE_COLUMN_CAP,
) -> int:
    """Return an automatic response-column chunk cap.

    Parameters
    ----------
    n_observations
        Number of spots for the NP backend or grid cells for the FFT backend.
    backend
        Backend used by the spatial variability test.  ``"np"`` estimates
        dense RHS and kernel-product buffers; ``"fft"`` also accounts for
        complex Fourier work arrays.
    n_jobs
        Joblib-style worker count.  The default budget is interpreted per
        worker, so an 8-worker run targets roughly 16 GiB aggregate live
        memory.
    dtype_bytes
        Bytes per real-valued input element.
    memory_budget
        Per-worker live-memory budget in bytes.  Defaults to 2 GiB.
    max_columns
        Hard performance cap for response columns/channels.  SPLISOSM uses
        ``32`` by default for both NP and FFT SV tests.

    Returns
    -------
    int
        Positive response-column cap.  Genes are never split across chunks;
        a single gene with more columns than this cap should be processed as
        a singleton chunk.
    """
    if backend not in {"np", "fft"}:
        raise ValueError("`backend` must be either 'np' or 'fft'.")
    if n_observations < 1:
        raise ValueError("`n_observations` must be positive.")
    if dtype_bytes < 1:
        raise ValueError("`dtype_bytes` must be positive.")
    if memory_budget < 1:
        raise ValueError("`memory_budget` must be positive.")
    if max_columns < 1:
        raise ValueError("`max_columns` must be positive.")

    _resolve_n_jobs(n_jobs)  # validation; budget is intentionally per worker.
    if backend == "np":
        bytes_per_column = 3 * int(n_observations) * int(dtype_bytes)
    else:
        # Input + complex spectrum + power/weighted temporaries.  The exact
        # scipy.fft workspace is backend dependent, so keep this conservative.
        bytes_per_column = 6 * int(n_observations) * int(dtype_bytes)

    memory_limited = max(1, int(memory_budget) // max(1, bytes_per_column))
    return max(1, min(int(max_columns), memory_limited))


def resolve_chunk_size(
    chunk_size: int | Literal["auto"],
    *,
    n_observations: int,
    backend: Literal["np", "fft"],
    n_jobs: int = 1,
    dtype_bytes: int = 8,
) -> int:
    """Resolve a user chunk-size argument to a positive column cap."""
    if chunk_size == "auto":
        return auto_chunk_size(
            n_observations,
            backend=backend,
            n_jobs=n_jobs,
            dtype_bytes=dtype_bytes,
        )
    if not isinstance(chunk_size, int):
        raise TypeError("`chunk_size` must be a positive integer or 'auto'.")
    if chunk_size < 1:
        raise ValueError("`chunk_size` must be positive.")
    return int(chunk_size)


def pack_gene_chunks(widths: Sequence[int], column_cap: int) -> list[list[int]]:
    """Pack whole genes into chunks under a response-column cap.

    A gene whose width exceeds ``column_cap`` is emitted as a singleton chunk.
    """
    if column_cap < 1:
        raise ValueError("`column_cap` must be positive.")

    chunks: list[list[int]] = []
    current: list[int] = []
    current_width = 0

    for idx, width in enumerate(widths):
        width_i = max(1, int(width))
        if not current:
            current = [idx]
            current_width = width_i
            continue

        if current_width + width_i <= column_cap:
            current.append(idx)
            current_width += width_i
        else:
            chunks.append(current)
            current = [idx]
            current_width = width_i

    if current:
        chunks.append(current)
    return chunks
