"""Convenience imports for reusable SPLISOSM helper functions.

Canonical helper modules are:

* :mod:`splisosm.utils.preprocessing` for data preparation.
* :mod:`splisosm.utils.stats` for standalone tests and p-value helpers.
* :mod:`splisosm.utils.simulation` for synthetic data generation.
"""

from __future__ import annotations

from splisosm.utils.preprocessing import (
    add_ratio_layer,
    auto_chunk_size,
    compute_feature_summaries,
    counts_to_ratios,
    extract_counts_n_ratios,
    extract_gene_level_statistics,
    get_cov_sp,
    prepare_inputs_from_anndata,
)
from splisosm.utils.simulation import (
    simulate_isoform_counts,
    simulate_isoform_counts_single_gene,
)
from splisosm.utils.stats import (
    false_discovery_control,
    run_hsic_gc,
    run_sparkx,
)

__all__ = [
    "get_cov_sp",
    "counts_to_ratios",
    "false_discovery_control",
    "prepare_inputs_from_anndata",
    "add_ratio_layer",
    "extract_counts_n_ratios",
    "compute_feature_summaries",
    "extract_gene_level_statistics",
    "auto_chunk_size",
    "run_sparkx",
    "run_hsic_gc",
    "simulate_isoform_counts_single_gene",
    "simulate_isoform_counts",
]
