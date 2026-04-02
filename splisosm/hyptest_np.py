"""Non-parametric hypothesis tests for spatial isoform usage."""

from __future__ import annotations

import warnings
import re
from typing import Any, Optional, Union, Literal
from scipy.stats import ttest_ind, combine_pvalues, norm as _norm_dist
import numpy as np
import scipy.sparse
import pandas as pd
import torch
from tqdm.auto import tqdm
from anndata import AnnData

from splisosm.utils import (
    counts_to_ratios,
    false_discovery_control,
    prepare_inputs_from_anndata,
    run_sparkx,
)
from splisosm.kernel import SpatialCovKernel
from splisosm.likelihood import liu_sf
from splisosm.kernel_gpr import (
    linear_hsic_test,
    fit_kernel_gpr,
    make_kernel_gpr,
    _DEFAULT_GPR_CONFIGS,
)

__all__ = [
    "linear_hsic_test",
    "fit_kernel_gpr",
    "SplisosmNP",
]


def _calc_ttest_differential_usage(
    data, groups, combine_pval=True, combine_method="tippett"
):
    """Calculate the two-sample t-test statistic for differential usage.

    The t-test is applied to each isoform independently and combined if combine_pval is True.

    Parameters
    ----------
    data : torch.Tensor
        Shape (n_spots, n_isos), the observed isoform ratios for a given gene.
    groups : torch.Tensor or scipy.sparse.spmatrix
        Shape ``(n_spots,)`` or ``(n_spots, 1)``, the binary group labels for
        each spot.  When a scipy sparse matrix is provided (e.g. a column from
        a one-hot-encoded design matrix), the nonzero rows are taken as group 1
        and zero rows as group 0, avoiding full densification of the column.
    combine_pval : bool, optional
        Whether to combine p-values across isoforms.
    combine_method : str, optional
        The method to combine p-values. See scipy.stats.combine_pvalues() for more details.

    Returns
    -------
    stats : float or numpy scalar
        Combined test statistic (scalar when combine_pval=True).
    pval : float or numpy scalar
        P-value.
    """
    if scipy.sparse.issparse(groups):
        # Sparse column (n, 1): nonzero rows belong to group 1, zeros to group 0.
        # This avoids creating a full dense boolean mask.
        groups_csr = groups.tocsr()
        g1_idx = groups_csr.nonzero()[0]  # row indices where value is nonzero
        n = groups_csr.shape[0]
        g0_mask = np.ones(n, dtype=bool)
        g0_mask[g1_idx] = False
        t1 = data[g1_idx]  # (k, n_isos)
        t2 = data[g0_mask]  # (n - k, n_isos)
    else:
        # Dense path: groups is a 1-D tensor or array.
        _g = torch.unique(
            groups
            if isinstance(groups, torch.Tensor)
            else torch.from_numpy(np.asarray(groups))
        )
        if len(_g) > 2:
            raise ValueError(
                "More than two groups detected. Only two are allowed for the two-sample t-test."
            )
        t1 = data[groups == _g[0], :]  # (k, n_isos)
        t2 = data[groups == _g[1], :]  # (n_spots - k, n_isos)

    stats, pval = ttest_ind(t1, t2, axis=0, nan_policy="omit")

    # combine p-values across isoforms
    if combine_pval:
        stats, pval = combine_pvalues(pval, method=combine_method)

    return stats, pval


class SplisosmNP:
    """Non-parametric spatial isoform statistical model.

    Examples
    --------
    Setup data:

    >>> from splisosm import SplisosmNP
    >>> import torch
    >>> # Simulate data for 10 genes with different number of isoforms
    >>> data_3_iso = [torch.randint(low=0, high=5, size=(100, 3)) for _ in range(5)]  # 5 genes with 3 isoforms
    >>> data_4_iso = [torch.randint(low=0, high=5, size=(100, 4)) for _ in range(5)]  # 5 genes with 4 isoforms
    >>> data = data_3_iso + data_4_iso
    >>> coordinates = torch.rand(100, 2)  # 100 spots with 2D coordinates
    >>> design_mtx = torch.rand(100, 2)  # 100 spots with 2 covariates

    Spatial variability test:

    >>> model = SplisosmNP()
    >>> model.setup_data(data, coordinates)
    >>> model.test_spatial_variability(method='hsic-ir')
    >>> sv_results = model.get_formatted_test_results('sv')
    >>> print(sv_results.head())

    Differential usage test:

    >>> model = SplisosmNP()
    >>> model.setup_data(data, coordinates, design_mtx=design_mtx)
    >>> model.test_differential_usage(method='hsic-gp', residualize='cov_only')
    >>> du_results = model.get_formatted_test_results('du')
    >>> print(du_results.head())
    """

    k_neighbors: int
    """Number of nearest neighbours for the CAR spatial kernel (set in :meth:`__init__`)."""

    rho: float
    """Spatial autocorrelation strength for the CAR kernel (set in :meth:`__init__`)."""

    standardize_cov: bool
    """Whether to standardise the spatial covariance matrix (set in :meth:`__init__`)."""

    n_genes: int
    """Number of genes."""

    n_spots: int
    """Number of spots."""

    n_isos: list[int]
    """List of numbers of isoforms per gene."""

    n_factors: int
    """Number of covariates to test for differential usage."""

    gene_names: list[str]
    """List of gene names corresponding to the genes in the model."""

    covariate_names: list[str]
    """List of covariate names corresponding to columns of the design matrix."""

    adata: Optional[AnnData]
    """Source :class:`anndata.AnnData` object when using AnnData input mode;
    ``None`` in legacy (list-of-tensors) mode."""

    data: list[torch.Tensor]
    """List of isoform count tensors, one per gene, each of shape
    ``(n_spots, n_isos)``.  Set by :meth:`setup_data`."""

    coordinates: torch.Tensor
    """Spatial coordinates of shape ``(n_spots, n_dims)``.
    Set by :meth:`setup_data`."""

    corr_sp: Any
    """Spatial covariance kernel (:class:`~splisosm.kernel.SpatialCovKernel`)
    constructed from :attr:`coordinates`.  Set by :meth:`setup_data`."""

    design_mtx: Optional[torch.Tensor]
    """Design matrix of shape ``(n_spots, n_factors)``; ``None`` if no
    covariates were provided to :meth:`setup_data`."""

    sv_test_results: dict
    """Dictionary to store the spatial variability test results after running test_spatial_variability().
    It contains the following keys:

    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes,), the test statistic for each gene.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes,), the p-value for each gene.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes,), the BH adjusted p-value for each gene.
    """

    du_test_results: dict
    """Dictionary to store the differential usage test results after running test_differential_usage().
    It contains the following keys:

    - ``'method'``: str, the method used for the test.
    - ``'statistic'``: numpy.ndarray of shape (n_genes, n_factors), the test statistic for each gene and covariate.
    - ``'pvalue'``: numpy.ndarray of shape (n_genes, n_factors), the p-value for each gene and covariate.
    - ``'pvalue_adj'``: numpy.ndarray of shape (n_genes, n_factors), the BH adjusted p-value for each gene and covariate. Each column/covariate is adjusted separately.
    """

    def __init__(
        self,
        k_neighbors: int = 4,
        rho: float = 0.99,
        standardize_cov: bool = True,
    ) -> None:
        """Initialise the model.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbours used to build the spatial adjacency
            graph for the CAR kernel (default 4).
        rho : float, optional
            Spatial autocorrelation strength in the CAR model (default 0.99).
            Values close to 1 give a smoother spatial kernel.
        standardize_cov : bool, optional
            Whether to standardise the spatial covariance matrix so that its
            diagonal entries are 1 (default ``True``).
        """
        # spatial kernel hyperparameters (used in setup_data)
        self.k_neighbors = k_neighbors
        self.rho = rho
        self.standardize_cov = standardize_cov

        # to be set after running setup_data()
        self.n_genes = None  # number of genes
        self.n_spots = None  # number of spots
        self.n_isos = None  # list of number of isoforms for each gene
        self.n_factors = None  # number of covariates to test for differential usage
        self.adata = None  # optional anndata source for the new setup path
        self._setup_input_mode = None  # "legacy" or "anndata"

        # feature summary cache (populated by _compute_feature_summaries)
        self._filtered_adata = None
        self._gene_summary = None
        self._isoform_summary = None

        # to store the spatial variability test results after running test_spatial_variability()
        self.sv_test_results = {}

        # to store the differential usage test results after running test_differential_usage()
        self.du_test_results = {}

    def __str__(self) -> str:
        """Return string representation of the model."""
        _sv_status = (
            f"Completed ({self.sv_test_results['method']})"
            if len(self.sv_test_results) > 0
            else "NA"
        )
        _du_status = (
            f"Completed ({self.du_test_results['method']})"
            if len(self.du_test_results) > 0
            else "NA"
        )
        return (
            "=== Non-parametric SPLISOSM model for spatial isoform testings\n"
            + f"- Number of genes: {self.n_genes}\n"
            + f"- Number of spots: {self.n_spots}\n"
            + f"- Number of covariates: {self.n_factors}\n"
            + f"- Average number of isoforms per gene: {np.mean(self.n_isos) if self.n_isos is not None else None}\n"
            + "=== Test results\n"
            + f"- Spatial variability test: {_sv_status}\n"
            + f"- Differential usage test: {_du_status}"
        )

    def setup_data(
        self,
        adata: AnnData,
        *,
        spatial_key: str = "spatial",
        adj_key: Optional[str] = None,
        layer: str = "counts",
        group_iso_by: str = "gene_symbol",
        gene_names: Optional[str] = None,
        design_mtx: Optional[
            Union[torch.Tensor, np.ndarray, pd.DataFrame, str, list[str]]
        ] = None,
        covariate_names: Optional[list[str]] = None,
        min_counts: int = 10,
        min_bin_pct: float = 0.0,
        filter_single_iso_genes: bool = True,
        min_component_size: int = 1,
    ) -> None:
        """Setup isoform-level spatial data for hypothesis testing.

        Extracts isoform count tensors from an AnnData object, optionally
        filters disconnected graph components, builds a spatial covariance
        kernel, and resolves the design matrix.

        Parameters
        ----------
        adata : anndata.AnnData
            Annotated data matrix.  Counts are read from
            ``adata.layers[layer]`` grouped by ``group_iso_by``, and
            spatial coordinates from ``adata.obsm[spatial_key]``.
            See :func:`splisosm.utils.prepare_inputs_from_anndata` for
            full preprocessing details.
        spatial_key : str, optional
            Key in ``adata.obsm`` for spatial coordinates (default
            ``"spatial"``).
        adj_key : str or None, optional
            Key in ``adata.obsp`` for a pre-built adjacency matrix.
            When provided, it overrides the k-NN graph construction
            from coordinates and be used directly to build the spatial kernel.
            The adjacency matrix is symmetrized internally.
        layer : str, optional
            Layer in ``adata.layers`` that stores isoform counts (default
            ``"counts"``).
        group_iso_by : str, optional
            Column in ``adata.var`` used to group isoforms by gene
            (default ``"gene_symbol"``).
        gene_names : str or None, optional
            Column name in ``adata.var`` used as display names for genes.
            If ``None``, the values of ``group_iso_by`` are used.
        design_mtx : tensor, array, DataFrame, str, or list of str, optional
            Design matrix for differential-usage tests.  Accepts an
            array/tensor/DataFrame of shape ``(n_spots, n_factors)``, a
            single obs-column name (str), or a list of obs-column names.
            Categorical obs columns are one-hot encoded automatically.

            When a **scipy sparse matrix** is passed directly, it is stored as
            scipy CSR internally and all differential-usage methods handle it
            without densifying: ``"hsic"`` uses a sparse matrix-multiply path
            in :func:`linear_hsic_test`; ``"t-fisher"`` and ``"t-tippett"``
            extract group indices directly from the sparse non-zero structure.
            ``"hsic-gp"`` densifies each column via :meth:`_get_design_col`
            before GPR fitting (GPR residuals are always dense).

            All other input types (obs column names, array, tensor, DataFrame)
            are converted to a dense torch float32 tensor.
        covariate_names : list of str or None, optional
            Explicit covariate names.  When ``design_mtx`` is given as
            column name(s) and this is ``None``, the column names are used
            automatically; otherwise auto-generated as ``factor_1``, etc.
        min_counts : int, optional
            Minimum total isoform count across spots required to retain an
            isoform (default 10).
        min_bin_pct : float, optional
            Minimum fraction/percentage of spots where an isoform must be
            expressed (default 0.0).
        filter_single_iso_genes : bool, optional
            Whether to remove genes with fewer than two retained isoforms
            (default ``True``).
        min_component_size : int, optional
            Minimum number of spots a connected component must contain to
            be retained.  Spots in smaller components are removed from all
            data structures before the spatial kernel is built.  Default 1
            disables filtering.  A ``UserWarning`` is issued when spots are
            removed.

        Raises
        ------
        ValueError
            If input arguments are invalid or required fields are missing.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("`adata` must be an AnnData object.")

        (
            data,
            coordinates,
            resolved_gene_names,
            resolved_design,
            resolved_cov_names,
            adj_matrix,
            filtered_adata,
        ) = prepare_inputs_from_anndata(
            adata=adata,
            layer=layer,
            group_iso_by=group_iso_by,
            spatial_key=spatial_key,
            min_counts=min_counts,
            min_bin_pct=min_bin_pct,
            filter_single_iso_genes=filter_single_iso_genes,
            gene_names=gene_names,
            design_mtx=design_mtx,
            covariate_names=covariate_names,
            min_component_size=min_component_size,
            adj_key=adj_key,
            k_neighbors=self.k_neighbors,
            return_filtered_anndata=True,
        )

        self.adata = adata
        self._setup_input_mode = "anndata"
        self._counts_layer = layer
        self._group_iso_by = group_iso_by
        self._filtered_adata = filtered_adata
        self._gene_summary = None
        self._isoform_summary = None

        self.n_genes = len(data)
        self.n_spots = coordinates.shape[0]
        self.n_isos = [g.shape[1] for g in data]
        self.gene_names = resolved_gene_names

        # Convert to float tensors
        self.data = [g.float() for g in data]
        self.coordinates = coordinates

        # Build spatial kernel from the adjacency returned by prepare_inputs_from_anndata.
        # adj_matrix is not None when:
        # (1) min_component_size > 1, or,
        # (2) adj_key is provided
        if adj_matrix is not None:
            self.corr_sp = SpatialCovKernel(
                coords=None,
                adj_matrix=adj_matrix,
                rho=self.rho,
                standardize_cov=self.standardize_cov,
                centering=True,
            )
        else:
            self.corr_sp = SpatialCovKernel(
                coords=coordinates,
                adj_matrix=None,
                k_neighbors=self.k_neighbors,
                rho=self.rho,
                standardize_cov=self.standardize_cov,
                centering=True,
            )

        # Process design matrix from _process_design_mtx.
        # resolved_design is a numpy float32 array, a scipy sparse CSR matrix, or None.
        # Sparse design matrices are kept as scipy CSR to avoid densifying large
        # one-hot-encoded covariate tables; columns are extracted one at a time during
        # hypothesis testing.
        if resolved_design is not None:
            n_factors = resolved_design.shape[1]

            # Check for constant/zero-variance covariates without densifying
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                if scipy.sparse.issparse(resolved_design):
                    _means = np.asarray(resolved_design.mean(axis=0)).ravel()
                    _sq_means = np.asarray(
                        resolved_design.power(2).mean(axis=0)
                    ).ravel()
                    _stds = np.sqrt(np.maximum(_sq_means - _means**2, 0.0))
                    zero_var_indices = np.where(_stds < 1e-5)[0]
                else:
                    design_mtx_t = torch.from_numpy(
                        np.asarray(resolved_design, dtype=np.float32)
                    )
                    if design_mtx_t.dim() == 1:
                        design_mtx_t = design_mtx_t.unsqueeze(1)
                    _stds_t = design_mtx_t.std(dim=0)
                    zero_var_indices = torch.where(_stds_t < 1e-5)[0].numpy()
                for idx in zero_var_indices:
                    _cname = (
                        resolved_cov_names[int(idx)]
                        if resolved_cov_names is not None
                        else str(int(idx))
                    )
                    warnings.warn(
                        f"Covariate '{_cname}' has near-zero variance "
                        "(std < 1e-5). Consider removing it."
                    )

            # Store: sparse CSR when the input was sparse; dense torch tensor otherwise.
            if scipy.sparse.issparse(resolved_design):
                self.design_mtx = resolved_design.tocsr()
            else:
                self.design_mtx = design_mtx_t  # already constructed above
            self.n_factors = n_factors
            self.covariate_names = resolved_cov_names
        else:
            self.design_mtx = None
            self.n_factors = 0
            self.covariate_names = None

    def _get_design_col(self, factor_idx: int) -> torch.Tensor:
        """Extract one covariate column as a dense (n_spots, 1) float32 tensor.

        Works for both torch-tensor and scipy-sparse design matrices so that
        the bulk of the design matrix is never fully densified.
        """
        if scipy.sparse.issparse(self.design_mtx):
            col = np.asarray(self.design_mtx.getcol(factor_idx).todense()).ravel()
            return torch.from_numpy(col.astype(np.float32)).unsqueeze(1)
        return self.design_mtx[:, factor_idx].clone().float().unsqueeze(1)

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if self._gene_summary is not None and self._isoform_summary is not None:
            return

        adata = self._filtered_adata
        n_bins = int(adata.n_obs)
        iso_counts = adata.layers[self._counts_layer]
        is_sparse = scipy.sparse.issparse(iso_counts)

        if is_sparse:
            if not scipy.sparse.isspmatrix_csc(iso_counts):
                iso_counts = iso_counts.tocsc()
        else:
            iso_counts = np.asarray(iso_counts, dtype=float)

        # Derive per-gene isoform lists from the filtered adata var.
        # adata.var columns are already in the same order as counts_list
        # (i.e., all isos of gene 0 first, then gene 1, etc.)
        iso_groups = list(
            adata.var.groupby(self._group_iso_by, observed=True, sort=False)
        )

        gene_rows: list[dict] = []
        iso_rows: list[dict] = []
        all_iso_names: list[str] = []

        iterator = tqdm(
            zip(self.gene_names, iso_groups),
            desc="Genes",
            total=len(self.gene_names),
            disable=not print_progress,
        )

        for gene_name, (_, iso_group_df) in iterator:
            iso_names = iso_group_df.index.tolist()
            iso_idx = adata.var_names.get_indexer(iso_names)

            if is_sparse:
                gene_counts = iso_counts[:, iso_idx]
                iso_total = np.asarray(gene_counts.sum(axis=0), dtype=float).ravel()
                iso_sumsq = np.asarray(
                    gene_counts.power(2).sum(axis=0), dtype=float
                ).ravel()
                iso_nnz = np.diff(gene_counts.indptr).astype(float)
                row_sums = np.asarray(gene_counts.sum(axis=1), dtype=float).ravel()
            else:
                gene_counts = np.asarray(iso_counts[:, iso_idx], dtype=float)
                iso_total = gene_counts.sum(axis=0)
                iso_sumsq = np.square(gene_counts).sum(axis=0)
                iso_nnz = np.count_nonzero(gene_counts, axis=0).astype(float)
                row_sums = gene_counts.sum(axis=1)

            gene_total = float(iso_total.sum())
            valid_rows = np.flatnonzero(row_sums > 0.0)
            n_valid = int(valid_rows.size)

            iso_count_avg = iso_total / n_bins
            iso_count_var = np.maximum(
                (iso_sumsq / n_bins) - np.square(iso_count_avg), 0.0
            )
            iso_count_std = np.sqrt(iso_count_var)
            iso_pct_bin_on = iso_nnz / n_bins

            if gene_total > 0.0:
                ratio_total = iso_total / gene_total
            else:
                ratio_total = np.zeros(len(iso_names), dtype=float)

            if n_valid > 0:
                if is_sparse:
                    ratio_counts = gene_counts.tocsr()[valid_rows]
                    ratio_counts = ratio_counts.multiply(
                        (1.0 / row_sums[valid_rows])[:, None]
                    )
                    ratio_sum = np.asarray(
                        ratio_counts.sum(axis=0), dtype=float
                    ).ravel()
                    ratio_sumsq = np.asarray(
                        ratio_counts.power(2).sum(axis=0), dtype=float
                    ).ravel()
                else:
                    ratio_counts = gene_counts[valid_rows] / row_sums[valid_rows, None]
                    ratio_sum = ratio_counts.sum(axis=0)
                    ratio_sumsq = np.square(ratio_counts).sum(axis=0)

                ratio_avg = ratio_sum / n_valid
                ratio_var = np.maximum(
                    (ratio_sumsq / n_valid) - np.square(ratio_avg), 0.0
                )
                ratio_std = np.sqrt(ratio_var)
            else:
                ratio_avg = np.zeros(len(iso_names), dtype=float)
                ratio_std = np.zeros(len(iso_names), dtype=float)

            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -(np.log(ratio_total) * ratio_total)
                entropy = float(np.nan_to_num(entropy).sum())

            gene_count_avg = float(gene_total / n_bins)
            gene_count_sumsq = float(np.square(row_sums).sum())
            gene_count_var = max((gene_count_sumsq / n_bins) - (gene_count_avg**2), 0.0)

            gene_rows.append(
                {
                    "gene": gene_name,
                    "n_isos": len(iso_names),
                    "perplexity": float(np.exp(entropy)),
                    "pct_bin_on": float(n_valid / n_bins),
                    "count_avg": gene_count_avg,
                    "count_std": float(np.sqrt(gene_count_var)),
                }
            )

            all_iso_names.extend(iso_names)
            for (
                iso_name,
                pct_bin_on,
                count_total,
                count_avg,
                count_std,
                iso_ratio_total,
                iso_ratio_avg,
                iso_ratio_std,
            ) in zip(
                iso_names,
                iso_pct_bin_on,
                iso_total,
                iso_count_avg,
                iso_count_std,
                ratio_total,
                ratio_avg,
                ratio_std,
            ):
                iso_rows.append(
                    {
                        "isoform": iso_name,
                        "pct_bin_on": float(pct_bin_on),
                        "count_total": float(count_total),
                        "count_avg": float(count_avg),
                        "count_std": float(count_std),
                        "ratio_total": float(iso_ratio_total),
                        "ratio_avg": float(iso_ratio_avg),
                        "ratio_std": float(iso_ratio_std),
                    }
                )

        self._gene_summary = pd.DataFrame(gene_rows).set_index("gene")

        var_df = adata.var.loc[all_iso_names].copy()
        stats_df = pd.DataFrame(iso_rows).set_index("isoform")
        self._isoform_summary = pd.concat([var_df, stats_df], axis=1)

    def extract_feature_summary(
        self,
        level: Literal["gene", "isoform"] = "gene",
        print_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute filtered feature-level summary statistics.

        Gene-level statistics are aggregated across all isoforms that passed
        the filters applied in :meth:`setup_data`.  Isoform-level statistics
        are computed per isoform and augmented onto the corresponding rows of
        ``adata.var``.

        Results are cached: repeated calls with the same ``level`` return the
        cached :class:`pandas.DataFrame` without recomputation.

        Parameters
        ----------
        level
            Summary granularity.
            ``'gene'``: one row per gene.
            ``'isoform'``: one row per isoform that passed filtering.
        print_progress
            Whether to show a progress bar.

        Returns
        -------
        pandas.DataFrame
            For ``level='gene'``, the index is the gene display name and the
            columns are:

            - ``'n_isos'``: int. Number of isoforms retained after filtering.
            - ``'perplexity'``: float. Effective number of isoforms based on
              the marginal isoform usage entropy.
            - ``'pct_bin_on'``: float. Fraction of spots with non-zero total
              gene counts.
            - ``'count_avg'``: float. Mean per-spot total count for the gene.
            - ``'count_std'``: float. Std of per-spot total count for the gene.

            For ``level='isoform'``, the index is the isoform name (matching
            ``adata.var_names``) and the columns are the original ``adata.var``
            columns plus:

            - ``'pct_bin_on'``: float. Fraction of spots with count > 0.
            - ``'count_total'``: float. Total counts across all spots.
            - ``'count_avg'``: float. Mean count per spot.
            - ``'count_std'``: float. Std of count per spot.
            - ``'ratio_total'``: float. Fraction of total gene counts
              attributable to this isoform.
            - ``'ratio_avg'``: float. Mean per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).
            - ``'ratio_std'``: float. Std of per-spot isoform usage ratio
              (computed over spots with non-zero gene coverage).

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        ValueError
            If ``level`` is not ``'gene'`` or ``'isoform'``.
        """
        if self._filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary

    def get_formatted_test_results(
        self, test_type: Literal["sv", "du"]
    ) -> pd.DataFrame:
        """Get formatted test results as a pandas DataFrame.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Which results to retrieve: ``"sv"`` for spatial variability or
            ``"du"`` for differential usage.

        Returns
        -------
        pandas.DataFrame
            Formatted test results.
        """
        assert test_type in [
            "sv",
            "du",
        ], "Invalid test type. Must be one of 'sv' or 'du'."
        if test_type == "sv":
            # check if the spatial variability test has been run
            assert (
                len(self.sv_test_results) > 0
            ), "No spatial variability test results found. Please run test_spatial_variability() first."
            # format the results
            res = pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self.sv_test_results["statistic"],
                    "pvalue": self.sv_test_results["pvalue"],
                    "pvalue_adj": self.sv_test_results["pvalue_adj"],
                }
            )
            return res
        else:
            # check if the differential usage test has been run
            assert (
                len(self.du_test_results) > 0
            ), "No differential usage test results found. Please run test_differential_usage() first."
            # format the results
            res = pd.DataFrame(
                {
                    "gene": np.repeat(self.gene_names, self.n_factors),
                    "covariate": np.tile(self.covariate_names, self.n_genes),
                    "statistic": self.du_test_results["statistic"].reshape(-1),
                    "pvalue": self.du_test_results["pvalue"].reshape(-1),
                    "pvalue_adj": self.du_test_results["pvalue_adj"].reshape(-1),
                }
            )
            return res

    def test_spatial_variability(
        self,
        method: Literal["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"] = "hsic-ir",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        null_method: Literal["eig", "trace", "perm"] = "eig",
        null_configs: Optional[dict[str, Any]] = None,
        return_results: bool = False,
        print_progress: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial variability.

        Kernel-based multivariate hypothesis testing for spatial variability in

        - gene-level total counts (``"hsic-gc"`` or ``"spark-x"`` :cite:`zhu2021spark`)
        - isoform usage ratios (``"hsic-ir"``)
        - isoform counts (``"hsic-ic"``)

        Test statistics and p-values are computed per gene for each gene separately.

        Parameters
        ----------
        method : {"hsic-ir", "hsic-ic", "hsic-gc", "spark-x"}, optional
            Test target: ``"hsic-ir"`` (isoform usage ratios), ``"hsic-ic"``
            (isoform counts), ``"hsic-gc"`` (gene-level counts), or
            ``"spark-x"`` (SPARK-X :cite:`zhu2021spark`).
        ratio_transformation : {"none", "clr", "ilr", "alr", "radial"}, optional
            Compositional transformation applied to isoform ratios when
            ``method="hsic-ir"``.  See :func:`splisosm.utils.counts_to_ratios`
            and :cite:`park2022kernel` for details.
        nan_filling : {"mean", "none"}, optional
            Strategy for NaN values in isoform ratios.
            See :func:`splisosm.utils.counts_to_ratios` for details.
        null_method : {"eig", "trace", "perm"}, optional
            Method for computing the null distribution of the test statistic:

            * ``"eig"`` (default): asymptotic chi-square mixture using kernel
              eigenvalues; Liu's method :cite:`liu2009new`.  Supports optional
              ``null_configs["approx_rank"]`` (int) to use only the top-k
              eigenvalues. By default, approx_rank = np.ceil(np.sqrt(n_spots) * 4)
              for large datasets (n_spots > 5000). Set it to None to use
              all eigenvalues, which can be slow for large n_spots.
            * ``"trace"``: moment-matching normal approximation using
              tr(K') and tr(K'²) of the (centred) spatial kernel.
            * ``"perm"``: permutation-based null distribution.  Supports
              optional ``null_configs["n_perms_per_gene"]`` (default 1000),
              and ``null_configs["perm_batch_size"]`` (default 50, larger values
              lead to more memory usage) for batch-wise null statistic computation.
        null_configs : dict or None, optional
            Extra keyword arguments for the chosen ``null_method``.
        return_results : bool, optional
            If ``True``, return the result dict.  Otherwise store results in
            :attr:`sv_test_results` and return ``None``.
        print_progress : bool, optional
            Whether to show a progress bar.

        Returns
        -------
        dict or None
            If `return_results` is True, returns dict with test statistics and p-values.
            Otherwise, returns None and stores results in self.sv_test_results.

        Notes
        -----
        To run the SPARK-X test, the R-package `SPARK` must be installed and accessible from Python via `rpy2`.
        """

        valid_methods = ["hsic-ir", "hsic-ic", "hsic-gc", "spark-x"]
        valid_null_methods = ["eig", "trace", "perm"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["mean", "none"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            null_method in valid_null_methods
        ), f"Invalid null method. Must be one of {valid_null_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid ratio transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid NaN filling method. Must be one of {valid_nan_filling}."

        if method == "spark-x":  # run the gene-level SPARK-X test
            # prepare the data in gene-level counts
            counts_g = torch.concat(
                [_counts.sum(1, keepdim=True) for _counts in self.data], axis=1
            )  # tensor(n_spots, n_genes)
            self.sv_test_results = run_sparkx(
                counts_g.numpy(), self.coordinates.numpy()
            )
        else:
            # use a global spatial kernel unless nan_filling is 'none'
            n_spots = self.n_spots
            K_sp = self.corr_sp  # the Kernel class object was already centered

            # pre-compute null distribution inputs (once, before per-gene loop)
            configs = null_configs or {}
            if null_method == "eig":
                _rank = (
                    np.ceil(np.sqrt(self.n_spots) * 4).astype(int)
                    if self.n_spots > 5000
                    else self.n_spots
                )
                approx_rank = configs.get("approx_rank", _rank)
                if approx_rank is None and self.n_spots > 5000:
                    warnings.warn(
                        "Computing all eigenvalues for null distribution can be slow for large n_spots. "
                        "Consider setting a small value for null_configs['approx_rank'] to use low-rank approximation.",
                        UserWarning,
                        stacklevel=2,
                    )
                lambda_sp = self.corr_sp.eigenvalues(k=approx_rank)
                lambda_sp = lambda_sp[lambda_sp > 1e-5]
                k_eff = len(lambda_sp)
                # Extract low-rank factor Q_k (shape n × k_eff) so that
                # K ≈ Q_k Q_k^T. Used to compute a rank-consistent test stat
                # for p-value, preventing scale mismatch with the Liu null.
                _Q_sp = (
                    self.corr_sp.Q[:, :k_eff] if self.corr_sp.Q is not None else None
                )
            elif null_method == "trace":
                trK = self.corr_sp.trace()
                trK2 = self.corr_sp.square_trace()
            elif null_method == "perm":
                n_nulls = int(configs.get("n_perms_per_gene", 1000))
                _perm_batch_size = int(configs.get("perm_batch_size", 50))

            # iterate over genes and calculate the HSIC statistic
            hsic_list, pvals_list = [], []
            for counts in tqdm(
                self.data,
                desc=f"SV [{method}]",
                total=self.n_genes,
                disable=not print_progress,
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()

                if method == "hsic-ir" and nan_filling == "none":
                    # spetial treatment for the isoform ratio test when nan_filling is 'none'
                    # need to adjust the effective spot number (non NaN spots) and spatial kernel
                    y = counts_to_ratios(
                        counts,
                        transformation=ratio_transformation,
                        nan_filling="none",
                        fill_before_transform=False,
                    )
                    # remove spots with NaN values
                    is_nan = torch.isnan(y).any(1)  # spots with NaN values
                    y = y[~is_nan]  # (n_non_nan, n_isos)

                    # adjust the effective number of spots and update the spatial kernel
                    n_spots = y.shape[0]
                    # H = torch.eye(n_spots) - 1/n_spots
                    # K_sp = H @ self.corr_sp[~is_nan, :][:, ~is_nan] @ H # centered spatial kernel
                    K_sp = self.corr_sp.realization()[~is_nan, :][:, ~is_nan]
                    K_sp = K_sp - K_sp.mean(dim=0, keepdim=True)
                    K_sp = K_sp - K_sp.mean(dim=1, keepdim=True)

                    # calculate the eigenvalues of the new per-gene spatial kernel
                    if null_method == "eig":
                        lambda_sp = torch.linalg.eigvalsh(
                            K_sp
                        )  # eigenvalues of length n_spots
                        lambda_sp = lambda_sp[
                            lambda_sp > 1e-5
                        ]  # remove small eigenvalues

                    # compute the hsic statistic
                    hsic_scaled = torch.trace(y.T @ K_sp @ y)

                else:  # one global spatial kernel for all genes
                    if method == "hsic-ic":  # use isoform-level count data
                        y = counts - counts.mean(
                            0, keepdim=True
                        )  # centering per isoform
                    elif method == "hsic-gc":  # use gene-level count data
                        y = counts.sum(1, keepdim=True)
                        y = y - y.mean()  # centering per isoform
                    else:  # use isoform ratio
                        # calculate the isoform ratio from counts
                        y = counts_to_ratios(
                            counts,
                            transformation=ratio_transformation,
                            nan_filling="mean",
                            fill_before_transform=False,
                        )
                        y = y - y.mean(0, keepdim=True)  # centering per isoform

                    # compute the hsic statistic
                    if null_method == "eig" and _Q_sp is not None:
                        # when low-rank approximation is available,
                        # compute the statistic using the low-rank factor to align with
                        # the eigenvalues used in the Liu null distribution
                        xtQ = y.t() @ _Q_sp  # (n_isos, k_eff)
                        hsic_scaled = torch.trace(xtQ @ xtQ.t())
                    else:
                        # use the exact full kernel to compute the quadratic form
                        # even when low-rank approximation K_sp.Q is available
                        hsic_scaled = torch.trace(K_sp.xtKx_exact(y))

                hsic_list.append(hsic_scaled / (n_spots - 1) ** 2)

                if null_method == "eig":  # asymptotic chi-square mixture (Liu's method)
                    try:
                        lambda_y = torch.linalg.eigvalsh(y.T @ y)  # length of n_isos
                    except torch._C._LinAlgError:
                        # Add a small jitter to the diagonal and retry
                        lambda_y = torch.linalg.eigvalsh(
                            y.T @ y + 1e-6 * torch.eye(y.shape[1])
                        )

                    lambda_y = lambda_y[lambda_y > 1e-5]  # remove small eigenvalues
                    lambda_spy = (
                        lambda_sp.unsqueeze(0) * lambda_y.unsqueeze(1)
                    ).reshape(
                        -1
                    )  # n_spots * (n_isos or 1)
                    pval = liu_sf((hsic_scaled * n_spots).numpy(), lambda_spy.numpy())

                elif null_method == "trace":  # moment-matching normal approximation
                    S = y.T @ y  # (n_isos, n_isos)
                    trS = torch.trace(S).item()
                    trS2 = torch.trace(S @ S).item()
                    # Under the permutation null (centered K, centered y):
                    # E[tr(y.T K y)] = tr(K) * tr(y.T y) / (n-1)
                    # Var[tr(y.T K y)] ≈ 2 * tr(K^2) * tr((y.T y)^2) / (n-1)^2
                    n1 = n_spots - 1
                    mean_null = trK.item() * trS / n1
                    var_null = 2.0 * trK2.item() * trS2 / (n1**2)
                    z = (hsic_scaled.item() - mean_null) / (var_null**0.5 + 1e-12)
                    pval = float(_norm_dist.sf(z))

                elif null_method == "perm":  # permutation-based null distribution
                    p_isos = y.shape[1]
                    null_stats = []
                    for chunk_start in range(0, n_nulls, _perm_batch_size):
                        B = min(_perm_batch_size, n_nulls - chunk_start)
                        # Concatenate B permuted copies of y along the feature axis → (n, B·p)
                        y_batch = torch.cat(
                            [y[torch.randperm(n_spots)] for _ in range(B)], dim=1
                        )
                        if isinstance(K_sp, torch.Tensor):
                            # NaN path: K_sp is a dense per-gene submatrix tensor
                            R = y_batch.T @ K_sp @ y_batch  # (B·p, B·p)
                        else:
                            # Global path: one LU solve for B·p right-hand sides
                            R = K_sp.xtKx(y_batch)  # (B·p, B·p)
                        # Recover per-permutation traces from diagonal blocks:
                        # tr(y_i.T K y_i) = diagonal(R)[i·p:(i+1)·p].sum()
                        null_stats.append(
                            torch.diagonal(R).reshape(B, p_isos).sum(dim=1)
                        )
                    null_m = torch.cat(null_stats)  # (n_nulls,)
                    pval = float((null_m > hsic_scaled).sum() / n_nulls)

                pvals_list.append(pval)

                # store the results
                self.sv_test_results = {
                    "statistic": torch.tensor(hsic_list).numpy(),
                    "pvalue": torch.tensor(pvals_list).numpy(),
                    "method": method,
                    "null_method": null_method,
                }

            # calculate adjusted p-values
            self.sv_test_results["pvalue_adj"] = false_discovery_control(
                self.sv_test_results["pvalue"]
            )

        # return results
        if return_results:
            return self.sv_test_results

    def test_differential_usage(
        self,
        method: Literal["hsic", "hsic-gp", "t-fisher", "t-tippett"] = "hsic-gp",
        ratio_transformation: Literal["none", "clr", "ilr", "alr", "radial"] = "none",
        nan_filling: Literal["mean", "none"] = "mean",
        gpr_backend: Literal["sklearn", "gpytorch"] = "sklearn",
        gpr_configs: Optional[dict[str, Any]] = None,
        residualize: Literal["cov_only", "both"] = "cov_only",
        print_progress: bool = True,
        return_results: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Test for spatial isoform differential usage.

        Before running this function, the design matrix must be set up using :func:`setup_data`.
        Each column of the design matrix corresponds to a covariate to test for differential
        association with the isoform usage ratios of each gene.
        Test statistics and p-values are computed per (gene, covariate) pair separately.

        Two types of association tests are supported:

        - Unconditional (``"hsic"``, ``"t-fisher"``, ``"t-tippett"``): test the
          unconditional association between isoform usage ratios and the covariate.
        - Conditional (``"hsic-gp"``): test the association conditioned on spatial
          coordinates via Gaussian process regression.  See :cite:`zhang2012kernel`
          for more details.

        Parameters
        ----------
        method : str, optional
            Method for association testing:

            * ``"hsic"``: Unconditional HSIC test (multivariate RV coefficient).
              For continuous factors, equivalent to the multivariate Pearson correlation
              test.  For binary factors, equivalent to the two-sample Hotelling T**2 test.
            * ``"hsic-gp"``: Conditional HSIC test.  Spatial effects are removed via
              Gaussian process regression before computing the HSIC statistic.

            Or one of the T-tests (binary factors only):

            * ``"t-fisher"``, ``"t-tippett"``: each isoform is tested independently
              and p-values are combined gene-wise via Fisher's or Tippett's method.
        ratio_transformation : str, optional
            Compositional transformation for isoform ratios.
            One of ``'none'``, ``'clr'``, ``'ilr'``, ``'alr'``, ``'radial'``
            :cite:`park2022kernel`.  See :func:`splisosm.utils.counts_to_ratios`.
        nan_filling : str, optional
            How to fill NaN values in isoform ratios.  One of ``'mean'`` or ``'none'``.
            See :func:`splisosm.utils.counts_to_ratios`.
        gpr_backend : str, optional
            GPR backend to use for ``method='hsic-gp'``.
            One of ``'sklearn'`` (default) or ``'gpytorch'``.
            For FFT-accelerated spatial GP on regular grids use
            :class:`~splisosm.hyptest_fft.SplisosmFFT` instead.
        gpr_configs : dict, optional
            Nested configuration dict for the GPR objects, with optional keys
            ``'covariate'`` and/or ``'isoform'``.  Each sub-dict is forwarded to
            :func:`splisosm.kernel_gpr.make_kernel_gpr`.  Unspecified keys use the
            defaults from :data:`splisosm.kernel_gpr._DEFAULT_GPR_CONFIGS`::

                {
                    "covariate": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                        "n_inducing": 5000,
                    },
                    "isoform": {
                        "constant_value": 1.0,
                        "constant_value_bounds": (1e-3, 1e3),
                        "length_scale": 1.0,
                        "length_scale_bounds": "fixed",
                        "n_inducing": 5000,
                    },
                }

            ``"n_inducing"`` *(int or None)* controls the scale of spatial GP
            fitting for each backend:

            * **sklearn** — maximum number of observations used for
              hyperparameter fitting.  Full exact GP when ``n_obs ≤ n_inducing``
              (or ``None``); a randomly sub-sampled **subset-of-data** of
              ``n_inducing`` points otherwise (**not** the same inducing-point
              approximation as gpytorch).  Default: ``5000``.  Set to ``None``
              to use all observations (warns when ``n_obs > 10_000``).
            * **gpytorch** — FITC sparse-GP inducing-point approximation with
              ``n_inducing`` points; set to ``None`` for exact GP.
              Default: ``5000``.

        residualize : {"cov_only", "both"}, optional
            Controls which signals are spatially residualized when
            ``method="hsic-gp"``:

            * ``"cov_only"`` (default): residualize covariates only; test
              HSIC(Z_res, Y_raw).  Fastest; calibration matches ``"both"``
              when covariate GPR captures most spatial confounding.
            * ``"both"``: residualize both covariates and isoform ratios.
        print_progress : bool, optional
            Whether to show the progress bar. Default to True.
        return_results : bool, optional
            Whether to return the test statistics and p-values.
            If False, the results are stored in ``self.du_test_results``.

        Returns
        -------
        results : dict or None
            If ``return_results`` is True, returns dict with test statistics and
            p-values. Otherwise, returns None and stores results in
            ``self.du_test_results``.
        """
        if self.design_mtx is None:
            raise ValueError(
                "Cannot find the design matrix. Perhaps you forgot to set it up using setup_data()."
            )

        n_spots, n_factors = self.design_mtx.shape

        # check the validity of the specified method and transformation
        valid_methods = ["hsic", "hsic-gp", "t-fisher", "t-tippett"]
        valid_transformations = ["none", "clr", "ilr", "alr", "radial"]
        valid_nan_filling = ["none", "mean"]
        valid_residualize = ["cov_only", "both"]
        assert (
            method in valid_methods
        ), f"Invalid method. Must be one of {valid_methods}."
        assert (
            ratio_transformation in valid_transformations
        ), f"Invalid transformation. Must be one of {valid_transformations}."
        assert (
            nan_filling in valid_nan_filling
        ), f"Invalid nan_filling. Must be one of {valid_nan_filling}."
        assert (
            residualize in valid_residualize
        ), f"Invalid residualize. Must be one of {valid_residualize}."

        n_genes = self.n_genes

        if method == "hsic":  # unconditional HSIC test (multivariate RV coefficient)
            # Pre-compute covariates: keep sparse when design_mtx is scipy sparse
            # so that linear_hsic_test can use the memory-efficient sparse-X path.
            z_list = []
            for _ind in range(n_factors):
                if scipy.sparse.issparse(self.design_mtx):
                    z = self.design_mtx.getcol(_ind)  # scipy sparse (n_spots, 1)
                    _mean = float(z.mean())
                    _sq_mean = float(z.multiply(z).mean())
                    _std = float(np.sqrt(max(_sq_mean - _mean**2, 0.0)))
                else:
                    z = self._get_design_col(_ind)  # dense (n_spots, 1) tensor
                    _std = float(z.std())
                assert (
                    _std > 1e-5
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z_list.append(z)

            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            # Outer loop: one gene at a time to avoid materialising all dense counts
            for _g, counts in enumerate(
                tqdm(
                    self.data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                y = counts_to_ratios(
                    counts,
                    transformation=ratio_transformation,
                    nan_filling=nan_filling,
                    fill_before_transform=False,
                )
                for _f, z in enumerate(z_list):
                    hsic_all[_g, _f], pvals_all[_g, _f] = linear_hsic_test(
                        z, y, centering=True
                    )

            self.du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        elif method == "hsic-gp":  # conditional HSIC via GP regression residuals
            # Build GPR configs (merge user overrides over defaults)
            cov_config = {**_DEFAULT_GPR_CONFIGS["covariate"]}
            iso_config = {**_DEFAULT_GPR_CONFIGS["isoform"]}
            if gpr_configs is not None:
                if "covariate" in gpr_configs:
                    cov_config.update(gpr_configs["covariate"])
                if "isoform" in gpr_configs:
                    iso_config.update(gpr_configs["isoform"])

            # Normalize spatial coordinates once
            x = self.coordinates.clone()  # (n_spots, n_dims)
            x = (x - x.mean(0)) / x.std(0)
            x[torch.isinf(x)] = 0  # guard against constant coordinate axes

            # Fit GPR for covariates and get residuals (n_factors small tensors, never sparse)
            gpr_cov = make_kernel_gpr(gpr_backend, **cov_config)
            z_res_list = []
            for _ind in tqdm(
                range(n_factors),
                desc="Covariates",
                total=n_factors,
                disable=not print_progress,
            ):
                z = self._get_design_col(_ind).squeeze(1)
                assert (
                    z.std() > 0
                ), f"The factor of interest {self.covariate_names[_ind]} have zero variance."
                z = (z - z.mean()) / z.std()
                z_res_list.append(gpr_cov.fit_residuals(x, z.unsqueeze(1)))

            # Optionally fit GPR for isoform ratios to residualize spatial effects in the response as well.
            if residualize == "both":
                gpr_iso = make_kernel_gpr(gpr_backend, **iso_config)
                # Warm up the shared eigendecomposition for backends that support it
                # (e.g. sklearn with fixed signal bounds) so the first gene does not
                # pay the cost of a redundant full GP fit.
                if (
                    hasattr(gpr_iso, "precompute_shared_kernel")
                    and gpr_iso.signal_bounds_fixed
                ):
                    gpr_iso.precompute_shared_kernel(x)

            # --- Main loop: densify and process one gene at a time ---
            hsic_all = torch.empty(n_genes, n_factors)
            pvals_all = torch.empty(n_genes, n_factors)

            for _g, counts in enumerate(
                tqdm(
                    self.data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                y = counts_to_ratios(
                    counts,
                    transformation=ratio_transformation,
                    nan_filling=nan_filling,
                    fill_before_transform=False,
                )

                if residualize == "both":
                    y = gpr_iso.fit_residuals(x, y)

                for _f, z_res in enumerate(z_res_list):
                    hsic_all[_g, _f], pvals_all[_g, _f] = linear_hsic_test(
                        z_res, y, centering=True
                    )

            self.du_test_results = {
                "statistic": hsic_all.numpy(),
                "pvalue": pvals_all.numpy(),
                "method": method,
            }

        else:  # two-sample t-test
            # method to combine p-values across isoforms, either 'fisher' or 'tippett'
            combine_method = re.findall(r"^t-(.+)", method)[0]

            stats_all = np.empty((n_genes, n_factors))
            pvals_all = np.empty((n_genes, n_factors))

            # Pre-extract group columns once (sparse or dense) to avoid repeated
            # column lookups inside the gene loop.
            _design_is_sparse = scipy.sparse.issparse(self.design_mtx)
            groups_list = [
                (
                    self.design_mtx.getcol(_ind)  # scipy sparse (n, 1)
                    if _design_is_sparse
                    else self.design_mtx[:, _ind]
                )  # dense 1-D tensor
                for _ind in range(n_factors)
            ]

            # Outer loop: one gene at a time so each sparse tensor is densified once
            for _g, counts in enumerate(
                tqdm(
                    self.data,
                    desc=f"DU [{method}]",
                    total=self.n_genes,
                    disable=not print_progress,
                )
            ):
                if counts.is_sparse:
                    counts = counts.to_dense()
                ratios = counts_to_ratios(
                    counts,
                    transformation=ratio_transformation,
                    nan_filling=nan_filling,
                    fill_before_transform=False,
                )
                for _ind, groups in enumerate(groups_list):
                    _stats, _pvals = _calc_ttest_differential_usage(
                        ratios,
                        groups,
                        combine_pval=True,
                        combine_method=combine_method,
                    )
                    stats_all[_g, _ind] = _stats
                    pvals_all[_g, _ind] = _pvals

            self.du_test_results = {
                "statistic": stats_all,  # (n_genes, n_factors)
                "pvalue": pvals_all,  # (n_genes, n_factors)
                "method": method,
            }

        # calculate adjusted p-values (independently for each factor)
        self.du_test_results["pvalue_adj"] = false_discovery_control(
            self.du_test_results["pvalue"], axis=0
        )

        # return the results if needed
        if return_results:
            return self.du_test_results
