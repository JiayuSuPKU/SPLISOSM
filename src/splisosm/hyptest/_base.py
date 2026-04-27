"""Shared implementation helpers for SPLISOSM hypothesis-test classes."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from splisosm.utils.preprocessing import compute_feature_summaries


class _FeatureSummaryMixin:
    """Shared feature-summary API for model classes with filtered AnnData."""

    def _feature_summary_adata(self) -> AnnData:
        """Return the filtered AnnData used for feature summaries."""
        filtered_adata = getattr(self, "_filtered_adata", None)
        if filtered_adata is None:
            raise RuntimeError("Data is not initialized. Call setup_data() first.")
        return filtered_adata

    def _compute_feature_summaries(self, print_progress: bool = True) -> None:
        """Compute and cache both gene-level and isoform-level summaries."""
        if self._gene_summary is not None and self._isoform_summary is not None:
            return
        self._gene_summary, self._isoform_summary = compute_feature_summaries(
            self._feature_summary_adata(),
            self.gene_names,
            layer=self._counts_layer,
            group_iso_by=self._group_iso_by,
            print_progress=print_progress,
        )

    def extract_feature_summary(
        self,
        level: Literal["gene", "isoform"] = "gene",
        print_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute filtered feature-level summary statistics.

        Gene-level statistics are aggregated across all isoforms retained by
        :meth:`setup_data`. Isoform-level statistics are computed per isoform
        and joined to the corresponding rows of ``adata.var``.

        Results are cached: repeated calls with the same ``level`` return the
        cached :class:`pandas.DataFrame` without recomputation.

        Parameters
        ----------
        level : {"gene", "isoform"}, optional
            Summary granularity. ``"gene"`` returns one row per gene;
            ``"isoform"`` returns one row per retained isoform.
        print_progress : bool, optional
            Whether to show a progress bar while computing summaries.

        Returns
        -------
        pandas.DataFrame
            Feature summary table. Gene-level summaries include ``n_isos``,
            ``perplexity``, ``pct_bin_on``, ``count_avg``, and
            ``count_std``. Isoform-level summaries include the retained
            ``adata.var`` columns plus count and usage-ratio summaries.

        Raises
        ------
        RuntimeError
            If :meth:`setup_data` has not been called.
        ValueError
            If ``level`` is not ``"gene"`` or ``"isoform"``.
        """
        if level not in {"gene", "isoform"}:
            raise ValueError("`level` must be one of 'gene' or 'isoform'.")

        self._compute_feature_summaries(print_progress=print_progress)

        if level == "gene":
            return self._gene_summary
        return self._isoform_summary


class _ResultsMixin:
    """Shared formatting API for SV and DU test results."""

    def get_formatted_test_results(
        self,
        test_type: Literal["sv", "du"],
        with_gene_summary: bool = False,
    ) -> pd.DataFrame:
        """Get formatted test results as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        test_type : {"sv", "du"}
            Which results to retrieve: ``"sv"`` for spatial variability or
            ``"du"`` for differential usage.
        with_gene_summary : bool, optional
            If ``True``, append gene-level summary statistics from
            :meth:`extract_feature_summary`.

        Returns
        -------
        pandas.DataFrame
            Result table. SV results contain ``gene``, ``statistic``,
            ``pvalue``, and ``pvalue_adj``. DU results additionally contain
            ``covariate`` and include one row per gene-covariate pair.

        Raises
        ------
        ValueError
            If ``test_type`` is invalid or the requested test has not been run.
        """
        if test_type not in {"sv", "du"}:
            raise ValueError("test_type must be 'sv' or 'du'.")

        if test_type == "sv":
            if len(self._sv_test_results) == 0:
                raise ValueError(
                    "No spatial variability results. Run test_spatial_variability() first."
                )
            df = pd.DataFrame(
                {
                    "gene": self.gene_names,
                    "statistic": self._sv_test_results["statistic"],
                    "pvalue": self._sv_test_results["pvalue"],
                    "pvalue_adj": self._sv_test_results["pvalue_adj"],
                }
            )
        else:
            if len(self._du_test_results) == 0:
                raise ValueError(
                    "No differential usage results. Run test_differential_usage() first."
                )
            covariate_names = (
                self.covariate_names
                if self.covariate_names is not None and len(self.covariate_names) > 0
                else [f"factor_{i}" for i in range(self.n_factors)]
            )
            df = pd.DataFrame(
                {
                    "gene": np.repeat(self.gene_names, self.n_factors),
                    "covariate": np.tile(covariate_names, self.n_genes),
                    "statistic": self._du_test_results["statistic"].reshape(-1),
                    "pvalue": self._du_test_results["pvalue"].reshape(-1),
                    "pvalue_adj": self._du_test_results["pvalue_adj"].reshape(-1),
                }
            )

        if with_gene_summary:
            gene_df = self.extract_feature_summary(level="gene", print_progress=False)
            df = df.merge(gene_df, left_on="gene", right_index=True, how="left")

        return df
