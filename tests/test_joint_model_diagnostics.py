"""Numerical guardrails for diagnostic perturbations and missing modalities."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.joint_model_diagnostics import (
    residualize_latent_dimension, compute_latent_technical_associations,
    compute_neighbor_depth_dependence, compute_stratified_correlations,
    drop_latent_dimensions, eta_squared, evaluate_assay_predictability,
    compare_rna_feature_panels, decision_tables)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.obs = pd.DataFrame({'assay': ['CITE']*80+['Multiome']*80,
                                 'DonorID': ['d1']*40+['d2']*40+['d1']*40+['d2']*40,
                                 'Site': ['s1']*160, 'cell_type_harmonized': ['T']*160,
                                 'protein_counts': np.r_[np.arange(1, 81), np.full(80, np.nan)],
                                 'shared_rna_counts': rng.integers(20, 100, 160),
                                 'atac_counts': np.r_[np.full(80, np.nan), np.arange(1, 81)]})
        self.z = rng.normal(size=(160, 4))
        self.z[:80, 0] = 4*np.log1p(self.obs.protein_counts[:80])+2

    def test_residualization_preserves_means_and_missing_capture(self):
        out, _ = residualize_latent_dimension(self.z, self.obs, 0, min_cells=20)
        np.testing.assert_array_equal(out[80:], self.z[80:])
        np.testing.assert_array_equal(out[:, 1:], self.z[:, 1:])
        for ids in [np.arange(40), np.arange(40, 80)]:
            self.assertAlmostEqual(out[ids, 0].mean(), self.z[ids, 0].mean())
            self.assertLess(out[ids, 0].std(), 1e-10)

    def test_absent_modalities_never_create_correlations(self):
        a = compute_latent_technical_associations(self.z, self.obs)
        self.assertEqual(set(a[a.variable.eq('protein_counts')].assay), {'CITE'})
        self.assertEqual(set(a[a.variable.eq('atac_counts')].assay), {'Multiome'})
        s = compute_stratified_correlations(self.z, self.obs, [0], min_cells=20)
        self.assertTrue(s[s.variable.eq('protein_counts')].spearman.gt(.99).all())
        self.assertEqual(len(compute_stratified_correlations(self.z, self.obs, [0], min_cells=100)), 0)

    def test_neighbor_missingness_and_self_exclusion_reference(self):
        nn = np.tile(np.arange(80, 110), (160, 1))
        out = compute_neighbor_depth_dependence(self.obs, nn)
        self.assertTrue(out.absolute_log_depth_gap.isna().all())
        self.assertTrue(out.observed_protein_neighbors.eq(0).all())

    def test_removal_and_effect_size(self):
        np.testing.assert_array_equal(drop_latent_dimensions(self.z, [0]), self.z[:, 1:])
        self.assertAlmostEqual(eta_squared([0, 0, 1, 1], ['a', 'a', 'b', 'b']), 1)
        with self.assertRaises(ValueError):
            drop_latent_dimensions(self.z, [0, 1, 2, 3])

    def test_predictability_holds_out_donors(self):
        rows, coefficients = evaluate_assay_predictability({'full': self.z}, self.obs)
        self.assertEqual(set(rows.held_out_donor), {'d1', 'd2'})
        self.assertTrue(rows.n_test.eq(80).all())
        self.assertEqual(len(coefficients), 8)

    def test_broad_panel_counts_and_cell_identity(self):
        import anndata as ad
        from scipy import sparse
        rng = np.random.default_rng(4)
        counts = sparse.csr_matrix(rng.poisson(4, (40, 12)).astype(float))
        obs = pd.DataFrame(index=[f'CITE::{i}' for i in range(40)])
        data = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f'g{i}' for i in range(12)]))
        data.layers['counts'] = counts.copy()
        bundle = dict(rna_all=counts[:, :5], ids=np.arange(20), obs_all=obs,
                      genes=data.var_names[:5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shared_rna.h5ad'
            # Reversed storage order must be recovered by exact cell identity.
            data[::-1].copy().write_h5ad(path)
            spaces, audit, _ = compare_rna_feature_panels(bundle, folder)
            self.assertEqual(spaces['rna_broad'].shape[0], 20)
            self.assertEqual(audit.iloc[1].genes, 12)
            data.layers['counts'][0, 0] += 1
            data.write_h5ad(path)
            with self.assertRaisesRegex(ValueError, 'counts disagree'):
                compare_rna_feature_panels(bundle, folder)

    def test_completed_broad_panel_report_never_requests_missing_file(self):
        rows = []
        for name, mixing in [('joint_full', .12), ('minus_top1', .13),
                              ('depth_residualized', .12), ('rna_994', .02), ('rna_broad', .025)]:
            rows.append(dict(representation=name, cross_capture_mixing=mixing,
                             cell_type_purity=.9, RNA_genes=12059 if name == 'rna_broad' else 994,
                             problematic_cell_types=11, evaluable_cell_types=13,
                             problematic_donor_celltypes=39, evaluable_donor_celltypes=42))
        associations = pd.DataFrame([dict(variable='protein_counts', dimension=0, spearman=.9)])
        strata = pd.DataFrame([dict(variable='protein_counts', dimension=0,
                                    median_absolute=.9, n_evaluable=48)])
        predictions = pd.DataFrame([dict(representation=name, auroc=.99, balanced_accuracy=.99)
                                    for name in ['joint_full', 'minus_top1']])
        failure = pd.DataFrame({'main_failure_mode': ['B: capture-specific neighborhoods']})
        decisions, report = decision_tables(pd.DataFrame(rows), associations, strata, predictions,
                                           failure, 'full available RNA module scores')
        self.assertNotIn('restore nb10', report)
        self.assertNotIn('unavailable', report)
        self.assertIn('12,059 genes', report)
        self.assertIn('Option E', report)


if __name__ == '__main__':
    unittest.main()
