"""Paired-cell, coverage, confound and complete saved-output regression checks."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.atac_programs import (read_gene_activity, balanced_standardize, relative_state,
                               residual_correlations, analyze_programs, run_atac)


class AtacTests(unittest.TestCase):
    def test_reference_weights_types_equally(self):
        x = np.array([[0, 1], [2, 1], [10, 1], [10, 1], [10, 1]])
        z, mean, sd, valid = balanced_standardize(x, ['a', 'a', 'b', 'b', 'b'])
        self.assertAlmostEqual(mean[0], 5.5)
        self.assertAlmostEqual(sd[0] ** 2, 20.75)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertTrue(np.isnan(z[:, 1]).all())

    def test_states_keep_intermediate_and_missing(self):
        self.assertIn('intermediate', relative_state(.1, .1))
        self.assertIn('high RNA and ATAC', relative_state(.6, .6))
        self.assertIn('potentially permissive', relative_state(-.6, .6))
        self.assertIn('candidate decoupling', relative_state(.6, -.6))
        self.assertEqual(relative_state(np.nan, 1), 'insufficient evidence')

    def test_pairing_for_dense_and_sparse_and_reordered_cells(self):
        import anndata as ad
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'activity.h5ad'
            obs = pd.DataFrame({'DonorID': ['d'] * 3, 'Site': ['s'] * 3, 'cell_type': ['NK'] * 3}, index=['a', 'b', 'c'])
            obj = ad.AnnData(sparse.csr_matrix(np.ones((3, 1))), obs=obs)
            obj.uns['ATAC_gene_activity_var_names'] = np.array(['G1', 'G2', 'OTHER'])
            activity = np.array([[1., 2., 10.], [3., 4., 20.], [5., 6., 30.]])
            for representation in [activity, sparse.csr_matrix(activity), sparse.csc_matrix(activity)]:
                obj.obsm['ATAC_gene_activity'] = representation
                obj.write_h5ad(path)
                result, genes, qc = read_gene_activity(path, obs.loc[['c', 'a']], {'G2'}, chunk_size=1)
                np.testing.assert_array_equal(result.toarray(), [[6], [2]])
                np.testing.assert_array_equal(qc.atac_activity_total, [41, 13])
                self.assertEqual(genes.tolist(), ['G2'])
            bad_obs = obs.copy()
            bad_obs.loc['a', 'DonorID'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                read_gene_activity(path, bad_obs, {'G2'})
            bad_obs.index = ['missing', 'b', 'c']
            with self.assertRaisesRegex(ValueError, 'missing from source'):
                read_gene_activity(path, bad_obs, {'G2'})
            obj.obsm['ATAC_gene_activity'] = -activity
            obj.write_h5ad(path)
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                read_gene_activity(path, obs, {'G2'})

    def test_depth_adjustment_removes_simulated_common_depth(self):
        rng = np.random.default_rng(123)
        depth = rng.uniform(0, 4, 2000)
        obs = pd.DataFrame({'rna_library_counts': np.expm1(depth), 'atac_depth_covariate': np.expm1(depth), 'Site': 's'})
        r, a = depth + rng.normal(0, .1, len(depth)), depth + rng.normal(0, .1, len(depth))
        self.assertGreater(np.corrcoef(r, a)[0, 1], .98)
        self.assertLess(abs(residual_correlations(r, a, obs)[0]), .1)

    def test_insufficient_coverage_never_becomes_negative_support(self):
        x = np.arange(24).reshape(12, 2)
        obs = pd.DataFrame({'cell_type_harmonized': ['a'] * 6 + ['b'] * 6})
        chosen = pd.DataFrame({'cell_type': ['a'], 'pathway': ['P'], 'shared_leading_edge': ['A;B']})
        tables = analyze_programs(x, x, ['A', 'B'], obs, chosen, {'P': ['A', 'B', 'C']}, {'A', 'B'}, min_genes=3)
        self.assertFalse(tables['coverage'].analyzed.iloc[0])
        self.assertTrue(tables['population_summary'].empty)
        self.assertTrue(tables['cell_scores'].empty)

    def test_complete_pipeline(self):
        import anndata as ad
        from src.single_donor_workflow import prepare_donor, characterize_donor, run_concordance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, rng = {}, np.random.default_rng(11)
            genes = [f'G{i:03}' for i in range(80)]
            for dataset, n in [('cite', 24), ('multiome', 20)]:
                labels = np.repeat(['CD4+ T naive', 'CD14+ Mono', 'NK'], n)
                counts = rng.poisson(1, (len(labels), 80))
                for k in range(3):
                    counts[k*n:(k+1)*n, k*20:(k+1)*20] += rng.poisson(8, (n, 20))
                obs = pd.DataFrame({'DonorID': 'd1', 'Site': ['s1', 's2'] * (len(labels)//2), 'cell_type': labels,
                                    'ATAC_nCount_peaks': rng.integers(100, 1000, len(labels))},
                                   index=[f'{dataset}_{i}' for i in range(len(labels))])
                obj = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame({'feature_types': 'GEX'}, index=genes))
                obj.layers['counts'] = obj.X.copy()
                if dataset == 'multiome':
                    activity = np.log1p(counts + rng.poisson(1, counts.shape))
                    activity[0] = 0  # explicit zero-ATAC-cell exclusion from BOTH score matrices
                    obj.obsm['ATAC_gene_activity'] = sparse.csr_matrix(activity)
                    obj.uns['ATAC_gene_activity_var_names'] = np.array(genes)
                paths[dataset] = root / f'{dataset}.h5ad'
                obj.write_h5ad(paths[dataset])
            gmt = root / 'test.gmt'
            gmt.write_text('\n'.join(f'P{k}\ttest\t' + '\t'.join(genes[k*20:(k+1)*20]) for k in range(3)))
            output = root / 'results'
            _, _, _, data = prepare_donor(paths, output, min_cells=10)
            characterize_donor(data, output, gmt, permutations=100)
            _, _, shortlist, _ = run_concordance(output, repeats=1, cap=20)
            chosen = shortlist[shortlist.selected_for_followup]
            self.assertFalse(chosen.empty)
            requested = list(zip(chosen.cell_type, chosen.pathway))[:2]
            before = (output / 'manifest.json').read_bytes()
            tables, target, provenance = run_atac(output, paths['multiome'], requested, min_genes=5, min_cells=10)
            self.assertEqual(provenance['n_cells'], 59)
            self.assertEqual(provenance['n_programs_analyzed'], len(requested))
            self.assertEqual(provenance['depth_covariate'], 'ATAC_nCount_peaks')
            self.assertEqual(len(target), len(requested))
            self.assertEqual(before, (output / 'manifest.json').read_bytes())
            self.assertTrue((output / 'atac/figures/rna_atac_module_scatter.png').is_file())
            self.assertEqual(json.loads((output / 'atac/status.json').read_text())['state'], 'complete')

    def test_notebook(self):
        import nbformat
        path = Path(__file__).resolve().parents[1] / 'notebooks/nb06_pathway_atac_support.ipynb'
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertFalse(any(o.output_type == 'error' for o in cell.outputs))
        text = '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown')
        for header in ['Scientific question', 'Analysis strategy', 'Main findings', 'Limitations']:
            self.assertIn('## ' + header, text)


if __name__ == '__main__':
    unittest.main()
