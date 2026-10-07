"""Synthetic software checks, not biological validation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.celltype_harmonization import mapping_table
from src.donor_selection import select_donor
from src.expression_programs import normalize_counts, characterize, balanced_indices
from src.pathway_enrichment import enrichment_score, preranked, bh_adjust
from src.rna_concordance import compare_pathways, select_programs


class NumericalTests(unittest.TestCase):
    def test_hallmark_download_and_cache(self):
        import hashlib
        import io
        from unittest.mock import patch
        from src.pathway_enrichment import cache_hallmark, read_gmt
        data = ('\ufeff' + '\n'.join(
            f'{name}\t\tIL6\tSTAT3\t' for name in
            ['TNF-alpha Signaling via NF-kB', 'Hypoxia'] + [f'Program {i}' for i in range(48)]
        ) + '\n\n').encode('utf-8')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hallmark.gmt'
            with patch('src.pathway_enrichment.urlopen', return_value=io.BytesIO(data)):
                digest = cache_hallmark(path)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(digest, hashlib.sha256(data).hexdigest())
            self.assertEqual(len(read_gmt(path)), 50)
            with patch('src.pathway_enrichment.urlopen', side_effect=AssertionError('Should reuse cache')):
                self.assertEqual(cache_hallmark(path), digest)
            path.write_text('Hypoxia\t\tIL6\n')
            with self.assertRaisesRegex(ValueError, 'Expected 50'):
                cache_hallmark(path)
            for invalid in [b'<html>error</html>', b'Empty\t\t\n', b'Duplicate\t\tA\nDuplicate\t\tB']:
                target = Path(directory) / 'invalid.gmt'
                with patch('src.pathway_enrichment.urlopen', return_value=io.BytesIO(invalid)):
                    with self.assertRaises(ValueError):
                        cache_hallmark(target)
                self.assertFalse(target.exists())

    def test_normalization(self):
        x = sparse.csr_matrix([[1, 3], [2, 6]])
        actual = normalize_counts(x).toarray()
        np.testing.assert_allclose(actual[0], np.log1p([2500, 7500]))
        np.testing.assert_allclose(actual[0], actual[1])
        np.testing.assert_array_equal(x.toarray(), [[1, 3], [2, 6]])
        for bad in ([[0, 0]], [[1, -1]], [[1.5, 2]], [[np.nan, 2]]):
            with self.assertRaises(ValueError):
                normalize_counts(bad)

    def test_specificity(self):
        x = sparse.csr_matrix([[5, 0], [1, 2], [3, 4], [3, 4], [3, 4]])
        out = characterize(x, ['A', 'B'], ['x', 'y', 'z', 'z', 'z'], min_cells=1, median=True)
        a = out[(out.cell_type == 'x') & (out.gene == 'A')].iloc[0]
        self.assertEqual(a.specificity, 3)
        self.assertEqual(a.median_expression, 5)

    def test_mapping(self):
        m = mapping_table(['CD8+ T CD57+ CD45RO+', 'MAIT', 'new label'], ['CD8+ T', 'HSC'])
        self.assertEqual(m.loc[m.original_label == 'CD8+ T CD57+ CD45RO+', 'harmonized_label'].iloc[0], 'CD8 T cells')
        self.assertTrue(m.loc[m.original_label.isin(['MAIT', 'new label']), 'harmonized_label'].isna().all())

    def test_balanced(self):
        a, b = np.array(['x'] * 20 + ['y'] * 10), np.array(['x'] * 8 + ['y'] * 25)
        ia, ib = balanced_indices(a, b, cap=12)
        self.assertEqual(len(ia), len(set(ia)))
        for ct in ['x', 'y']:
            self.assertEqual((a[ia] == ct).sum(), (b[ib] == ct).sum())

    def test_score_reference(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            scores = np.sort(rng.normal(size=40))[::-1]
            hits = np.sort(rng.choice(40, 8, replace=False))
            inc = np.full(40, -1 / 32)
            inc[hits] = abs(scores[hits]) / abs(scores[hits]).sum()
            walk = np.cumsum(inc)
            self.assertAlmostEqual(enrichment_score(scores, hits)[0], walk[np.argmax(abs(walk))])
        self.assertAlmostEqual(enrichment_score(np.arange(20, 0, -1), np.arange(15, 20))[0], -1)

    def test_permutation(self):
        rank = pd.Series(np.linspace(3, -3, 100), index=[f'G{i}' for i in range(100)])
        sets = {'top': list(rank.index[:15]), 'bottom': list(rank.index[-15:])}
        a = preranked(rank, sets, permutations=100)
        pd.testing.assert_frame_equal(a, preranked(rank, sets, permutations=100))
        self.assertGreater(a.set_index('pathway').loc['top', 'NES'], 0)
        self.assertLess(a.set_index('pathway').loc['bottom', 'NES'], 0)
        self.assertTrue((a.p_value > 0).all())
        np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])

    def test_states(self):
        a = pd.DataFrame({'cell_type': ['x'] * 4, 'pathway': ['pos', 'neg', 'null', 'one'],
                          'NES': [2, -2, 1, 2], 'q_value': [.01, .01, .9, .01], 'leading_edge': ['A;B'] * 4})
        out = compare_pathways(a, a.iloc[:3].copy()).set_index('pathway')
        self.assertEqual(out.loc['neg', 'concordance'], 'concordant negative')
        self.assertEqual(out.loc['null', 'concordance'], 'unsupported')
        self.assertEqual(out.loc['one', 'concordance'], 'not tested in both')
        self.assertEqual(select_programs(out.reset_index()).selected_for_followup.sum(), 1)

    def test_empty(self):
        result = preranked(pd.Series([1., 0., -1.], index=['a', 'b', 'c']), {'none': ['z']}, permutations=100)
        comparison = compare_pathways(result.assign(cell_type='x'), result.assign(cell_type='x'))
        self.assertTrue(select_programs(comparison).empty)

    def test_donor(self):
        obs = pd.DataFrame({'DonorID': ['balanced'] * 12 + ['dominated'] * 12, 'Site': ['site1'] * 24,
                            'cell_type': ['a'] * 4 + ['b'] * 4 + ['c'] * 4 + ['a'] * 10 + ['b', 'c']})
        obs['cell_type_harmonized'] = obs.cell_type
        ranking = select_donor({'cite': obs, 'multiome': obs.copy()}, min_cells=1)
        self.assertEqual(ranking.loc[ranking.selected, 'DonorID'].iloc[0], 'balanced')


class WorkflowTests(unittest.TestCase):
    def test_feature_names_scoped_to_modality(self):
        import anndata as ad
        from src.single_donor_io import inspect_h5ad, read_rna_counts
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'collision.h5ad'
            obs = pd.DataFrame({'DonorID': ['d1'], 'Site': ['s1'], 'cell_type': ['NK']}, index=['cell1'])
            var = pd.DataFrame({'feature_types': ['GEX', 'ADT', 'GEX']}, index=['CD14', 'CD14', 'NKG7'])
            obj = ad.AnnData(sparse.csr_matrix([[2, 900, 3]]), obs=obs, var=var)
            obj.layers['counts'] = obj.X.copy()
            obj.write_h5ad(path)
            _, inspected_var, audit = inspect_h5ad(path)
            counts, genes = read_rna_counts(path, [0], inspected_var)
            self.assertEqual(genes.tolist(), ['CD14', 'NKG7'])
            np.testing.assert_array_equal(counts.toarray(), [[2, 3]])
            self.assertEqual(audit['repeated_feature_names'], [{'name': 'CD14', 'feature_types': ['GEX', 'ADT']}])
            obj.var['feature_types'] = ['GEX', 'GEX', 'GEX']
            obj.write_h5ad(path)
            with self.assertRaisesRegex(ValueError, 'Duplicate names within GEX'):
                inspect_h5ad(path)
            obj.var_names = ['A', 'B', 'C']
            doubled = ad.concat([obj, obj], merge='same')
            doubled.write_h5ad(path)
            with self.assertRaisesRegex(ValueError, 'Duplicate cell IDs'):
                inspect_h5ad(path)

    def test_end_to_end(self):
        import anndata as ad
        from src.single_donor_workflow import prepare_donor, characterize_donor, run_concordance, load_characterization
        with tempfile.TemporaryDirectory() as directory:
            root, paths = Path(directory), {}
            rng = np.random.default_rng(4)
            genes = [f'G{i:03d}' for i in range(80)]
            for dataset, n in [('cite', 18), ('multiome', 15)]:
                labels = np.repeat(['CD4+ T naive', 'CD14+ Mono', 'NK'], n)
                obs = pd.DataFrame({'DonorID': 'd1', 'Site': 's1', 'cell_type': labels},
                                   index=[f'{dataset}_{i}' for i in range(len(labels))])
                counts = rng.poisson(1, (len(labels), 81))
                for k in range(3):
                    counts[k*n:(k+1)*n, k*20:(k+1)*20] += rng.poisson(6, (n, 20))
                counts[:, -1] = 100000
                var = pd.DataFrame({'feature_types': ['GEX'] * 80 + ['ADT' if dataset == 'cite' else 'ATAC']},
                                   index=genes + ['OTHER'])
                obj = ad.AnnData(sparse.csr_matrix(counts * 3.14), obs=obs, var=var)
                obj.layers['counts'] = sparse.csr_matrix(counts)
                paths[dataset] = root / f'{dataset}.h5ad'
                obj.write_h5ad(paths[dataset])
            gmt = root / 'test.gmt'
            gmt.write_text('\n'.join(f'P{k}\ttest\t' + '\t'.join(genes[k*20:(k+1)*20]) for k in range(3)))
            output = root / 'results'
            donor, ranking, counts, data = prepare_donor(paths, output, min_cells=10)
            self.assertEqual(donor, 'd1')
            for x, obs, names in data.values():
                np.testing.assert_allclose(np.expm1(x.toarray()).sum(axis=1), 10000)
            characterize_donor(data, output, gmt, permutations=100)
            corr, pathways, followup, report = run_concordance(output, repeats=2, cap=12)
            self.assertEqual(len(corr), 3)
            self.assertTrue((corr.pearson > .8).all())
            self.assertTrue((output / 'rna_concordance/figures/expression_scatter.png').exists())
            self.assertEqual(json.loads((output / 'status.json').read_text())['rna_concordance'], 'complete')
            with (output / 'rna_cite/genes.csv').open('a') as handle:
                handle.write('stale\n')
            with self.assertRaisesRegex(ValueError, 'Changed/stale'):
                load_characterization(output)

    def test_notebooks(self):
        import ast
        import nbformat
        root = Path(__file__).resolve().parents[1]
        for filename in ['nb04_single_donor_rna_characterization.ipynb', 'nb05_rna_concordance.ipynb']:
            nb = nbformat.read(root / 'notebooks' / filename, as_version=4)
            nbformat.validate(nb)
            text = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'markdown')
            for header in ['## Scientific question', '## Analysis strategy', '## Main findings', '## Limitations']:
                self.assertIn(header, text)
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source)
                    # Real Colab outputs are now intentionally preserved in nb04/05.
                    self.assertFalse(any(o.output_type == 'error' for o in cell.outputs))


if __name__ == '__main__':
    unittest.main()
