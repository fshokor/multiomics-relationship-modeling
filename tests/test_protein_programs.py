import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.protein_programs import panel_mapping, centered_log_adt, read_adt_counts, analyze_protein, association


class ProteinTests(unittest.TestCase):
    def test_conservative_mapping(self):
        result = panel_mapping(['CD3', 'CD16', 'CD57', 'CD45RA', 'HLA-DR', 'CD14', 'IgD', 'IgG1-control'], ['CD3D', 'FCGR3A', 'B3GAT1', 'PTPRC', 'HLA-DRA', 'CD14', 'IGHD'])
        self.assertFalse(result.direct_match.iloc[:5].any())
        self.assertTrue(result.direct_match.iloc[5])
        self.assertFalse(result.is_control.iloc[6])
        self.assertTrue(result.is_control.iloc[7])

    def test_transform(self):
        result = centered_log_adt([[0, 3], [0, 0]])
        np.testing.assert_allclose(result, [[-np.log(2), np.log(2)], [0, 0]])

    def test_paired_counts_all_storage_formats(self):
        import anndata as ad
        obs = pd.DataFrame({'DonorID': 'd', 'Site': 's', 'cell_type': 'a'}, index=['a', 'b', 'c'])
        counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.h5ad'
            for matrix in [counts, sparse.csr_matrix(counts), sparse.csc_matrix(counts)]:
                obj = ad.AnnData(matrix, obs=obs, var=pd.DataFrame({'feature_types': ['GEX', 'ADT', 'ADT']}, index=['CD14', 'CD14', 'CD86']))
                obj.layers['counts'] = matrix
                obj.write_h5ad(path)
                result, names = read_adt_counts(path, obs.loc[['c', 'a']], chunk_size=1, progress=lambda x: None)
                np.testing.assert_array_equal(result, [[8, 9], [2, 3]])
                self.assertEqual(names.tolist(), ['CD14', 'CD86'])
            bad = obs.copy()
            bad.loc['a', 'Site'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                read_adt_counts(path, bad)

    def test_unmeasured_pathway_is_not_negative(self):
        obs = pd.DataFrame({'DonorID': 'd', 'Site': 's', 'cell_type_harmonized': ['a'] * 6 + ['b'] * 6, 'rna_library_counts': 100}, index=[f'c{i}' for i in range(12)])
        selected = pd.DataFrame({'cell_type': ['a'], 'pathway': ['E2F Targets']})
        tables, pairs = analyze_protein(sparse.csr_matrix(np.arange(24).reshape(12, 2)), ['A', 'B'], np.ones((12, 1)), ['CD14'], obs, selected, {'E2F Targets': ['A', 'B']}, {'A', 'B'}, min_rna_genes=2, progress=lambda x: None)
        self.assertEqual(tables['coverage'].n_direct_genes.iloc[0], 0)
        self.assertTrue(tables['associations'].empty)
        self.assertTrue(tables['evidence'].empty)
        self.assertEqual(pairs, [])

    def test_depth_confound(self):
        rng = np.random.default_rng(42)
        depth = rng.uniform(0, 4, 1000)
        obs = pd.DataFrame({'Site': 's', 'rna_library_counts': np.expm1(depth), 'adt_library_counts': np.expm1(depth)})
        result = association(depth + rng.normal(0, .1, 1000), depth + rng.normal(0, .1, 1000), obs)
        self.assertGreater(result['pearson'], .95)
        self.assertLess(abs(result['adjusted_pearson']), .1)

    def test_notebook(self):
        import nbformat
        nb = nbformat.read(Path(__file__).resolve().parents[1] / 'notebooks/nb07_pathway_protein_support.ipynb', as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertFalse(any(o.output_type == 'error' for o in cell.outputs))
