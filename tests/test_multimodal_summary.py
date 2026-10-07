import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.multimodal_summary import site_summary, unique


class SummaryTests(unittest.TestCase):
    def test_missing_site_is_not_zero_or_positive(self):
        frame = pd.DataFrame(dict(cell_type=['a']*3, pathway=['p']*3, adt=['CD14']*3,
                                  scope=['target_by_site']*3, group=['s1','s2','s3'],
                                  adjusted_pearson=[.2, -.1, np.nan]))
        row = site_summary(frame).iloc[0]
        self.assertEqual(row.n_testable_sites, 2)
        self.assertEqual(row.n_sites, 3)
        self.assertEqual(row.n_positive_sites, 1)
        self.assertIn('s3: untested', row.site_values)

    def test_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            unique(pd.DataFrame({'key': ['a','a']}), ['key'], 'test')

    def test_notebook(self):
        import nbformat
        nb = nbformat.read(Path(__file__).resolve().parents[1] / 'notebooks/nb08_multilayer_pathway_states.ipynb', as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
