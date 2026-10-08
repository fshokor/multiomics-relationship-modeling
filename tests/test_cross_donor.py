import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.cross_donor import pseudobulk, run_cross_donor
from src.single_donor_io import sha256


class CrossDonorTests(unittest.TestCase):
    def test_count_sums_and_mean_are_different(self):
        counts = sparse.csr_matrix([[1, 3], [5, 1], [2, 2]])
        obs = pd.DataFrame({'DonorID': ['a']*3, 'Site': ['s1','s1','s2'], 'cell_type_harmonized': ['t']*3})
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            pseudobulk(counts, obs, ['A','B'], p, min_cells=2)
            np.testing.assert_array_equal(sparse.load_npz(p/'raw_count_sums.npz').toarray(), [[6,4],[2,2]])
            samples = pd.read_csv(p/'samples.csv')
            self.assertEqual(samples.eligible_min_cells.tolist(), [True, False])
            self.assertFalse(np.allclose(sparse.load_npz(p/'sum_log1p_cp10k.npz').toarray()[0], sparse.load_npz(p/'mean_cell_log1p_cp10k.npz').toarray()[0]))
            with self.assertRaisesRegex(ValueError, 'integer'):
                pseudobulk(counts * .3, obs, ['A','B'], p)

    def test_complete_two_donor_pipeline(self):
        import anndata as ad
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discovery = root/'discovery'
            (discovery/'gsea').mkdir(parents=True)
            (discovery/'rna_concordance').mkdir()
            genes = ['CD14', 'C5AR1', 'ICAM1'] + [f'G{i}' for i in range(57)]
            pd.DataFrame({'gene': genes, 'included_in_comparison': True}).to_csv(discovery/'rna_concordance/shared_gene_universe.csv', index=False)
            (discovery/'gsea/used_collection.gmt').write_text('Inflammatory Response\tx\t'+'\t'.join(genes[:25])+'\nTNF-alpha Signaling via NF-kB\tx\t'+'\t'.join(genes[2:27])+'\n')
            manifest = {'donor': 'd0', 'retained_cell_types': ['CD14 monocytes', 'NK cells'], 'artifact_sha256': {p: sha256(discovery/p) for p in ['gsea/used_collection.gmt','rna_concordance/shared_gene_universe.csv']}}
            (discovery/'manifest.json').write_text(json.dumps(manifest))
            paths = {}
            rng = np.random.default_rng(1)
            for capture in ['cite','multiome']:
                donor = np.repeat(['d0','d1'], 40)
                types = np.tile(np.repeat(['CD14+ Mono','NK'],20),2)
                obs = pd.DataFrame({'DonorID': donor, 'Site': np.tile(['s1','s2'],40), 'cell_type': types, 'ATAC_nCount_peaks': rng.integers(100,200,80)}, index=[f'{capture}_{i}' for i in range(80)])
                counts = rng.poisson(2,(80,60))
                counts[types == 'CD14+ Mono', :25] += rng.poisson(5, (40,25))
                if capture == 'cite':
                    stored = np.column_stack([counts, counts[:,:3]+1, rng.poisson(3,80)])
                    var = pd.DataFrame({'feature_types': ['GEX']*60+['ADT']*4}, index=genes+['CD14','CD88','CD54','CD86'])
                else:
                    stored = counts
                    var = pd.DataFrame({'feature_types': ['GEX']*60}, index=genes)
                obj = ad.AnnData(sparse.csr_matrix(stored), obs=obs, var=var)
                obj.layers['counts'] = obj.X.copy()
                if capture == 'multiome':
                    obj.obsm['ATAC_gene_activity'] = sparse.csr_matrix(np.log1p(counts))
                    obj.uns['ATAC_gene_activity_var_names'] = np.asarray(genes)
                paths[capture] = root/f'{capture}.h5ad'
                obj.write_h5ad(paths[capture])
            evidence, summary = run_cross_donor(paths, discovery, root/'output', min_cells=5, permutations=100)
            self.assertEqual(len(evidence),4)
            self.assertEqual(set(evidence.role), {'discovery','validation'})
            self.assertTrue((summary.n_eligible_validation_donors == 1).all())
            self.assertTrue((root/'output/donor_d1/pseudobulk_cite/raw_count_sums.npz').exists())
            self.assertTrue((root/'output/donor_evidence_1.png').exists())
            self.assertEqual(json.loads((root/'output/status.json').read_text())['state'],'complete')
            with self.assertRaises(FileExistsError):
                run_cross_donor(paths, discovery, root/'output')

    def test_notebook(self):
        import nbformat
        path = Path(__file__).resolve().parents[1]/'notebooks/nb09_cross_donor_validation.ipynb'
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
