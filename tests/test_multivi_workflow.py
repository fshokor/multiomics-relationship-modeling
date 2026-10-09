import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from src.single_donor_io import sha256
from src.multivi_workflow import run_multivi, evaluate_saved


class MultiviTests(unittest.TestCase):
    def test_notebook_and_load_mode(self):
        import nbformat
        import runpy
        import sys
        root = Path(__file__).resolve().parents[1]
        nb = nbformat.read(root/'notebooks/nb12_multiome_multivi.ipynb', as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
        # LOAD dispatch must not import training modules or access raw inputs.
        exec(nb.cells[3].source, {'MODE':'LOAD'})
        sys.path.insert(0, str(root/'scripts'))
        generated = runpy.run_path(str(root/'scripts/build_nb12.py'))['cells']
        self.assertEqual([c.source for c in nb.cells], [''.join(c['source']) for c in generated])

    def test_paired_training_and_recovery(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cohort = root/'cohort'; cohort.mkdir()
            output = root/'output'
            donors = [str(i) for i in range(8)]
            obs = pd.DataFrame({'DonorID':np.repeat(donors,20), 'Site':'site1',
                'cell_type':np.tile(['B cells','CD14 monocytes'],80)}, index=[f'c{i}' for i in range(160)])
            # Include cross-modality name collision and a labelled control.
            genes = ['CD14']+[f'G{i}' for i in range(39)]
            proteins = [f'chr1:{i*100}-{i*100+50}' for i in range(50)]
            counts = rng.poisson(.6,(160,90)).astype(np.float32)
            counts[0,:40] = 0
            obj = ad.AnnData(sparse.csr_matrix(counts), obs=obs,
                var=pd.DataFrame({'feature_types':['GEX']*40+['ATAC']*50}, index=genes+proteins))
            obj.layers['counts'] = obj.X.copy()
            source = root/'multiome.h5ad'; obj.write_h5ad(source)
            pd.DataFrame({'donor':donors,'eligible':True}).to_csv(cohort/'donor_eligibility.csv',index=False)
            pd.DataFrame([{'dataset':'multiome','original_label':x,'harmonized_label':x}
                          for x in ['B cells','CD14 monocytes']]).to_csv(cohort/'mapping_audit.csv',index=False)
            (cohort/'status.json').write_text('{"state":"complete"}')
            (cohort/'protocol.json').write_text('{}')
            (cohort/'manifest.json').write_text(json.dumps(dict(
                protocol_sha256=sha256(cohort/'protocol.json'), source_sha256={'multiome':sha256(source)},
                outputs_sha256={x:sha256(cohort/x) for x in ['donor_eligibility.csv','mapping_audit.csv']})))
            run_multivi(source, cohort, output, n_hvgs=20, max_peaks=30, min_peak_cells=2, max_epochs=3, min_epochs=2,
                        warmup_epochs=1, patience=2, n_latent=3, accelerator='cpu')
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'complete')
            data = ad.read_h5ad(output/'model_input.h5ad')
            self.assertEqual(data.n_obs,159)
            self.assertEqual(int(data.uns['n_regions']),30)
            self.assertTrue((data.var.feature_types.iloc[:int(data.uns['n_genes'])]=='GEX').all())
            self.assertTrue((data.var.feature_types.iloc[int(data.uns['n_genes']):]=='ATAC').all())
            self.assertTrue(np.isfinite(data.obsm['X_MultiVI']).all())
            self.assertEqual(json.loads((output/'training_summary.json').read_text())['final_kl_weight'],1.)
            saved_hash = sha256(output/'model/model.pt')
            # Simulate evaluation recovery after model save but before embedding save.
            del data.obsm['X_MultiVI']; data.write_h5ad(output/'model_input.h5ad')
            evaluate_saved(output,cap=120)
            self.assertEqual(sha256(output/'model/model.pt'), saved_hash)
            table = pd.read_csv(output/'metric_comparison.csv')
            self.assertIn('cell_type_purity',table.metric.tolist())
            self.assertEqual(len(list((output/'figures').glob('*.png'))),10)
            with self.assertRaises(FileExistsError):
                run_multivi(source,cohort,output)


if __name__ == '__main__': unittest.main()
