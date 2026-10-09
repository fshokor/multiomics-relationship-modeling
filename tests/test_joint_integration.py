import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from src.joint_integration import combine_inputs,validate_missingness,train_model,evaluate_saved
from src.single_donor_io import sha256

class JointTests(unittest.TestCase):
    def inputs(self):
        rng=np.random.default_rng(9);n=80
        def obs(assay):
            return pd.DataFrame(dict(DonorID=np.repeat([str(i) for i in range(8)],10),
                Site='s',cell_type_harmonized=np.tile(['B cells','CD14 monocytes'],40),
                mapping_resolved=True,source_cell_id=[f'c{i}' for i in range(n)],assay=assay),
                index=[f'{assay}::c{i}' for i in range(n)])
        cite=ad.AnnData(sparse.csr_matrix(rng.poisson(3,(n,20)).astype(np.float32)),
            obs=obs('CITE'),var=pd.DataFrame(index=[f'G{i}' for i in range(20)]))
        cite.layers['counts']=cite.X.copy()
        cite.obsm['protein_counts']=pd.DataFrame(rng.poisson(3,(n,4)).astype(np.float32),
            index=cite.obs_names,columns=['CD14','CD19','CD3','CD4'])
        multi=ad.AnnData(sparse.csr_matrix(rng.poisson(2,(n,50)).astype(np.float32)),
            obs=obs('Multiome'),var=pd.DataFrame({'feature_types':['GEX']*20+['ATAC']*30},
            index=[f'G{i}' for i in range(20)]+[f'peak{i}' for i in range(30)]))
        multi.uns['n_genes']=20;multi.uns['n_regions']=30
        return cite,multi

    def test_missing_masks_training_recovery(self):
        data,audit=combine_inputs(*self.inputs())
        self.assertEqual(data.shape,(160,50));self.assertTrue(audit.included.all())
        self.assertEqual(data.obsm['protein_counts'][80:].nnz,0)
        self.assertEqual(data.X[:80,20:].nnz,0)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            data.write_h5ad(output/'model_input.h5ad')
            model=train_model(data,output,max_epochs=3,min_epochs=2,warmup_epochs=1,
                              patience=2,n_latent=3,accelerator='cpu')
            self.assertEqual(model.module.n_input_proteins,4)
            self.assertTrue(np.isfinite(data.obsm['X_joint']).all())
            import torch
            with torch.inference_mode():
                tensors=next(iter(model._make_data_loader(model.adata,batch_size=160,shuffle=False)))
                _,_,loss=model.module(tensors)
            self.assertTrue(torch.all(loss.reconstruction_loss['reconstruction_loss_accessibility'][:80]==0))
            self.assertTrue(torch.all(loss.reconstruction_loss['reconstruction_loss_protein'][80:]==0))
            self.assertGreater(float(loss.reconstruction_loss['reconstruction_loss_protein'][:80].sum()),0)
            saved=sha256(output/'model/model.pt')
            # Input saved before training deliberately lacks latent coordinates.
            evaluate_saved(output,cap=160)
            self.assertEqual(saved,sha256(output/'model/model.pt'))
            self.assertEqual(len(list((output/'figures').glob('*.png'))),9)
            self.assertTrue((output/'donor_celltype_alignment.csv').exists())
            self.assertEqual(json.loads((output/'training_summary.json').read_text())['final_kl_weight'],1)

    def test_reject_wrong_missingness(self):
        data,_=combine_inputs(*self.inputs())
        data.obs.loc[data.obs.index[0],'assay']='Multiome'
        with self.assertRaises(ValueError):validate_missingness(data)

    def test_notebook(self):
        import nbformat
        path=Path(__file__).resolve().parents[1]/'notebooks/nb13_joint_rna_atac_protein.ipynb'
        if not path.exists():self.skipTest('Notebook not generated yet')
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type=='code':ast.parse(c.source)
        exec(nb.cells[3].source,{'MODE':'LOAD'})

if __name__=='__main__':unittest.main()
