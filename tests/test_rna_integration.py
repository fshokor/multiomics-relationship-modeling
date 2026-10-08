import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
from src.rna_integration_metrics import (evaluation_indices,knn_indices,baseline_metrics,
                                        group_alignment,gradient_agreement,bridge_decision)
from src.rna_integration import prepare_shared_rna_counts, select_shared_hvgs, score_validated_modules, evaluate_spaces
from src.single_donor_io import sha256


class IntegrationTests(unittest.TestCase):
    def test_neighbors_remove_self_even_with_ties(self):
        z=np.zeros((10,3))
        ni=knn_indices(z,4)
        self.assertTrue(all(i not in row for i,row in enumerate(ni)))
        self.assertEqual(ni.shape,(10,4))

    def test_mixing_and_identity_are_distinct(self):
        rng=np.random.default_rng(2)
        types=np.repeat(['A','B'],100)
        assay=np.tile(np.repeat(['CITE','Multiome'],50),2)
        obs=pd.DataFrame({'assay':assay,'cell_type_harmonized':types,'DonorID':'d','Site':'s'})
        z=rng.normal(0,.1,(200,3))
        z[:,0]+=(types=='B')*20
        z[:,1]+=(assay=='Multiome')*5
        separated,_=baseline_metrics(z,obs,k=10)
        z[:,1]-=(assay=='Multiome')*5
        aligned,_=baseline_metrics(z,obs,k=10)
        self.assertGreater(aligned.opposite_assay_fraction.mean(),separated.opposite_assay_fraction.mean()+.2)
        self.assertGreater(aligned.cell_type_purity.mean(),.95)
        table=group_alignment(z,obs,['cell_type_harmonized'],min_cells=20,k=10)
        self.assertTrue(table.sufficient_cells.all())
        self.assertTrue((table.mixing_ratio>.6).all())
        small=group_alignment(z[:110],obs.iloc[:110],['cell_type_harmonized'],min_cells=20)
        self.assertFalse(small.loc[small.cell_type_harmonized=='B','sufficient_cells'].iloc[0])

    def test_gradient_requires_local_agreement(self):
        values=np.tile(np.linspace(-2,2,100),2)
        obs=pd.DataFrame({'assay':np.repeat(['CITE','Multiome'],100),'DonorID':'d','cell_type_harmonized':'CD14 monocytes'})
        z=np.column_stack([values,np.zeros(200)])
        result=gradient_agreement(z,obs,pd.DataFrame({'module':values}),min_cells=20,k=5)
        self.assertTrue((result.spearman>.95).all())
        self.assertEqual(len(result),2)

    def test_bad_biology_is_a_concern_despite_mixing(self):
        comparison=pd.DataFrame({'metric':['opposite_assay_fraction','cell_type_purity','cell_type_silhouette'],
                                 'uncorrected':[.1,.95,.6],'integrated':[.5,.5,.1]})
        types=pd.DataFrame({'cell_type_harmonized':['A'],'sufficient_cells':[True],'alignment_status':['supported']})
        donors=pd.DataFrame({'sufficient_cells':[True],'scaled_centroid_distance':[.1],'rna_profile_pearson':[.8]})
        gradients=pd.DataFrame({'module':['P'],'spearman_integrated':[.4],'spearman_uncorrected':[.4]})
        criteria=bridge_decision(comparison,types,donors,gradients)
        self.assertEqual(criteria.loc[criteria.criterion=='Cell identity preservation','status'].iloc[0],'concern')

    def test_cohort_loading_evaluation_and_outputs(self):
        import anndata as ad
        from src.rna_integration_plots import make_figures
        rng=np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); cohort=root/'cohort'; discovery=root/'discovery'; output=root/'out'
            cohort.mkdir(); (discovery/'gsea').mkdir(parents=True)
            genes=[f'G{i}' for i in range(40)]
            gmt=discovery/'gsea/used_collection.gmt'
            gmt.write_text('Inflammatory Response\tx\t'+'\t'.join(genes[:20])+'\nTNF-alpha Signaling via NF-kB\tx\t'+'\t'.join(genes[10:30]))
            donors=[f'd{i}' for i in range(8)]
            pd.DataFrame({'donor':donors,'eligible':True}).to_csv(cohort/'donor_eligibility.csv',index=False)
            mapping=pd.DataFrame([dict(dataset=d,original_label=label,harmonized_label=label) for d in ['cite','multiome'] for label in ['CD14 monocytes','B cells']])
            mapping.to_csv(cohort/'mapping_audit.csv',index=False)
            protocol=dict(discovery_donor='d0',reference_types=['CD14 monocytes','B cells'],universe=genes,gmt_sha256=sha256(gmt))
            (cohort/'protocol.json').write_text(json.dumps(protocol))
            (cohort/'status.json').write_text('{"state":"complete"}')
            paths={}
            for d in ['cite','multiome']:
                n=160
                obs=pd.DataFrame({'DonorID':np.repeat(donors,20),'Site':'s','cell_type':np.tile(np.repeat(['CD14 monocytes','B cells'],10),8)},index=[f'barcode{i}' for i in range(n)])
                counts=rng.poisson(2,(n,41))
                obj=ad.AnnData(sparse.csr_matrix(np.log1p(counts)),obs=obs,var=pd.DataFrame({'feature_types':['GEX']*40+['ADT']},index=genes+['ADT']))
                obj.layers['counts']=sparse.csr_matrix(counts)
                path=root/f'{d}.h5ad'; obj.write_h5ad(path); paths[d]=path
            manifest=dict(protocol_sha256=sha256(cohort/'protocol.json'),outputs_sha256={p:sha256(cohort/p) for p in ['donor_eligibility.csv','mapping_audit.csv']},source_sha256={d:sha256(p) for d,p in paths.items()})
            (cohort/'manifest.json').write_text(json.dumps(manifest))
            data,protocol=prepare_shared_rna_counts(paths,cohort,discovery,output)
            self.assertEqual(data.shape,(320,40))
            self.assertTrue(data.obs_names.is_unique)
            self.assertEqual(set(data.obs.assay),{'CITE','Multiome'})
            self.assertNotIn('ADT',data.var_names)
            scores,membership=score_validated_modules(data,discovery,protocol)
            self.assertTrue(np.isfinite(scores.to_numpy()).all())
            z=rng.normal(size=(320,5))
            data.obsm['X_pca']=z; data.obsm['X_scVI']=z.copy()
            data.obsm['X_umap_uncorrected']=z[:,:2]; data.obsm['X_umap_integrated']=z[:,:2]
            results=evaluate_spaces(data,scores,output,cap=320,min_cells=5,k=5)
            np.testing.assert_allclose(results[0]['change'].dropna(),0,atol=1e-10)
            self.assertTrue(results[0].set_index('metric').loc['site_silhouette',['uncorrected','integrated']].isna().all())
            self.assertTrue((output/'bridge_criteria.csv').exists())
            names=make_figures(data,scores,*results,output)
            self.assertEqual(len(names),16)
            self.assertTrue((output/'figures'/names[0]).exists())
            if importlib.util.find_spec('scanpy'):
                hvgs=select_shared_hvgs(data,n_top=20,per_assay_cap=100)
                self.assertTrue(hvgs.highly_variable.any())
                from src.rna_integration import build_uncorrected_embedding
                build_uncorrected_embedding(data,n_pcs=5,k=5)
                self.assertEqual(data.obsm['X_umap_uncorrected'].shape,(320,2))
            data.write_h5ad(output/'shared_rna.h5ad')
            restored=ad.read_h5ad(output/'shared_rna.h5ad')
            self.assertEqual(restored.shape,data.shape)
            (cohort/'mapping_audit.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                prepare_shared_rna_counts(paths,cohort,discovery,root/'other')

    def test_notebook(self):
        import nbformat
        nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/nb10_shared_rna_integration.ipynb',as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code': ast.parse(cell.source)

    @unittest.skipUnless(importlib.util.find_spec('scvi'), 'scvi-tools not installed locally; real model training is a Colab validation step')
    def test_scvi_training_and_model_save(self):
        import anndata as ad
        from src.rna_integration import train_shared_scvi
        rng=np.random.default_rng(1)
        data=ad.AnnData(sparse.csr_matrix(rng.poisson(2,(100,30)).astype(np.float32)),
                        obs=pd.DataFrame({'assay':np.repeat(['CITE','Multiome'],50)},index=[f'c{i}' for i in range(100)]))
        data.layers['counts']=data.X.copy()
        data.var['highly_variable']=True
        with tempfile.TemporaryDirectory() as directory:
            from src.rna_integration import training_device_check
            self.assertEqual(training_device_check(Path(directory), 'cpu'), 'cpu')
            self.assertEqual(json.loads((Path(directory)/'training_device.json').read_text())['state'],'passed')
            train_shared_scvi(data,Path(directory),n_latent=3,max_epochs=4,kl_warmup_epochs=1,min_epochs=3,accelerator='cpu')
            self.assertEqual(data.obsm['X_scVI'].shape,(100,3))
            self.assertTrue((Path(directory)/'scvi_model').is_dir())
            summary=json.loads((Path(directory)/'training_summary.json').read_text())
            self.assertTrue(summary['full_kl_weight_reached'])
            self.assertGreaterEqual(summary['epochs_completed'],3)


if __name__=='__main__': unittest.main()
