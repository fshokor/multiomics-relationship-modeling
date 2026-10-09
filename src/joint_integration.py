"""Joint RNA/ATAC/protein MultiVI pilot from frozen nb11/nb12 raw model inputs."""
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.single_donor_io import sha256, write_json
from src.expression_programs import normalize_counts
from src.rna_integration_metrics import evaluation_indices, baseline_metrics, group_alignment


def check_counts(x):
    values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values,np.rint(values),atol=1e-6,rtol=0):
        raise ValueError('Expected finite raw nonnegative integer counts')


def combine_inputs(cite, multiome):
    import anndata as ad
    if not cite.obs_names.is_unique or not multiome.obs_names.is_unique:
        raise ValueError('Source cell IDs must be unique')
    n = int(multiome.uns['n_genes'])
    if not (multiome.var.feature_types.iloc[:n]=='GEX').all() or not (multiome.var.feature_types.iloc[n:]=='ATAC').all():
        raise ValueError('Multiome feature ordering mismatch')
    genes = sorted(set(cite.var_names)&set(multiome.var_names[:n]))
    if len(genes)<3: raise ValueError('Insufficient shared RNA genes')
    rna_c = sparse.csr_matrix(cite.layers['counts'][:,cite.var_names.get_indexer(genes)], dtype=np.float32)
    rna_m = sparse.csr_matrix(multiome.X[:,multiome.var_names.get_indexer(genes)], dtype=np.float32)
    peaks = sparse.csr_matrix(multiome.X[:,n:], dtype=np.float32)
    proteins = cite.obsm['protein_counts']
    if not isinstance(proteins,pd.DataFrame) or not proteins.columns.is_unique:
        raise ValueError('Require named unique proteins')
    if not proteins.index.equals(cite.obs_names): raise ValueError('Protein cell order mismatch')
    protein = sparse.csr_matrix(proteins.to_numpy(dtype=np.float32))
    for x in [rna_c,rna_m,peaks,protein]:check_counts(x)
    if set(cite.obs.DonorID.astype(str)) != set(multiome.obs.DonorID.astype(str)) or cite.obs.DonorID.nunique()!=8:
        raise ValueError('Require the same eight donors')
    obs_parts=[]
    for source,assay in [(cite,'CITE'),(multiome,'Multiome')]:
        obs=source.obs[['DonorID','Site','cell_type_harmonized','mapping_resolved','source_cell_id']].copy()
        for col in ['DonorID','Site','cell_type_harmonized','source_cell_id']:obs[col]=obs[col].astype(str)
        obs['assay']=assay
        obs['rna_observed']=True;obs['atac_observed']=assay=='Multiome';obs['protein_observed']=assay=='CITE'
        obs.index=pd.Index(assay+'::'+obs.source_cell_id,name='cell_id')
        obs_parts.append(obs)
    obs=pd.concat(obs_parts)
    if not obs.index.is_unique:raise ValueError('Composite cell IDs must be unique')
    rna=sparse.vstack([rna_c,rna_m],format='csr')
    atac=sparse.vstack([sparse.csr_matrix((cite.n_obs,peaks.shape[1])),peaks],format='csr',dtype=np.float32)
    protein=sparse.vstack([protein,sparse.csr_matrix((multiome.n_obs,protein.shape[1]))],format='csr',dtype=np.float32)
    rt=np.asarray(rna.sum(axis=1)).ravel();at=np.asarray(atac.sum(axis=1)).ravel();pt=np.asarray(protein.sum(axis=1)).ravel()
    keep=(rt>0)&np.where(obs.assay.eq('CITE'),pt>0,at>0)
    audit=pd.DataFrame({'cell_id':obs.index,'included':keep,'shared_rna_counts':rt,'atac_counts':at,'protein_counts':pt})
    obs=obs.loc[keep].copy();rna=rna[keep];atac=atac[keep];protein=protein[keep]
    if obs.groupby('assay').DonorID.nunique().min()!=8:raise ValueError('Filtering lost a donor')
    obs['shared_rna_counts']=rt[keep];obs['atac_counts']=at[keep];obs['protein_counts']=pt[keep]
    peak_names=multiome.var_names[n:].astype(str).tolist()
    if len(set(genes+peak_names))!=len(genes)+len(peak_names):raise ValueError('Feature name collision')
    data=ad.AnnData(sparse.hstack([rna,atac],format='csr'),obs=obs,
        var=pd.DataFrame({'feature_types':['GEX']*len(genes)+['ATAC']*len(peak_names)},index=genes+peak_names))
    data.uns['n_genes']=len(genes);data.uns['n_regions']=len(peak_names)
    data.uns['protein_names']=proteins.columns.to_numpy(dtype=str)
    data.obsm['protein_counts']=protein
    validate_missingness(data)
    return data,audit


def validate_missingness(data):
    n=int(data.uns['n_genes']);obs=data.obs
    r=np.asarray(data.X[:,:n].sum(axis=1)).ravel()>0
    a=np.asarray(data.X[:,n:].sum(axis=1)).ravel()>0
    p=np.asarray(data.obsm['protein_counts'].sum(axis=1)).ravel()>0
    if not r.all() or not np.array_equal(a,obs.assay.eq('Multiome')) or not np.array_equal(p,obs.assay.eq('CITE')):
        raise ValueError('Observed modality masks do not match capture; never replace measured zeros with missing data')


def model_modalities(data):
    import anndata as ad
    import mudata as md
    validate_missingness(data)
    n=int(data.uns['n_genes'])
    m=md.MuData({
        'rna':ad.AnnData(data.X[:,:n].copy(),obs=data.obs.copy(),var=data.var.iloc[:n].copy()),
        'atac':ad.AnnData(data.X[:,n:].copy(),obs=data.obs.copy(),var=data.var.iloc[n:].copy()),
        'protein':ad.AnnData(data.obsm['protein_counts'].copy(),obs=data.obs.copy(),
                            var=pd.DataFrame(index=data.uns['protein_names']))})
    if not m.obs_names.equals(data.obs_names):raise ValueError('MuData cell order changed')
    return m


def train_model(data, output, max_epochs=300, min_epochs=100, warmup_epochs=50,
                patience=30, seed=42, accelerator='auto', batch_size=128, n_latent=30):
    import scvi
    import torch
    if not 0 <= warmup_epochs < min_epochs <= max_epochs:
        raise ValueError('Require warmup_epochs < min_epochs <= max_epochs')
    output = Path(output)
    torch.set_float32_matmul_precision('highest')
    scvi.settings.seed = seed
    mdata = model_modalities(data)
    scvi.model.MULTIVI.setup_mudata(mdata, batch_key='assay', modalities={'rna_layer':'rna','atac_layer':'atac',
        'protein_layer':'protein','batch_key':'rna'})
    model = scvi.model.MULTIVI(mdata, n_genes=int(data.uns['n_genes']),
        n_regions=int(data.uns['n_regions']), n_latent=n_latent, n_hidden=128, gene_likelihood='nb')
    stopping = scvi.train.LoudEarlyStopping(monitor='elbo_validation', mode='min',
        patience=patience, warmup_epochs=warmup_epochs)
    model.train(max_epochs=max_epochs, min_epochs=min_epochs, train_size=.9,
        batch_size=batch_size, accelerator=accelerator, devices=1, precision='32-true',
        n_epochs_kl_warmup=warmup_epochs, early_stopping=False, callbacks=[stopping],
        check_val_every_n_epoch=1, log_every_n_steps=1, adversarial_mixing=True)
    # Persist model first, so downstream failures never require retraining.
    model.save(str(output/'model'), overwrite=False, save_anndata=False)
    data.obsm['X_joint'] = model.get_latent_representation()
    for name, table in model.history.items():
        if hasattr(table, 'to_csv'):
            table.to_csv(output/f'training_{name}.csv')
    history = model.history['elbo_validation'].iloc[:, 0].astype(float)
    weights = model.history.get('kl_weight')
    write_json(output/'training_summary.json', dict(epochs_completed=len(history),
        best_validation_epoch=int(history.idxmin()), final_validation_elbo=float(history.iloc[-1]),
        final_kl_weight=float(weights.iloc[-1, 0]) if weights is not None else None,
        reached_epoch_limit=len(history) >= max_epochs, warmup_epochs=warmup_epochs,
        note='Final weights, not restored best checkpoint. Review curves and biology.'))
    return model



def evaluate_saved(output, seed=42, cap=20000):
    import anndata as ad
    import scanpy as sc
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.rna_integration import donor_rna_profiles
    output=Path(output)
    data=ad.read_h5ad(output/'model_input.h5ad')
    validate_missingness(data)
    if 'X_joint' not in data.obsm:
        import scvi
        model=scvi.model.MULTIVI.load(str(output/'model'),adata=model_modalities(data),accelerator='cpu',device='auto')
        data.obsm['X_joint']=model.get_latent_representation()
    n=int(data.uns['n_genes'])
    rna=ad.AnnData(normalize_counts(data.X[:,:n]).astype(np.float32),obs=data.obs.copy(),var=data.var.iloc[:n].copy())
    sc.pp.pca(rna,n_comps=min(30,n-1,data.n_obs-1),svd_solver='arpack',random_state=seed)
    ids=evaluation_indices(data.obs,cap=cap,seed=seed)
    ids=ids[data.obs.iloc[ids].mapping_resolved.to_numpy()]
    obs=data.obs.iloc[ids].copy()
    obs.to_csv(output/'evaluation_cells.csv')
    if len(ids)<3:raise ValueError('Insufficient labelled evaluation cells')
    metrics,types,donors,depths={ },[],[],[]
    zspaces={'rna_pca':rna.obsm['X_pca'],'joint':data.obsm['X_joint']}
    for name,zall in zspaces.items():
        z=zall[ids]
        cells,_=baseline_metrics(z,obs,seed=seed)
        cells.to_csv(output/f'{name}_cell_metrics.csv')
        metrics[name]=cells.select_dtypes(include='number').mean()
        types.append(group_alignment(z,obs,['cell_type_harmonized']).assign(space=name))
        donor_groups=group_alignment(z,obs,['DonorID','cell_type_harmonized'])
        donors.append(donor_rna_profiles(rna,donor_groups).assign(space=name))
        for assay in ['CITE','Multiome']:
            mask=obs.assay.eq(assay).to_numpy()
            from scipy.stats import spearmanr
            fields=['shared_rna_counts','protein_counts' if assay=='CITE' else 'atac_counts']
            for field in fields:
                values=np.log1p(obs.loc[mask,field].to_numpy())
                rho=[spearmanr(z[mask,j],values).statistic for j in range(z.shape[1])]
                finite=np.asarray(rho)[np.isfinite(rho)]
                depths.append(dict(space=name,assay=assay,depth=field,max_absolute_spearman=float(np.max(np.abs(finite))) if len(finite) else None))
    comparison=pd.DataFrame(metrics).rename_axis('metric').reset_index()
    comparison['change']=comparison.joint-comparison.rna_pca
    comparison.to_csv(output/'metric_comparison.csv',index=False)
    pd.concat(types).to_csv(output/'celltype_alignment.csv',index=False)
    pd.concat(donors).to_csv(output/'donor_celltype_alignment.csv',index=False)
    pd.DataFrame(depths).to_csv(output/'depth_associations.csv',index=False)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    for name,zall in zspaces.items():
        subset=ad.AnnData(np.zeros((len(ids),1),dtype=np.float32),obs=obs.copy())
        subset.obsm['embedding']=zall[ids]
        sc.pp.neighbors(subset,use_rep='embedding',n_neighbors=min(30,len(ids)-1),random_state=seed)
        sc.tl.umap(subset,random_state=seed)
        coords=subset.obsm['X_umap']
        pd.DataFrame(coords,index=obs.index,columns=['UMAP1','UMAP2']).to_csv(output/f'{name}_umap.csv')
        for field in ['assay','cell_type_harmonized','DonorID','Site']:
            fig,ax=plt.subplots(figsize=(10,6))
            for j,label in enumerate(sorted(obs[field].astype(str).unique())):
                mask=obs[field].astype(str).eq(label).to_numpy()
                ax.scatter(coords[mask,0],coords[mask,1],s=2,alpha=.6,label=label,color=plt.get_cmap('tab20')(j%20))
            ax.set(title=f'{name}: {field}',xlabel='UMAP1',ylabel='UMAP2')
            ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8,markerscale=3)
            fig.tight_layout();fig.savefig(figures/f'{name}_{field}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for split in ['train','validation']:
        f=output/f'training_elbo_{split}.csv'
        if f.exists():
            table=pd.read_csv(f,index_col=0);axes[0].plot(table.index,table.iloc[:,0],label=split)
    axes[0].set(title='ELBO (lower is better)',xlabel='Epoch');axes[0].legend()
    f=output/'training_kl_weight.csv'
    if f.exists():
        table=pd.read_csv(f,index_col=0);axes[1].plot(table.index,table.iloc[:,0])
    axes[1].set(title='KL weight',xlabel='Epoch');fig.tight_layout()
    fig.savefig(figures/'training.png',dpi=150);plt.close(fig)
    write_json(output/'evaluation_status.json',dict(state='complete',n_evaluation_cells=len(ids),
        outputs_sha256={p.name:sha256(p) for p in output.glob('*.csv')},
        review='Descriptive pilot; no validated imputation, cell matching, or automatic acceptance.'))
    return comparison


def run_joint(cite_root, multiome_root, output, seed=42, **training):
    import anndata as ad
    cite_root,multiome_root,output=map(Path,(cite_root,multiome_root,output))
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose an empty output folder')
    manifests=[]
    for root in [cite_root,multiome_root]:
        if json.loads((root/'status.json').read_text()).get('state') not in ['trained','complete']:
            raise ValueError('Upstream training must have completed')
        manifests.append(json.loads((root/'manifest.json').read_text()))
    if manifests[0]['cohort_manifest_sha256']!=manifests[1]['cohort_manifest_sha256']:
        raise ValueError('Upstream cohort manifests differ')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'status.json',{'state':'preparing'})
    sources=[root/'model_input.h5ad' for root in [cite_root,multiome_root]]
    fingerprints={name:sha256(path) for name,path in zip(['cite','multiome'],sources)}
    cite,multiome=[ad.read_h5ad(path) for path in sources]
    data,audit=combine_inputs(cite,multiome)
    del cite,multiome
    audit.to_csv(output/'cell_audit.csv',index=False)
    data.obs.groupby(['assay','DonorID','Site'],observed=True).size().rename('n_cells').to_csv(output/'cohort_counts.csv')
    pd.Series(data.var_names[:int(data.uns['n_genes'])],name='gene').to_csv(output/'shared_genes.csv',index=False)
    write_json(output/'input_audit.json',dict(n_cells=data.n_obs,n_shared_genes=int(data.uns['n_genes']),
        n_peaks=int(data.uns['n_regions']),n_proteins=len(data.uns['protein_names']),
        excluded_zero_observed_cells=int((~audit.included).sum()),
        feature_scope='Intersection of nb11/nb12 RNA HVGs; frozen nb12 peaks and nb11 proteins. Not all shared RNA genes.',
        missingness='CITE ATAC and Multiome protein rows are storage placeholders masked by MultiVI; never measured zeros.',
        source_model_input_sha256=fingerprints))
    data.write_h5ad(output/'model_input.h5ad',compression='gzip')
    write_json(output/'manifest.json',dict(seed=seed,training=training,input_sha256=fingerprints,
        upstream_manifest_sha256={name:sha256(root/'manifest.json') for name,root in zip(['cite','multiome'],[cite_root,multiome_root])},
        versions={p:importlib.metadata.version(p) for p in ['scvi-tools','scanpy','anndata','mudata','torch']},
        model='New three-modality MultiVI, assay batch and adversarial mixing; NB RNA, hidden128, default modality penalty. No donor/site covariates or labels.'))
    write_json(output/'status.json',{'state':'training'})
    train_model(data,output,seed=seed,**training)
    data.write_h5ad(output/'model_input.h5ad',compression='gzip')
    write_json(output/'status.json',{'state':'trained'})
    evaluate_saved(output,seed=seed)
    write_json(output/'status.json',{'state':'complete','biological_review':'required'})
