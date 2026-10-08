"""RNA-only, unsupervised assay integration with explicit biological evaluation."""
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.single_donor_io import inspect_h5ad, read_rna_counts, sha256, write_json
from src.celltype_harmonization import apply_mapping
from src.expression_programs import normalize_counts
from src.pathway_enrichment import read_gmt
from src.atac_programs import balanced_standardize
from src.rna_integration_metrics import (evaluation_indices, baseline_metrics, group_alignment,
                                        gradient_agreement, bridge_decision)

MODULES = ['Inflammatory Response', 'TNF-alpha Signaling via NF-kB']


def get_shared_rna_features(inspected):
    return sorted(set.intersection(*(set(v.index[v.feature_types=='GEX'].astype(str)) for _,v,_ in inspected.values())))


def inspect_stored_x(path, var, n_rows=128):
    """A bounded diagnostic cannot prove the full object's transformation history."""
    import h5py
    from anndata.io import sparse_dataset
    cols=np.flatnonzero(var.feature_types.to_numpy()=='GEX')
    output={}
    with h5py.File(path,'r') as f:
        for field in ['X','layers/counts']:
            node=f[field]
            backed=sparse_dataset(node) if isinstance(node,h5py.Group) else node
            block=sparse.csr_matrix(backed[:n_rows,:])[:,cols]
            values=block.data
            output[field]=dict(sample_rows=block.shape[0], sample_min=float(min(0,values.min())) if len(values) else 0,
                               sample_max=float(values.max()) if len(values) else 0,
                               nonzero_integer_fraction=float(np.isclose(values,np.rint(values),atol=1e-6,rtol=0).mean()) if len(values) else 1,
                               median_gex_row_sum=float(np.median(np.asarray(block.sum(axis=1)).ravel())))
    output['interpretation']='Sample diagnostics only; .X transformation is not assumed. Integration starts from validated GEX counts.'
    return output


def prepare_shared_rna_counts(paths, cohort_root, discovery_root, output):
    import anndata as ad
    output,cohort_root,discovery_root=map(Path,[output,cohort_root,discovery_root])
    output.mkdir(parents=True,exist_ok=True)
    if json.loads((cohort_root/'status.json').read_text()).get('state')!='complete':
        raise ValueError('Complete nb09 before integration')
    protocol=json.loads((cohort_root/'protocol.json').read_text())
    run=json.loads((cohort_root/'manifest.json').read_text())
    for relative in ['donor_eligibility.csv','mapping_audit.csv']:
        if sha256(cohort_root/relative)!=run['outputs_sha256'][relative]:
            raise ValueError('nb09 audit fingerprint mismatch: '+relative)
    if sha256(cohort_root/'protocol.json')!=run['protocol_sha256']:
        raise ValueError('nb09 protocol changed')
    eligible=pd.read_csv(cohort_root/'donor_eligibility.csv',dtype={'donor':str})
    donors=sorted(eligible.loc[eligible.eligible.astype(str).str.lower().eq('true'),'donor'])
    if len(donors)!=8:
        raise ValueError(f'Expected the validated eight-donor cohort, found {len(donors)}; review cohort explicitly')
    mapping=pd.read_csv(cohort_root/'mapping_audit.csv')
    inspected={d:inspect_h5ad(p) for d,p in paths.items()}
    shared=get_shared_rna_features(inspected)
    matrices,metadata,audits=[],[],{}
    for d,path in paths.items():
        print('Preparing RNA only:',d,flush=True)
        # Streaming fingerprints detect changed source files, even if sizes match.
        if sha256(path)!=run['source_sha256'][d]:
            raise ValueError('Source differs from nb09: '+d)
        obs,var,source=inspected[d]
        part=mapping[mapping.dataset==d]
        if not set(obs.cell_type.astype(str))<=set(part.original_label):
            raise ValueError('nb09 mapping does not cover current labels')
        obs=apply_mapping(obs,mapping,d)
        mask=obs.DonorID.isin(donors)
        raw,genes=read_rna_counts(path,np.flatnonzero(mask),var)
        obs=obs.loc[mask].copy()
        if not np.isfinite(raw.data).all() or (raw.data<0).any() or not np.allclose(raw.data,np.rint(raw.data),atol=1e-6,rtol=0):
            raise ValueError('Integration requires raw nonnegative integer GEX counts')
        full_totals=np.asarray(raw.sum(axis=1)).ravel()
        raw=raw[:,pd.Index(genes).get_indexer(shared)].astype(np.float32)
        shared_totals=np.asarray(raw.sum(axis=1)).ravel()
        keep=shared_totals>0
        obs['included_shared_rna']=keep
        obs[['DonorID','Site','cell_type_original','cell_type_harmonized','included_shared_rna']].to_csv(output/f'{d}_cell_audit.csv',index_label='source_cell_id')
        obs=obs.loc[keep].copy()
        obs['source_cell_id']=obs.index.astype(str)
        obs['assay']='CITE' if d=='cite' else 'Multiome'
        obs['cell_type_harmonized']=obs.cell_type_harmonized.fillna('Unresolved').astype(str)
        obs['mapping_resolved']=obs.cell_type_harmonized.ne('Unresolved')
        obs['full_gex_counts']=full_totals[keep]
        obs['shared_gex_counts']=shared_totals[keep]
        obs['donor_role']=np.where(obs.DonorID.eq(protocol['discovery_donor']),'discovery','validation')
        obs.index=pd.Index(obs.assay+'::'+obs.source_cell_id,name='cell_id')
        matrices.append(raw[keep])
        metadata.append(obs[['source_cell_id','assay','DonorID','Site','cell_type_original','cell_type_harmonized',
                             'mapping_resolved','full_gex_counts','shared_gex_counts','donor_role']])
        audits[d]=dict(source=source,stored_values=inspect_stored_x(path,var),zero_shared_count_cells=int((~keep).sum()))
    counts=sparse.vstack(matrices,format='csr')
    obs=pd.concat(metadata)
    if not obs.index.is_unique:
        raise ValueError('Composite assay/cell identifiers must be unique')
    for field in ['assay','DonorID','Site','cell_type_original','cell_type_harmonized','donor_role']:
        obs[field]=obs[field].astype('category')
    data=ad.AnnData(normalize_counts(counts).astype(np.float32),obs=obs,var=pd.DataFrame(index=pd.Index(shared,name='gene')))
    data.layers['counts']=counts
    for field in ['DonorID','Site','cell_type_harmonized']:
        obs.groupby(['assay',field],observed=True).size().rename('n_cells').reset_index().to_csv(output/f'counts_by_{field}.csv',index=False)
    obs.groupby('assay',observed=True).size().rename('n_cells').to_csv(output/'counts_by_assay.csv')
    counts_by_type=obs.groupby(['cell_type_harmonized','assay'],observed=True).size().unstack(fill_value=0)
    write_json(output/'input_audit.json',dict(donors=donors,n_shared_genes=len(shared),
               n_common_cell_types=int((counts_by_type.gt(0).all(axis=1)&counts_by_type.index.to_series().ne('Unresolved')).sum()),
               inputs=audits,normalization='GEX intersection first; per-cell shared-count total 10000 then log1p, identically in both assays',
               unresolved_policy='Retained in unsupervised model, labelled Unresolved, excluded from biological preservation metrics'))
    pd.Series(shared,name='gene').to_csv(output/'shared_genes.csv',index=False)
    mapping.to_csv(output/'harmonization_mapping.csv',index=False)
    data.uns['log1p']={'base':None}
    return data,protocol


def select_shared_hvgs(data,n_top=3000,per_assay_cap=10000,seed=42):
    """Equal-size assay samples, assay-aware normalized-dispersion selection."""
    import scanpy as sc
    rng=np.random.default_rng(seed)
    groups=list(data.obs.groupby('assay',observed=True).indices.values())
    n=min(per_assay_cap,*(len(g) for g in groups))
    ids=np.sort(np.concatenate([rng.choice(g,n,replace=False) for g in groups]))
    sample=data[ids].copy()
    del sample.layers['counts']
    sc.pp.highly_variable_genes(sample,flavor='seurat',n_top_genes=min(n_top,data.n_vars),batch_key='assay')
    columns=['highly_variable','highly_variable_nbatches','means','dispersions_norm']
    data.var[columns]=sample.var[columns]
    data.uns['hvg_selection']={'method':'seurat log-normalized dispersion; batch_key assay', 'cells_per_assay':n,'seed':seed}
    if int(data.var.highly_variable.sum())<3:
        raise ValueError('Insufficient variable genes')
    return data.var[columns].copy()


def score_validated_modules(data,discovery_root,protocol):
    """Original GMT/universe; scores use expression, not corrected latent values."""
    discovery_root=Path(discovery_root)
    if sha256(discovery_root/'gsea/used_collection.gmt')!=protocol['gmt_sha256']:
        raise ValueError('Validated GMT differs from nb09')
    sets=read_gmt(discovery_root/'gsea/used_collection.gmt')
    scores=pd.DataFrame(index=data.obs_names)
    membership=[]
    for pathway in MODULES:
        genes=sorted(set(sets[pathway])&set(protocol['universe'])&set(data.var_names))
        values=data[:,genes].X.toarray()
        result=np.full(data.n_obs,np.nan)
        # Equal-weight donor/assay/type moments across resolved labels; global common
        # reference moments keep values comparable without forcing assay means equal.
        labels=(data.obs.assay.astype(str)+'|'+data.obs.DonorID.astype(str)+'|'+data.obs.cell_type_harmonized.astype(str)).to_numpy()
        ref=data.obs.cell_type_harmonized.isin(protocol['reference_types']).to_numpy()
        _,mean,sd,valid=balanced_standardize(values[ref],labels[ref])
        if valid.sum()>=10:
            result=((values[:,valid]-mean[valid])/sd[valid]).mean(axis=1)
        scores[pathway]=result
        membership.extend(dict(module=pathway,gene=g,variable=bool(v)) for g,v in zip(genes,valid))
    return scores,pd.DataFrame(membership)


def build_uncorrected_embedding(data,n_pcs=30,k=30,seed=42):
    import scanpy as sc
    n_pcs=min(n_pcs,int(data.var.highly_variable.sum())-1,data.n_obs-1)
    sc.pp.pca(data,n_comps=n_pcs,mask_var='highly_variable',zero_center=True,svd_solver='arpack',random_state=seed)
    sc.pp.neighbors(data,n_neighbors=k,use_rep='X_pca',key_added='baseline',random_state=seed)
    sc.tl.umap(data,neighbors_key='baseline',random_state=seed)
    data.obsm['X_umap_uncorrected']=data.obsm['X_umap'].copy()


def training_device_check(output,accelerator='auto'):
    """Exercise the CUDA linear forward/backward path before expensive preparation."""
    import torch
    if accelerator not in ['auto','cpu','gpu']:
        raise ValueError('accelerator must be auto, cpu or gpu')
    device='cuda' if accelerator=='gpu' or (accelerator=='auto' and torch.cuda.is_available()) else 'cpu'
    info=dict(torch_version=torch.__version__,torch_cuda=torch.version.cuda,
              requested_accelerator=accelerator,device=device)
    try:
        if device=='cuda':
            info['gpu']=torch.cuda.get_device_name(0)
            free,total=torch.cuda.mem_get_info()
            info.update(free_bytes=free,total_bytes=total)
        # Full FP32 is a conservative diagnostic, not a claimed cuBLAS repair.
        torch.set_float32_matmul_precision('highest')
        layer=torch.nn.Linear(3002,128).to(device)
        x=torch.randn(256,3002,device=device,requires_grad=True)
        layer(x).square().mean().backward()
        if device=='cuda': torch.cuda.synchronize()
        info['state']='passed'
        del layer,x
        if device=='cuda': torch.cuda.empty_cache()
    except RuntimeError as exc:
        info.update(state='failed',error=str(exc))
        write_json(Path(output)/'training_device.json',info)
        raise RuntimeError('GPU/PyTorch preflight failed before data preparation. Restart the Colab session and retry; if it persists, use accelerator="cpu" in a fresh run folder. See training_device.json. No model was trained.') from exc
    write_json(Path(output)/'training_device.json',info)
    return 'gpu' if device=='cuda' else 'cpu'


def train_shared_scvi(data,output,n_latent=30,max_epochs=300,seed=42,accelerator='auto',batch_size=256,
                      kl_warmup_epochs=50,min_epochs=100,early_stopping_patience=30):
    if not (0 <= kl_warmup_epochs < min_epochs <= max_epochs):
        raise ValueError('Require 0 <= kl_warmup_epochs < min_epochs <= max_epochs')
    import scvi
    scvi.settings.seed=seed
    # Exactly the same HVGs as baseline; raw counts, not normalized .X.
    model_data=data[:,data.var.highly_variable].copy()
    scvi.model.SCVI.setup_anndata(model_data,layer='counts',batch_key='assay')
    model=scvi.model.SCVI(model_data,n_latent=n_latent,n_layers=2,gene_likelihood='nb')
    model.train(max_epochs=max_epochs,min_epochs=min_epochs,early_stopping=True,
                early_stopping_patience=early_stopping_patience,
                early_stopping_warmup_epochs=kl_warmup_epochs,
                early_stopping_monitor='elbo_validation',check_val_every_n_epoch=1,
                log_every_n_steps=1,
                plan_kwargs={'n_epochs_kl_warmup':kl_warmup_epochs},
                train_size=.9,batch_size=batch_size,accelerator=accelerator,devices=1,precision='32-true')
    data.obsm['X_scVI']=model.get_latent_representation()
    model.save(str(Path(output)/'scvi_model'),overwrite=False,save_anndata=False)
    for name,history in model.history.items():
        if hasattr(history,'to_csv'):
            history.to_csv(Path(output)/f'training_{name}.csv')
    weights=model.history['kl_weight'].iloc[:,0]
    validation=model.history['elbo_validation'].iloc[:,0]
    write_json(Path(output)/'training_summary.json',dict(
        epochs_completed=len(validation),max_epochs=max_epochs,min_epochs=min_epochs,
        kl_warmup_epochs=kl_warmup_epochs,early_stopping_patience=early_stopping_patience,
        final_kl_weight=float(weights.iloc[-1]),full_kl_weight_reached=bool(weights.iloc[-1]>=0.999),
        best_validation_epoch=int(validation.astype(float).idxmin()),
        final_validation_elbo=float(validation.iloc[-1]),
        reached_epoch_limit=bool(len(validation)>=max_epochs),
        note='Final weights saved; inspect post-warmup ELBO and biological metrics. Stopping alone does not establish convergence.'))
    return model


def donor_rna_profiles(data,groups,min_cells=50):
    rows=[]
    # Correlate mean normalized shared RNA profiles, never latent coordinates.
    for key,ids in data.obs.groupby(['DonorID','cell_type_harmonized'],observed=True).indices.items():
        a=ids[data.obs.iloc[ids].assay.eq('CITE').to_numpy()]
        b=ids[data.obs.iloc[ids].assay.eq('Multiome').to_numpy()]
        r=np.nan
        if min(len(a),len(b))>=min_cells:
            ma=np.asarray(data.X[a].mean(axis=0)).ravel()
            mb=np.asarray(data.X[b].mean(axis=0)).ravel()
            if np.std(ma)>0 and np.std(mb)>0:
                r=float(np.corrcoef(ma,mb)[0,1])
        rows.append(dict(DonorID=key[0],cell_type_harmonized=key[1],rna_profile_pearson=r,
                         full_n_CITE=len(a),full_n_Multiome=len(b)))
    return groups.merge(pd.DataFrame(rows),on=['DonorID','cell_type_harmonized'],how='left',validate='one_to_one')


def evaluate_spaces(data,scores,output,k=30,cap=20000,min_cells=50,seed=42):
    output=Path(output)
    # One fixed sample, donor/assay balanced quota; all labels retained in embedding.
    ids=evaluation_indices(data.obs,cap,seed)
    ids=ids[data.obs.iloc[ids].mapping_resolved.to_numpy()]
    obs=data.obs.iloc[ids].copy()
    obs.to_csv(output/'evaluation_cells.csv',index_label='cell_id')
    metrics,types,donors,gradients={},[],[],[]
    for name,key in [('uncorrected','X_pca'),('integrated','X_scVI')]:
        z=data.obsm[key][ids]
        cells,_=baseline_metrics(z,obs,k=k,seed=seed)
        cells.to_csv(output/f'{name}_cell_metrics.csv',index_label='cell_id')
        numeric=cells.select_dtypes(include='number')
        metrics[name]=numeric.mean()
        cells.groupby('cell_type_harmonized',observed=True).mean(numeric_only=True).to_csv(output/f'{name}_metrics_by_celltype.csv')
        types.append(group_alignment(z,obs,['cell_type_harmonized'],min_cells,k).assign(space=name))
        donors.append(donor_rna_profiles(data,group_alignment(z,obs,['DonorID','cell_type_harmonized'],min_cells,k),min_cells).assign(space=name))
        gradients.append(gradient_agreement(z,obs,scores.iloc[ids],min_cells,k).assign(space=name))
    comparison=pd.DataFrame(metrics).rename_axis('metric').reset_index()
    comparison['change']=comparison.integrated-comparison.uncorrected
    comparison['interpretation']=comparison.metric.map(lambda m:'Higher mixing is conditional on preserved biology; donor/site mixing is diagnostic, not an optimization target' if any(w in m for w in ['entropy','assay','donor','site']) else 'Higher cell-type preservation preferred')
    comparison.to_csv(output/'metric_comparison.csv',index=False)
    types=pd.concat(types,ignore_index=True)
    donors=pd.concat(donors,ignore_index=True)
    types.to_csv(output/'celltype_alignment.csv',index=False)
    donors.to_csv(output/'donor_celltype_alignment.csv',index=False)
    gradients=pd.concat(gradients,ignore_index=True)
    gradients.to_csv(output/'cd14_gradient_agreement.csv',index=False)
    merged=gradients[gradients.space=='uncorrected'].merge(gradients[gradients.space=='integrated'],on=['donor','query_assay','module'],suffixes=('_uncorrected','_integrated'),validate='one_to_one')
    criteria=bridge_decision(comparison,types[types.space=='integrated'],donors[donors.space=='integrated'],merged)
    criteria.to_csv(output/'bridge_criteria.csv',index=False)
    return comparison,types,donors,gradients,criteria


def run_integration(paths,cohort_root,discovery_root,output,n_hvgs=3000,n_latent=30,max_epochs=300,seed=42,accelerator='auto',batch_size=256,
                      kl_warmup_epochs=50,min_epochs=100,early_stopping_patience=30):
    import scanpy as sc
    output=Path(output)
    if (output/'status.json').exists():
        raise FileExistsError('Use a fresh run folder; existing partial or completed run is preserved')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'status.json',{'state':'in_progress'})
    accelerator=training_device_check(output,accelerator)
    data,protocol=prepare_shared_rna_counts(paths,cohort_root,discovery_root,output)
    print('Selecting assay-aware HVGs',flush=True)
    hvg=select_shared_hvgs(data,n_hvgs,seed=seed)
    hvg.to_csv(output/'hvg_audit.csv',index_label='gene')
    pd.Series(data.var_names[data.var.highly_variable],name='gene').to_csv(output/'hvg_genes.csv',index=False)
    scores,membership=score_validated_modules(data,discovery_root,protocol)
    scores.to_csv(output/'module_scores.csv',index_label='cell_id')
    membership.to_csv(output/'module_membership.csv',index=False)
    print('Building uncorrected PCA/neighbors/UMAP',flush=True)
    build_uncorrected_embedding(data,n_pcs=n_latent,seed=seed)
    # Save baseline before potentially long training; no duplicate count matrix.
    np.save(output/'pca.npy',data.obsm['X_pca'])
    np.save(output/'umap_uncorrected.npy',data.obsm['X_umap_uncorrected'])
    print('Training assay-only scVI; no donor/site/cell-type covariates',flush=True)
    model=train_shared_scvi(data,output,n_latent,max_epochs,seed,accelerator,batch_size,
                            kl_warmup_epochs,min_epochs,early_stopping_patience)
    sc.pp.neighbors(data,n_neighbors=30,use_rep='X_scVI',key_added='integrated',random_state=seed)
    sc.tl.umap(data,neighbors_key='integrated',random_state=seed)
    data.obsm['X_umap_integrated']=data.obsm['X_umap'].copy()
    print('Evaluating the same sampled cells in both spaces',flush=True)
    results=evaluate_spaces(data,scores,output,seed=seed)
    from src.rna_integration_plots import make_figures
    figures=make_figures(data,scores,*results,output)
    # A single shared-RNA artifact retains normalized X and one raw counts layer.
    data.write_h5ad(output/'shared_rna.h5ad',compression='gzip')
    data.obs.to_csv(output/'cell_metadata.csv',index_label='cell_id')
    for key in ['X_scVI','X_umap_integrated']:
        np.save(output/(key+'.npy'),data.obsm[key])
    versions={p:importlib.metadata.version(p) for p in ['scanpy','scvi-tools','anndata','numpy','scipy','scikit-learn','torch']}
    write_json(output/'manifest.json',dict(seed=seed,n_hvgs=int(data.var.highly_variable.sum()),n_latent=n_latent,max_epochs=max_epochs,accelerator=accelerator,batch_size=batch_size,
        kl_warmup_epochs=kl_warmup_epochs,min_epochs=min_epochs,early_stopping_patience=early_stopping_patience,
        versions=versions,figures=figures,n_cells=data.n_obs,n_shared_genes=data.n_vars,
        cohort_protocol_sha256=sha256(Path(cohort_root)/'protocol.json'),
        integration='SCVI counts HVGs; batch_key assay only; NB likelihood; no labels_key; no donor/site correction',
        evaluation='Fixed assay/donor-quota sample <=20000; silhouette <=3000; neighborhood k=30; group minimum 50 per assay in evaluation sample',
        thresholds='Exploratory: mixing ratio >=0.8 and scaled centroid <=0.5; concern <0.5 or >1. Purity loss <=0.02, silhouette loss <=0.05. Gradient rho >=0.2 and change >=-0.05.',
        module_scaling='Frozen GMT/universe; shared-count normalization; common equal donor-assay-reference-type gene moments; not numerically identical to nb09',
        outputs_sha256={p.name:sha256(p) for p in output.glob('*.csv')}))
    write_json(output/'status.json',{'state':'complete','biological_review':'required; no automatic approval of totalVI/MultiVI'})
    return results
