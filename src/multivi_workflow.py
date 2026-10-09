"""Paired Multiome RNA/ATAC modeling, with independent saved-results evaluation."""
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from src.single_donor_io import inspect_h5ad, sha256, write_json
from src.celltype_harmonization import apply_mapping
from src.expression_programs import normalize_counts
from src.rna_integration_metrics import evaluation_indices, baseline_metrics


def prepare_multiome(source, cohort, output, n_hvgs=3000, max_peaks=20000, min_peak_cells=50, seed=42):
    import anndata as ad
    import scanpy as sc
    import h5py
    from anndata.io import sparse_dataset
    from sklearn.decomposition import TruncatedSVD
    source, cohort, output = map(Path, (source, cohort, output))
    if min_peak_cells < 1 or max_peaks < 3:
        raise ValueError('Require min_peak_cells >=1 and max_peaks >=3')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cohort/'manifest.json').read_text())
    if json.loads((cohort/'status.json').read_text()).get('state') != 'complete':
        raise ValueError('nb09 must be complete')
    for name in ['donor_eligibility.csv','mapping_audit.csv']:
        if sha256(cohort/name) != manifest['outputs_sha256'][name]:
            raise ValueError('nb09 fingerprint mismatch: '+name)
    if sha256(cohort/'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('nb09 protocol fingerprint mismatch')
    if sha256(source) != manifest['source_sha256']['multiome']:
        raise ValueError('Multiome source differs from nb09')
    eligible = pd.read_csv(cohort/'donor_eligibility.csv', dtype={'donor':str})
    donors = sorted(eligible.loc[eligible.eligible.astype(str).str.lower().eq('true'),'donor'])
    if len(set(donors)) != 8: raise ValueError('Expected eight eligible donors')
    obs, var, audit = inspect_h5ad(source)
    mapping = pd.read_csv(cohort/'mapping_audit.csv')
    if not set(obs.cell_type) <= set(mapping.loc[mapping.dataset.eq('multiome'),'original_label']):
        raise ValueError('Incomplete Multiome mapping')
    rows = np.flatnonzero(obs.DonorID.isin(donors))
    obs = apply_mapping(obs.iloc[rows].copy(), mapping, 'multiome')
    gene_cols = np.flatnonzero(var.feature_types.eq('GEX'))
    peak_cols = np.flatnonzero(var.feature_types.eq('ATAC'))
    if not len(peak_cols) or not var.index[peak_cols].is_unique:
        raise ValueError('Require unique genomic peak names')
    rna_blocks, peak_blocks = [], []
    with h5py.File(source,'r') as f:
        node = f['layers/counts']
        backed = sparse_dataset(node) if isinstance(node,h5py.Group) else node
        for start in range(0,len(rows),256):
            block = sparse.csr_matrix(backed[rows[start:start+256],:])
            if not np.isfinite(block.data).all() or (block.data<0).any() or not np.allclose(block.data,np.rint(block.data),atol=1e-6,rtol=0):
                raise ValueError('Require raw nonnegative integer RNA/peak counts')
            rna_blocks.append(block[:,gene_cols].astype(np.float32))
            peak_blocks.append(block[:,peak_cols].astype(np.float32))
    rna = sparse.vstack(rna_blocks,format='csr'); del rna_blocks
    peaks = sparse.vstack(peak_blocks,format='csr'); del peak_blocks
    peaks.eliminate_zeros()
    rna_total = np.asarray(rna.sum(axis=1)).ravel()
    peak_total = np.asarray(peaks.sum(axis=1)).ravel()
    eligible_cells = (rna_total>0)&(peak_total>0)
    detected = np.asarray((peaks[eligible_cells]>0).sum(axis=0)).ravel()
    candidates = np.flatnonzero(detected>=min_peak_cells)
    selected = candidates[np.argsort(-detected[candidates],kind='stable')[:max_peaks]]
    selected = np.sort(selected)
    if len(selected)<3: raise ValueError('Insufficient peaks after detection filter')
    peak_audit = pd.DataFrame({'peak':var.index[peak_cols], 'n_detected_cells':detected})
    peak_audit['selected'] = np.isin(np.arange(len(peak_cols)),selected)
    peak_audit.to_csv(output/'peak_audit.csv',index=False)
    peaks = peaks[:,selected].tocsr()
    keep = eligible_cells & (np.asarray(peaks.sum(axis=1)).ravel()>0)
    pd.DataFrame({'source_cell_id':obs.index,'rna_counts':rna_total,'atac_counts':peak_total,
                  'included':keep}).to_csv(output/'cell_audit.csv',index=False)
    obs = obs.loc[keep].copy(); rna=rna[keep]; peaks=peaks[keep]
    obs['source_cell_id']=obs.index.astype(str); obs['assay']='Multiome'
    obs['cell_type_harmonized']=obs.cell_type_harmonized.fillna('Unresolved').astype(str)
    obs['mapping_resolved']=obs.cell_type_harmonized.ne('Unresolved')
    obs['rna_counts']=rna_total[keep]; obs['atac_counts']=peak_total[keep]
    obs.index=pd.Index('Multiome::'+obs.source_cell_id,name='cell_id')
    normalized=ad.AnnData(normalize_counts(rna).astype(np.float32),obs=obs.copy(),
                          var=pd.DataFrame(index=var.index[gene_cols].astype(str)))
    rng=np.random.default_rng(seed)
    groups=list(obs.groupby('DonorID',observed=True).indices.values())
    if len(groups)!=8: raise ValueError('Filtering removed an entire donor')
    quota=min(2000,min(map(len,groups)))
    ids=np.sort(np.concatenate([rng.choice(g,quota,replace=False) for g in groups]))
    sample=normalized[ids].copy()
    sc.pp.highly_variable_genes(sample,flavor='seurat',n_top_genes=min(n_hvgs,normalized.n_vars))
    sample.var.to_csv(output/'hvg_audit.csv')
    hvg=sample.var.highly_variable.to_numpy()
    normalized=normalized[:,hvg].copy(); rna=rna[:,hvg]
    sc.pp.pca(normalized,n_comps=min(30,normalized.n_vars-1,normalized.n_obs-1),svd_solver='arpack',random_state=seed)
    binary=peaks.copy(); binary.data[:]=1
    totals=np.asarray(binary.sum(axis=1)).ravel()
    prevalence=np.asarray(binary.sum(axis=0)).ravel()
    tfidf=sparse.diags(1/totals)@binary@sparse.diags(np.log1p(len(obs)/np.maximum(prevalence,1)))
    # Drop LSI1 a priori; retain and report depth associations for the other dimensions.
    lsi=TruncatedSVD(n_components=min(31,len(obs)-1,peaks.shape[1]-1),random_state=seed).fit_transform(tfidf)
    features=list(normalized.var_names)+list(var.index[peak_cols[selected]].astype(str))
    if len(set(features))!=len(features): raise ValueError('Gene/peak name collision')
    data=ad.AnnData(sparse.hstack([rna,peaks],format='csr'),obs=obs,
        var=pd.DataFrame({'feature_types':['GEX']*rna.shape[1]+['ATAC']*peaks.shape[1]},index=features))
    data.uns['n_genes']=rna.shape[1];data.uns['n_regions']=peaks.shape[1]
    data.obsm['X_pca']=normalized.obsm['X_pca'];data.obsm['X_lsi']=lsi[:,1:]
    for field in ['DonorID','Site','cell_type_harmonized']:
        obs.groupby(field,observed=True).size().rename('n_cells').to_csv(output/f'counts_by_{field}.csv')
    write_json(output/'input_audit.json',dict(source=audit,donors=donors,n_cells=len(obs),
        n_genes=rna.shape[1],n_peaks=peaks.shape[1],excluded_cells=int((~keep).sum()),
        peak_selection='Detected in >= min_peak_cells; most prevalent up to max_peaks, stable ties, original order. May underrepresent rare states.',
        min_peak_cells=min_peak_cells,max_peaks=max_peaks,
        model_input='Raw GEX then raw genomic peak counts, paired by source row. No gene activity input.',
        baseline='RNA log1p CP10k before HVGs; binary peak TF-IDF SVD with first component excluded',
        covariates='None; donor/site diagnostic only'))
    return data


def model_modalities(data):
    """Reconstruct paired MuData in the persisted gene/peak/cell order."""
    import anndata as ad
    import mudata as md
    n = int(data.uns['n_genes'])
    rna = ad.AnnData(data.X[:,:n].copy(), obs=data.obs.copy(), var=data.var.iloc[:n].copy())
    atac = ad.AnnData(data.X[:,n:].copy(), obs=data.obs.copy(), var=data.var.iloc[n:].copy())
    result = md.MuData({'rna':rna, 'atac':atac})
    if not result.obs_names.equals(data.obs_names):
        raise ValueError('MuData cell order changed')
    return result


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
    scvi.model.MULTIVI.setup_mudata(mdata, modalities={'rna_layer':'rna','atac_layer':'atac'})
    model = scvi.model.MULTIVI(mdata, n_genes=int(data.uns['n_genes']),
        n_regions=int(data.uns['n_regions']), n_latent=n_latent, n_hidden=128, gene_likelihood='nb')
    stopping = scvi.train.LoudEarlyStopping(monitor='elbo_validation', mode='min',
        patience=patience, warmup_epochs=warmup_epochs)
    model.train(max_epochs=max_epochs, min_epochs=min_epochs, train_size=.9,
        batch_size=batch_size, accelerator=accelerator, devices=1, precision='32-true',
        n_epochs_kl_warmup=warmup_epochs, early_stopping=False, callbacks=[stopping],
        check_val_every_n_epoch=1, log_every_n_steps=1, adversarial_mixing=False)
    # Persist model first, so downstream failures never require retraining.
    model.save(str(output/'model'), overwrite=False, save_anndata=False)
    data.obsm['X_MultiVI'] = model.get_latent_representation()
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
    """Rebuild descriptive metrics and figures from saved data without model training."""
    import anndata as ad
    import scanpy as sc
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    data = ad.read_h5ad(output/'model_input.h5ad')
    if 'X_MultiVI' not in data.obsm:
        # Recovery if training saved model but stopped before writing the embedding.
        import scvi
        model = scvi.model.MULTIVI.load(str(output/'model'), adata=model_modalities(data), accelerator='cpu', device='auto')
        data.obsm['X_MultiVI'] = model.get_latent_representation()
    ids = evaluation_indices(data.obs, cap=cap, seed=seed)
    ids = ids[data.obs.iloc[ids].mapping_resolved.to_numpy()]
    data.obs.iloc[ids].to_csv(output/'evaluation_cells.csv')
    metrics, per_type, depth = {}, [], []
    obs = data.obs.iloc[ids]
    for name, key in [('rna_pca', 'X_pca'), ('atac_lsi', 'X_lsi'), ('MultiVI', 'X_MultiVI')]:
        z = data.obsm[key][ids]
        cells, neighbors = baseline_metrics(z, obs, seed=seed)
        columns = [c for c in cells if c.startswith(('cell_type_', 'donor_', 'site_')) and c != 'cell_type_harmonized']
        metrics[name] = cells[columns].mean()
        grouped = cells.groupby('cell_type_harmonized', observed=True)[columns].mean()
        grouped['n_evaluation_cells'] = cells.groupby('cell_type_harmonized', observed=True).size()
        per_type.append(grouped.reset_index().assign(space=name))
        for field in ['rna_counts', 'atac_counts']:
            correlations = [spearmanr(z[:, j], np.log1p(obs[field])).statistic for j in range(z.shape[1])]
            depth.append(dict(space=name, depth=field, max_absolute_latent_spearman=float(np.nanmax(np.abs(correlations)))))
    comparison = pd.DataFrame(metrics).rename_axis('metric').reset_index()
    comparison['change_vs_rna'] = comparison.MultiVI-comparison.rna_pca
    comparison['change_vs_atac'] = comparison.MultiVI-comparison.atac_lsi
    comparison.to_csv(output/'metric_comparison.csv', index=False)
    pd.concat(per_type).to_csv(output/'celltype_metrics.csv', index=False)
    pd.DataFrame(depth).to_csv(output/'depth_associations.csv', index=False)
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    # Same evaluation cells in both visualizations; no full-cohort duplicate graph.
    subset = data[ids].copy()
    for name, key in [('rna_pca', 'X_pca'), ('atac_lsi', 'X_lsi'), ('MultiVI', 'X_MultiVI')]:
        sc.pp.neighbors(subset, n_neighbors=min(30, len(ids)-1), use_rep=key, random_state=seed)
        sc.tl.umap(subset, random_state=seed)
        coords = subset.obsm['X_umap']
        pd.DataFrame(coords, index=obs.index, columns=['UMAP1','UMAP2']).to_csv(output/f'{name}_umap.csv')
        for field in ['cell_type_harmonized', 'DonorID', 'Site']:
            fig, ax = plt.subplots(figsize=(10, 6))
            for j, label in enumerate(sorted(obs[field].astype(str).unique())):
                mask = obs[field].astype(str).eq(label).to_numpy()
                ax.scatter(coords[mask,0], coords[mask,1], s=2, alpha=.6, label=label, color=plt.get_cmap('tab20')(j % 20))
            ax.set(title=f'{name}: {field}', xlabel='UMAP1', ylabel='UMAP2')
            ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', markerscale=3, fontsize=8)
            fig.tight_layout(); fig.savefig(figures/f'{name}_{field}.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for split in ['train','validation']:
        table = pd.read_csv(output/f'training_elbo_{split}.csv', index_col=0)
        axes[0].plot(table.index, table.iloc[:,0], label=split)
    axes[0].set(title='ELBO (lower is better)', xlabel='Epoch'); axes[0].legend()
    weight_path = output/'training_kl_weight.csv'
    if weight_path.exists():
        table = pd.read_csv(weight_path, index_col=0)
        axes[1].plot(table.index, table.iloc[:,0])
    axes[1].set(title='KL weight', xlabel='Epoch'); fig.tight_layout()
    fig.savefig(figures/'training.png', dpi=150); plt.close(fig)
    write_json(output/'evaluation_status.json', dict(state='complete', n_evaluation_cells=len(ids),
        outputs_sha256={p.name:sha256(p) for p in output.glob('*.csv')},
        note='Descriptive internal checks; donor/site mixing is not an optimization target. No cross-assay mapping.'))
    return comparison


def run_multivi(source, cohort, output, seed=42, n_hvgs=3000, max_peaks=20000, min_peak_cells=50, **training):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output folder; previous work is preserved')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'status.json', {'state':'preparing'})
    data = prepare_multiome(source, cohort, output, n_hvgs=n_hvgs, max_peaks=max_peaks, min_peak_cells=min_peak_cells, seed=seed)
    data.write_h5ad(output/'model_input.h5ad', compression='gzip')
    write_json(output/'manifest.json', dict(seed=seed, n_hvgs=int(data.uns['n_genes']), n_peaks=int(data.uns['n_regions']),
        max_peaks=max_peaks, min_peak_cells=min_peak_cells, n_cells=data.n_obs,
        training=training, versions={p:importlib.metadata.version(p) for p in ['scvi-tools','scanpy','anndata','torch']},
        source_sha256=sha256(source), cohort_manifest_sha256=sha256(Path(cohort)/'manifest.json'),
        model='MULTIVI paired NB RNA + peak accessibility; raw counts; no covariates/labels; no adversarial batch mixing; n_hidden=128'))
    write_json(output/'status.json', {'state':'training'})
    train_model(data, output, seed=seed, **training)
    data.write_h5ad(output/'model_input.h5ad', compression='gzip')
    write_json(output/'status.json', {'state':'trained', 'next':'evaluate_saved; no retraining needed'})
    evaluate_saved(output, seed=seed)
    write_json(output/'status.json', {'state':'complete', 'biological_review':'required'})
