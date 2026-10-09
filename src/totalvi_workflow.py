"""Unsupervised paired CITE RNA/protein model; training and evaluation are separable."""
import json
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.single_donor_io import inspect_h5ad, read_rna_counts, sha256, write_json
from src.protein_programs import read_adt_counts, panel_mapping
from src.celltype_harmonization import apply_mapping
from src.expression_programs import normalize_counts
from src.rna_integration_metrics import evaluation_indices, baseline_metrics


def prepare_cite(source, cohort, output, n_hvgs=3000, seed=42):
    import anndata as ad
    import scanpy as sc
    source, cohort, output = map(Path, (source, cohort, output))
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cohort/'manifest.json').read_text())
    if json.loads((cohort/'status.json').read_text()).get('state') != 'complete':
        raise ValueError('nb09 must be complete')
    for name in ['donor_eligibility.csv', 'mapping_audit.csv']:
        if sha256(cohort/name) != manifest['outputs_sha256'][name]:
            raise ValueError('nb09 fingerprint mismatch: '+name)
    if sha256(cohort/'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('nb09 protocol fingerprint mismatch')
    if sha256(source) != manifest['source_sha256']['cite']:
        raise ValueError('CITE source differs from nb09')
    eligibility = pd.read_csv(cohort/'donor_eligibility.csv', dtype={'donor': str})
    donors = sorted(eligibility.loc[eligibility.eligible.astype(str).str.lower().eq('true'), 'donor'])
    if len(set(donors)) != 8:
        raise ValueError('Expected the validated eight donors')
    obs, var, audit = inspect_h5ad(source)
    mapping = pd.read_csv(cohort/'mapping_audit.csv')
    if not set(obs.cell_type) <= set(mapping.loc[mapping.dataset.eq('cite'), 'original_label']):
        raise ValueError('Incomplete CITE label mapping')
    rows = np.flatnonzero(obs.DonorID.isin(donors))
    obs = obs.iloc[rows].copy()
    counts, genes = read_rna_counts(source, rows, var)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.allclose(counts.data, np.rint(counts.data), atol=1e-6, rtol=0):
        raise ValueError('RNA must contain raw nonnegative integer counts')
    proteins, names = read_adt_counts(source, obs)
    panel = panel_mapping(names, genes)
    panel['total_counts'] = proteins.sum(axis=0)
    panel['included'] = ~panel.is_control & panel.total_counts.gt(0)
    panel.to_csv(output/'protein_panel.csv', index=False)
    proteins = proteins[:, panel.included].astype(np.float32)
    names = names[panel.included.to_numpy()]
    if len(names) < 2:
        raise ValueError('Insufficient non-control proteins')
    rna_total = np.asarray(counts.sum(axis=1)).ravel()
    adt_total = proteins.sum(axis=1)
    keep = (rna_total > 0) & (adt_total > 0)
    pd.DataFrame({'source_cell_id': obs.index, 'rna_counts': rna_total,
                  'adt_counts': adt_total, 'included': keep}).to_csv(output/'cell_audit.csv', index=False)
    obs = apply_mapping(obs, mapping, 'cite').loc[keep].copy()
    obs['source_cell_id'] = obs.index.astype(str)
    obs['assay'] = 'CITE'
    obs['cell_type_harmonized'] = obs.cell_type_harmonized.fillna('Unresolved').astype(str)
    obs['mapping_resolved'] = obs.cell_type_harmonized.ne('Unresolved')
    obs['rna_counts'] = rna_total[keep]
    obs['adt_counts'] = adt_total[keep]
    obs.index = pd.Index('CITE::'+obs.source_cell_id, name='cell_id')
    counts = counts[keep].astype(np.float32)
    data = ad.AnnData(normalize_counts(counts).astype(np.float32), obs=obs,
                      var=pd.DataFrame(index=pd.Index(genes, name='gene')))
    data.layers['counts'] = counts
    data.obsm['protein_counts'] = pd.DataFrame(proteins[keep], index=obs.index, columns=names)
    # Equal donor quotas for feature selection; donor is not a model covariate.
    rng = np.random.default_rng(seed)
    groups = list(obs.groupby('DonorID', observed=True).indices.values())
    quota = min(2000, min(map(len, groups)))
    ids = np.sort(np.concatenate([rng.choice(g, quota, replace=False) for g in groups]))
    sample = data[ids].copy()
    sc.pp.highly_variable_genes(sample, flavor='seurat', n_top_genes=min(n_hvgs, data.n_vars))
    sample.var.to_csv(output/'hvg_audit.csv')
    data = data[:, sample.var.highly_variable.to_numpy()].copy()
    if data.n_vars < 3:
        raise ValueError('Insufficient HVGs')
    # X is normalized over all GEX before HVG selection; training uses raw counts.
    for field in ['DonorID', 'Site', 'cell_type_harmonized']:
        obs.groupby(field, observed=True).size().rename('n_cells').to_csv(output/f'counts_by_{field}.csv')
    write_json(output/'input_audit.json', dict(source=audit, donors=donors, n_cells=data.n_obs,
        n_hvgs=data.n_vars, n_proteins=len(names), excluded_zero_modality_cells=int((~keep).sum()),
        source_sha256=manifest['source_sha256']['cite'], cohort_manifest_sha256=sha256(cohort/'manifest.json'),
        model_covariates='None; no donor/site regression or label supervision',
        hvg_selection='Seurat dispersion on log1p CP10k all-GEX RNA; equal donor quotas',
        normalization='Model: raw RNA and ADT counts. X: log1p CP10k RNA for baseline only.'))
    return data


def train_model(data, output, max_epochs=300, min_epochs=100, warmup_epochs=50,
                patience=30, seed=42, accelerator='auto', batch_size=128, n_latent=30):
    import scvi
    import torch
    if not 0 <= warmup_epochs < min_epochs <= max_epochs:
        raise ValueError('Require warmup_epochs < min_epochs <= max_epochs')
    output = Path(output)
    torch.set_float32_matmul_precision('highest')
    scvi.settings.seed = seed
    scvi.model.TOTALVI.setup_anndata(data, layer='counts', protein_expression_obsm_key='protein_counts')
    model = scvi.model.TOTALVI(data, n_latent=n_latent, gene_likelihood='nb')
    model.train(max_epochs=max_epochs, min_epochs=min_epochs, train_size=.9,
        batch_size=batch_size, accelerator=accelerator, devices=1, precision='32-true',
        n_epochs_kl_warmup=warmup_epochs, early_stopping=True,
        early_stopping_warmup_epochs=warmup_epochs, early_stopping_patience=patience,
        early_stopping_monitor='elbo_validation', check_val_every_n_epoch=1, log_every_n_steps=1,
        reduce_lr_on_plateau=False)
    # Persist model first, so downstream failures never require retraining.
    model.save(str(output/'model'), overwrite=False, save_anndata=False)
    data.obsm['X_totalVI'] = model.get_latent_representation()
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
    if 'X_totalVI' not in data.obsm:
        # Recovery if training saved model but stopped before writing the embedding.
        import scvi
        model = scvi.model.TOTALVI.load(str(output/'model'), adata=data, accelerator='cpu', device='auto')
        data.obsm['X_totalVI'] = model.get_latent_representation()
    sc.pp.pca(data, n_comps=min(30, data.n_vars-1, data.n_obs-1), svd_solver='arpack', random_state=seed)
    ids = evaluation_indices(data.obs, cap=cap, seed=seed)
    ids = ids[data.obs.iloc[ids].mapping_resolved.to_numpy()]
    data.obs.iloc[ids].to_csv(output/'evaluation_cells.csv')
    metrics, per_type, depth, protein = {}, [], [], []
    obs = data.obs.iloc[ids]
    logged = np.log1p(data.obsm['protein_counts'].iloc[ids].to_numpy())
    # CLR only for observed-protein diagnostic, never model input.
    clr = logged-logged.mean(axis=1, keepdims=True)
    for name, key in [('rna_pca', 'X_pca'), ('totalVI', 'X_totalVI')]:
        z = data.obsm[key][ids]
        cells, neighbors = baseline_metrics(z, obs, seed=seed)
        columns = [c for c in cells if c.startswith(('cell_type_', 'donor_', 'site_')) and c != 'cell_type_harmonized']
        metrics[name] = cells[columns].mean()
        grouped = cells.groupby('cell_type_harmonized', observed=True)[columns].mean()
        grouped['n_evaluation_cells'] = cells.groupby('cell_type_harmonized', observed=True).size()
        per_type.append(grouped.reset_index().assign(space=name))
        for field in ['rna_counts', 'adt_counts']:
            correlations = [spearmanr(z[:, j], np.log1p(obs[field])).statistic for j in range(z.shape[1])]
            depth.append(dict(space=name, depth=field, max_absolute_latent_spearman=float(np.nanmax(np.abs(correlations)))))
        predicted = clr[neighbors].mean(axis=1)
        for j, marker in enumerate(data.obsm['protein_counts'].columns):
            rho = spearmanr(clr[:, j], predicted[:, j]).statistic if np.std(clr[:, j]) > 0 else np.nan
            protein.append(dict(space=name, protein=marker, neighbor_spearman=rho,
                                interpretation='Internal consistency; proteins also train totalVI, not held-out validation'))
    comparison = pd.DataFrame(metrics).rename_axis('metric').reset_index()
    comparison['change'] = comparison.totalVI-comparison.rna_pca
    comparison.to_csv(output/'metric_comparison.csv', index=False)
    pd.concat(per_type).to_csv(output/'celltype_metrics.csv', index=False)
    pd.DataFrame(depth).to_csv(output/'depth_associations.csv', index=False)
    pd.DataFrame(protein).to_csv(output/'protein_neighbor_agreement.csv', index=False)
    pd.DataFrame(clr, index=obs.index, columns=data.obsm['protein_counts'].columns).groupby(
        obs.cell_type_harmonized, observed=True).mean().to_csv(output/'observed_protein_means_by_celltype.csv')
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    # Same evaluation cells in both visualizations; no full-cohort duplicate graph.
    subset = data[ids].copy()
    for name, key in [('rna_pca', 'X_pca'), ('totalVI', 'X_totalVI')]:
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


def run_totalvi(source, cohort, output, seed=42, n_hvgs=3000, **training):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output folder; previous work is preserved')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'status.json', {'state':'preparing'})
    data = prepare_cite(source, cohort, output, n_hvgs=n_hvgs, seed=seed)
    data.write_h5ad(output/'model_input.h5ad', compression='gzip')
    write_json(output/'manifest.json', dict(seed=seed, n_hvgs=data.n_vars, n_cells=data.n_obs,
        training=training, versions={p:importlib.metadata.version(p) for p in ['scvi-tools','scanpy','anndata','torch']},
        source_sha256=sha256(source), cohort_manifest_sha256=sha256(Path(cohort)/'manifest.json'),
        model='TOTALVI NB RNA + raw ADT; no covariates or labels; no pathway selection'))
    write_json(output/'status.json', {'state':'training'})
    train_model(data, output, seed=seed, **training)
    data.write_h5ad(output/'model_input.h5ad', compression='gzip')
    write_json(output/'status.json', {'state':'trained', 'next':'evaluate_saved; no retraining needed'})
    evaluate_saved(output, seed=seed)
    write_json(output/'status.json', {'state':'complete', 'biological_review':'required'})
