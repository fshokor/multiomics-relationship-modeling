"""nb14: fixed-cell diagnostics of saved embeddings; no model training or imputation.

Absent modalities are NaN in all technical tests. Label-informed residualization is
a diagnostic perturbation, never an integration estimator or held-out validation.
"""
from pathlib import Path
import json
import importlib.metadata
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from src.rna_integration_metrics import (
    baseline_metrics, group_alignment, knn_indices, gradient_agreement)
from src.expression_programs import normalize_counts
from src.single_donor_io import sha256, write_json

GROUPS = ['DonorID', 'cell_type_harmonized']
MODULES = ['Inflammatory Response', 'TNF-alpha Signaling via NF-kB']
SUBDIRS = ['baseline', 'latent_associations', 'depth_coordinate', 'rna_panel',
           'celltype_failures', 'donor_celltype', 'module_preservation', 'figures']


def correlation(x, y, method='spearman'):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float((spearmanr if method == 'spearman' else pearsonr)(x[ok], y[ok]).statistic)


def eta_squared(x, labels):
    frame = pd.DataFrame({'x': x, 'label': labels}).dropna()
    total = ((frame.x-frame.x.mean())**2).sum()
    between = sum(len(g)*(g.x.mean()-frame.x.mean())**2
                  for _, g in frame.groupby('label', observed=True))
    return float(between/total) if total > 0 else np.nan


def load_joint_latent_outputs(root, project=None):
    """Read only saved coordinates, selected counts and metadata, never weights."""
    import h5py
    from anndata.io import read_elem, sparse_dataset
    root = Path(root)
    with h5py.File(root/'model_input.h5ad', 'r') as f:
        obs_all = read_elem(f['obs'])
        if 'X_joint' not in f['obsm']:
            raise ValueError('Saved X_joint missing. Export saved coordinates; do not retrain.')
        z_all = read_elem(f['obsm/X_joint'])
        var, uns = read_elem(f['var']), read_elem(f['uns'])
        n = int(uns['n_genes'])
        # Load RNA only, retaining all training cells for the PCA fit used by nb13.
        ds = sparse_dataset(f['X'])
        rna = sparse.vstack([ds[i:i+2048, :][:, :n]
                             for i in range(0, len(obs_all), 2048)], format='csr')
        proteins = read_elem(f['obsm/protein_counts'])
    ev = pd.read_csv(root/'evaluation_cells.csv', index_col=0,
                     dtype={'DonorID': str, 'Site': str})
    if not ev.index.is_unique or not obs_all.index.is_unique:
        raise ValueError('Duplicate saved cell IDs')
    ids = obs_all.index.get_indexer(ev.index)
    if (ids < 0).any():
        raise ValueError('Evaluation cells absent from saved joint data')
    obs = obs_all.iloc[ids].copy()
    for field in ['assay', 'DonorID', 'Site', 'cell_type_harmonized']:
        if not np.array_equal(obs[field].astype(str), ev[field].astype(str)):
            raise ValueError('Saved metadata disagree: '+field)
        obs[field] = obs[field].astype(str)
    obs['rna_detected_genes_994'] = np.asarray((rna[ids] > 0).sum(axis=1)).ravel()
    obs.loc[obs.assay.ne('CITE'), 'protein_counts'] = np.nan
    obs.loc[obs.assay.ne('Multiome'), 'atac_counts'] = np.nan
    obs['cell_type_original'] = pd.NA
    availability = [dict(field='shared_rna_counts', status='observed; selected RNA panel only'),
                    dict(field='atac_counts', status='observed Multiome; 20,000 selected peaks, not fragment totals'),
                    dict(field='protein_counts', status='observed CITE; non-control ADT panel')]
    if project is not None:
        project = Path(project)
        source = project/'data/benchmark/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad'
        if source.exists():
            with h5py.File(source, 'r') as f:
                raw_obs = read_elem(f['obs'])
            raw_obs.index = 'CITE::'+raw_obs.index.astype(str)
            common = obs.index.intersection(raw_obs.index)
            obs.loc[common, 'cell_type_original'] = raw_obs.loc[common, 'cell_type'].astype(str)
        # Frozen nb09 per-cell QC contains full CITE RNA counts, unlike nb13.
        qc = list((project/'results/cross_donor/nb09_run01').glob('donor_*/protein_cell_qc.csv'))
        if qc:
            q = pd.concat([pd.read_csv(p, index_col=0) for p in qc])
            q.index = 'CITE::'+q.index.astype(str)
            if not q.index.is_unique:
                raise ValueError('Duplicate nb09 QC IDs')
            obs['full_rna_counts'] = q.rna_library_counts.reindex(obs.index)
    for field in ['cell_type_original', 'full_rna_counts']:
        availability.append(dict(field=field, status=f'{obs[field].notna().sum()}/{len(obs)} cells available'
                                 if field in obs else 'unavailable'))
    availability.extend([dict(field='ATAC fragment/TSS/FRiP QC', status='not present in saved nb13 input'),
                         dict(field='full RNA detected genes', status='not present; selected-panel detection reported separately')])
    z = np.asarray(z_all[ids], float)
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite latent values')
    current_umap = pd.read_csv(root/'joint_umap.csv', index_col=0).reindex(obs.index)
    return dict(z=z, obs=obs, ids=ids, rna_all=rna, genes=var.index[:n],
                proteins=sparse.csr_matrix(proteins)[ids], protein_names=uns['protein_names'],
                umap=current_umap, availability=pd.DataFrame(availability),
                n_all=len(obs_all), obs_all=obs_all,
                dimensions=dict(cells=len(obs_all), evaluation_cells=len(obs), latent=z.shape[1],
                                RNA=n, ATAC=int(uns['n_regions']), protein=len(uns['protein_names'])))


def technical_fields(obs):
    return [c for c in ['protein_counts', 'shared_rna_counts', 'full_rna_counts',
                        'rna_detected_genes_994', 'atac_counts'] if c in obs]


def compute_latent_technical_associations(z, obs):
    rows = []
    for j in range(z.shape[1]):
        for field in technical_fields(obs):
            for assay in ['CITE', 'Multiome']:
                mask = obs.assay.eq(assay) & obs[field].notna()
                if mask.sum() < 3:
                    continue
                x, y = z[mask, j], obs.loc[mask, field].to_numpy(float)
                rows.append(dict(dimension=j, variable=field, assay=assay, n=int(mask.sum()),
                                 spearman=correlation(x, y), pearson_log1p=correlation(x, np.log1p(y), 'pearson'),
                                 eta_squared=np.nan))
        for field in ['assay', 'DonorID', 'Site', 'cell_type_harmonized']:
            rows.append(dict(dimension=j, variable=field, assay='all', n=len(obs),
                             spearman=np.nan, pearson_log1p=np.nan,
                             eta_squared=eta_squared(z[:, j], obs[field].to_numpy())))
    return pd.DataFrame(rows)


def compute_stratified_correlations(z, obs, dimensions, min_cells=50):
    rows = []
    # Assay must remain separate for RNA, too: never pool technical scales.
    for key, ids in obs.groupby(['assay']+GROUPS, observed=True).indices.items():
        for field in technical_fields(obs):
            good = ids[obs.iloc[ids][field].notna().to_numpy()]
            if len(good) < min_cells:
                continue
            for j in dimensions:
                rows.append(dict(assay=key[0], DonorID=key[1], cell_type_harmonized=key[2],
                                 variable=field, dimension=j, n=len(good),
                                 spearman=correlation(z[good, j], obs.iloc[good][field])))
    return pd.DataFrame(rows)


def summarize_correlations(table):
    rows = []
    for key, part in table.groupby(['variable', 'assay', 'dimension'], observed=True):
        x = part.spearman.dropna()
        rows.append(dict(zip(['variable', 'assay', 'dimension'], key), n_evaluable=len(x),
                         median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                         median_absolute=x.abs().median(), positive=(x > 0).mean(), negative=(x < 0).mean(),
                         **{f'abs_gt_{t}': (x.abs() > t).mean() for t in [.3, .5, .7]}))
    return pd.DataFrame(rows)


def drop_latent_dimensions(z, dimensions):
    if not dimensions or min(dimensions) < 0 or max(dimensions) >= z.shape[1] or len(set(dimensions)) >= z.shape[1]:
        raise ValueError('Invalid coordinate removal')
    return np.delete(z, sorted(set(dimensions)), axis=1)


def residualize_latent_dimension(z, obs, dimension, min_cells=50):
    """Subtract only the centered depth term within CITE donor/type strata.

    Preserve each stratum intercept and leave unobserved Multiome values unchanged.
    This avoids pretending Multiome has zero protein depth or erasing group means.
    """
    out = np.asarray(z, float).copy()
    rows = []
    for key, ids in obs.groupby(GROUPS, observed=True).indices.items():
        ids = ids[(obs.iloc[ids].assay.eq('CITE') & obs.iloc[ids].protein_counts.notna()).to_numpy()]
        row = dict(zip(GROUPS, key), n=len(ids), slope=np.nan, status='insufficient cells/variation')
        if len(ids) >= min_cells:
            x = np.log1p(obs.iloc[ids].protein_counts.to_numpy(float))
            xc = x-x.mean()
            if np.dot(xc, xc) > 1e-12:
                slope = np.dot(xc, z[ids, dimension]-z[ids, dimension].mean())/np.dot(xc, xc)
                out[ids, dimension] -= slope*xc
                row.update(slope=slope, status='centered depth term removed; intercept retained')
        rows.append(row)
    return out, pd.DataFrame(rows)


def compute_cross_assay_mixing(obs, neighbors):
    assay = obs.assay.to_numpy()
    return (assay[neighbors] != assay[:, None]).mean(axis=1)


def recompute_neighbor_metrics(z, obs):
    cells, neighbors = baseline_metrics(z, obs, seed=42)
    types = group_alignment(z, obs, ['cell_type_harmonized'])
    donors = group_alignment(z, obs, GROUPS)
    summary = dict(cell_type_purity=cells.cell_type_purity.mean(),
                   cross_capture_mixing=cells.opposite_assay_fraction.mean(),
                   problematic_cell_types=int(types.alignment_status.eq('concern').sum()),
                   evaluable_cell_types=int(types.sufficient_cells.sum()),
                   problematic_donor_celltypes=int(donors.alignment_status.eq('concern').sum()),
                   evaluable_donor_celltypes=int(donors.sufficient_cells.sum()))
    return dict(cells=cells, neighbors=neighbors, types=types, donors=donors, summary=summary)


def verify_baseline(result, root, output):
    """Stop on materially different metrics/statuses before any perturbation."""
    reference = pd.read_csv(Path(root)/'metric_comparison.csv').set_index('metric').joint
    rows = []
    for metric, column in [('cell_type_purity', 'cell_type_purity'),
                           ('cross_capture_mixing', 'opposite_assay_fraction')]:
        actual = result['summary'][metric]
        rows.append(dict(metric=metric, actual=actual, reference=reference[column],
                         tolerance=.002, passed=abs(actual-reference[column]) <= .002))
    for name, fname, keys in [('types', 'celltype_alignment.csv', ['cell_type_harmonized']),
                              ('donors', 'donor_celltype_alignment.csv', GROUPS)]:
        old = pd.read_csv(Path(root)/fname, dtype={'DonorID': str})
        old = old[old.space.eq('joint')].set_index(keys).sort_index()
        new = result[name].set_index(keys).sort_index()
        if not new.index.equals(old.index):
            raise ValueError('Baseline group identities disagree')
        for col, tol in [('mixing_ratio', .005), ('scaled_centroid_distance', .005)]:
            delta = np.abs(new[col]-old[col])
            same_missing = np.array_equal(new[col].isna(), old[col].isna())
            rows.append(dict(metric=name+' '+col, actual=delta.max(), reference=0,
                             tolerance=tol, passed=bool(same_missing and delta.max() <= tol)))
        rows.append(dict(metric=name+' statuses', actual=int((new.alignment_status != old.alignment_status).sum()),
                         reference=0, tolerance=0, passed=bool(new.alignment_status.equals(old.alignment_status))))
    s = result['summary']
    for metric, expected in [('problematic_cell_types', 11), ('evaluable_cell_types', 13),
                             ('problematic_donor_celltypes', 39), ('evaluable_donor_celltypes', 42)]:
        rows.append(dict(metric=metric, actual=s[metric], reference=expected, tolerance=0,
                         passed=s[metric] == expected))
    table = pd.DataFrame(rows)
    table.to_csv(Path(output)/'baseline/verification.csv', index=False)
    if not table.passed.all():
        raise ValueError('nb13 baseline discrepancy: inspect baseline/verification.csv before proceeding')
    return table


def compute_neighbor_depth_dependence(obs, neighbors):
    depth = np.log1p(obs.protein_counts.to_numpy(float))
    rows = pd.DataFrame(index=obs.index)
    # Only CITE queries and observed CITE neighbors are evaluable, not missing=zero.
    d = depth[neighbors]
    valid_counts = np.isfinite(d).sum(axis=1)
    med = np.full(len(obs), np.nan)
    ok = np.isfinite(depth) & (valid_counts > 0)
    med[ok] = np.nanmedian(d[ok], axis=1)
    rows['log_protein_depth'] = depth
    rows['neighbor_median_log_depth'] = med
    rows['absolute_log_depth_gap'] = abs(depth-med)
    rows['observed_protein_neighbors'] = valid_counts
    for col in ['assay', 'DonorID', 'Site', 'cell_type_harmonized']:
        a = obs[col].to_numpy()
        rows['same_'+col] = (a[neighbors] == a[:, None]).mean(axis=1)
        rows[col] = a
    # Reference median from CITE cells in the same donor/type, excluding query.
    rng = np.random.default_rng(42)
    ref = np.full(len(obs), np.nan)
    for _, ids in obs.groupby(GROUPS, observed=True).indices.items():
        ids = ids[np.isfinite(depth[ids])]
        if len(ids) < 2:
            continue
        for i in ids:
            pool = ids[ids != i]
            n = min(int(valid_counts[i]), len(pool))
            if n:
                ref[i] = np.median(depth[rng.choice(pool, n, replace=False)])
    rows['matched_random_depth_gap'] = abs(depth-ref)
    return rows


def evaluate_assay_predictability(spaces, obs):
    """Leave one donor out; identical folds; scaling fitted on training cells only."""
    y = obs.assay.eq('CITE').to_numpy(int)
    groups = obs.DonorID.to_numpy()
    rows, coefficients = [], []
    folds = list(LeaveOneGroupOut().split(np.zeros(len(obs)), y, groups))
    for name, z in spaces.items():
        for train, test in folds:
            donor = groups[test][0]
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000))
            model.fit(z[train], y[train])
            p = model.predict_proba(z[test])[:, 1]
            rows.append(dict(representation=name, held_out_donor=donor, n_test=len(test),
                             balanced_accuracy=balanced_accuracy_score(y[test], p >= .5),
                             auroc=roc_auc_score(y[test], p)))
            coefficients.extend(dict(representation=name, held_out_donor=donor, column=j,
                                     standardized_coefficient=c)
                                for j, c in enumerate(model[-1].coef_[0]))
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def pca_counts(counts):
    import scanpy as sc
    import anndata as ad
    data = ad.AnnData(normalize_counts(counts).astype(np.float32))
    sc.pp.pca(data, n_comps=min(30, data.n_vars-1, data.n_obs-1), svd_solver='arpack', random_state=42)
    return data.obsm['X_pca']


def compare_rna_feature_panels(bundle, nb10):
    """Same PCA, 30 PCs and panel-wise CP10k/log1p; all saved cells fit each PCA.

    Broad = all common genes saved by nb10 (not a new HVG selection). This tests
    the total effect of panel restriction including its normalization denominator.
    """
    spaces = {'rna_994': pca_counts(bundle['rna_all'])[bundle['ids']]}
    audit = [dict(representation='rna_994', genes=len(bundle['genes']), status='executed')]
    broad = None
    path = Path(nb10)/'shared_rna.h5ad'
    if path.exists():
        import anndata as ad
        broad = ad.read_h5ad(path)
        ids = broad.obs_names.get_indexer(bundle['obs_all'].index)
        if (ids < 0).any() or not broad.obs_names.is_unique:
            raise ValueError('nb10 cannot cover the fixed nb13 cohort')
        broad = broad[ids].copy()
        if 'counts' not in broad.layers or not set(bundle['genes']) <= set(broad.var_names):
            raise ValueError('nb10 must contain raw counts and all restricted genes')
        restricted = broad.layers['counts'][:, broad.var_names.get_indexer(bundle['genes'])]
        if (sparse.csr_matrix(restricted)-bundle['rna_all']).nnz:
            raise ValueError('nb10 and nb13 shared RNA counts disagree')
        spaces['rna_broad'] = pca_counts(broad.layers['counts'])[bundle['ids']]
        audit.append(dict(representation='rna_broad', genes=broad.n_vars, status='executed'))
    else:
        audit.append(dict(representation='rna_broad', genes=np.nan,
                          status='unresolved: restore nb10 run03/shared_rna.h5ad (raw counts, exact cell IDs)'))
    return spaces, pd.DataFrame(audit), broad


def classify_alignment_failure(result, obs):
    """Nonexclusive descriptive flags; local enrichment is not causal attribution."""
    rows = []
    cells = result['cells']
    for _, r in result['types'].iterrows():
        mask = obs.cell_type_harmonized.eq(r.cell_type_harmonized)
        part = obs.loc[mask]
        c = cells.loc[mask]
        flags, evidence = [], []
        if not r.sufficient_cells:
            flags.append('F: insufficient cells')
        else:
            if r.scaled_centroid_distance > 1:
                flags.append('A: cross-capture centroid shift')
            if r.mixing_ratio < .5:
                flags.append('B: capture-specific neighborhoods (subclustering candidate)')
        for field, metric, label in [('DonorID', 'donor_purity', 'C: donor-enriched neighborhoods'),
                                      ('Site', 'site_purity', 'D: site-enriched neighborhoods')]:
            expected = float((part[field].value_counts(normalize=True)**2).sum())
            excess = float(c[metric].mean()-expected)
            evidence.append(f'{field} excess same-label neighbors={excess:.3f}')
            if part[field].nunique() > 1 and excess > .15:
                flags.append(label)
        if c.cell_type_purity.mean() < .8:
            flags.append('E: weak biological separation')
        evidence.append(f'purity={c.cell_type_purity.mean():.3f}; mixing ratio={r.mixing_ratio:.3f}; scaled centroid={r.scaled_centroid_distance:.3f}')
        rows.append(dict(cell_type=r.cell_type_harmonized, main_failure_mode='; '.join(flags) or 'no threshold flag',
                         evidence='; '.join(evidence), severity=r.alignment_status))
    return pd.DataFrame(rows)


def donor_group_diagnostics(z, obs, alignment):
    rows = []
    for key, ids in obs.groupby(GROUPS, observed=True).indices.items():
        row = dict(zip(GROUPS, key))
        for assay in ['CITE', 'Multiome']:
            ix = ids[obs.iloc[ids].assay.eq(assay).to_numpy()]
            part = obs.iloc[ix]
            row[assay+'_sites'] = ';'.join(sorted(part.Site.unique()))
            if len(ix):
                row.update({f'{assay}_centroid_{j}': v for j, v in enumerate(z[ix].mean(axis=0))})
            for field in technical_fields(obs):
                row[assay+'_'+field] = part[field].median()
        rows.append(row)
    return alignment.merge(pd.DataFrame(rows), on=GROUPS, validate='one_to_one')


def evaluate_module_preservation(spaces, obs, scores):
    rows = []
    for name, z in spaces.items():
        # Stratified within-assay gradients plus independent cross-assay directional prediction.
        for key, ids in obs.groupby(['assay']+GROUPS, observed=True).indices.items():
            if key[2] != 'CD14 monocytes' or len(ids) < 50:
                continue
            nn = knn_indices(z[ids])
            for module in scores:
                x = scores.iloc[ids][module].to_numpy()
                pred = x[nn].mean(axis=1)
                rng = np.random.default_rng(42)
                null = [correlation(x, rng.permutation(x)[nn].mean(axis=1)) for _ in range(20)]
                rows.append(dict(representation=name, assay=key[0], DonorID=key[1], module=module,
                                 spearman=correlation(x, pred), shuffled_median=np.nanmedian(null),
                                 shuffled_q95=np.nanquantile(null, .95), n=len(ids)))
    local = pd.DataFrame(rows)
    cross = pd.concat([gradient_agreement(z, obs, scores).assign(representation=name)
                       for name, z in spaces.items()], ignore_index=True)
    return local, cross


def module_diagnostics(bundle, project, broad=None):
    from src.pathway_enrichment import read_gmt
    from src.rna_integration import score_validated_modules
    project = Path(project)
    discovery = project/'results/single_donor'
    protocol = json.loads((project/'results/cross_donor/nb09_run01/protocol.json').read_text())
    gmt = discovery/'gsea/used_collection.gmt'
    if sha256(gmt) != protocol['gmt_sha256']:
        raise ValueError('Validated module GMT fingerprint changed')
    sets = read_gmt(gmt)
    genes = set(bundle['genes'])
    membership, retention = [], []
    enrichment = pd.read_csv(discovery/'gsea/all_enrichment.csv')
    for module in MODULES:
        original = set(sets[module])
        valid = original & set(protocol['universe'])
        selected = enrichment[enrichment.pathway.eq(module) & enrichment.cell_type.eq('CD14 monocytes')]
        leading = set(';'.join(selected.leading_edge.dropna().astype(str)).split(';'))-{''}
        retention.append(dict(module=module, original_genes=len(original), validated_universe_genes=len(valid),
                              retained=len(original & genes), retained_percent=100*len(original & genes)/len(original),
                              retained_leading_edge=';'.join(sorted(leading & genes)),
                              missing_leading_edge=';'.join(sorted(leading-genes))))
        membership.extend(dict(module=module, gene=g, retained=g in genes, validated=g in valid,
                               leading_edge=g in leading) for g in sorted(original))
    # Restricted scores are clearly labelled proxies, not full validated scores.
    import anndata as ad
    restricted = ad.AnnData(normalize_counts(bundle['rna_all']).astype(np.float32),
                            obs=bundle['obs_all'].copy(), var=pd.DataFrame(index=bundle['genes']))
    rscores, _ = score_validated_modules(restricted, discovery, protocol)
    scores = rscores.iloc[bundle['ids']].copy()
    status = 'restricted-panel proxy only; full validated module preservation unresolved'
    correlations = []
    if broad is not None:
        broad.X = normalize_counts(broad.layers['counts']).astype(np.float32)
        full, _ = score_validated_modules(broad, discovery, protocol)
        full = full.iloc[bundle['ids']]
        for key, ids in bundle['obs'].groupby(['assay']+GROUPS, observed=True).indices.items():
            if key[2] != 'CD14 monocytes' or len(ids) < 50:
                continue
            for module in MODULES:
                correlations.append(dict(assay=key[0], DonorID=key[1], module=module, n=len(ids),
                                         score_correlation=correlation(full.iloc[ids][module], scores.iloc[ids][module])))
        scores = full
        status = 'full available RNA module scores; restricted vs full comparisons executed'
    return pd.DataFrame(retention), pd.DataFrame(membership), scores, pd.DataFrame(correlations), status


def protein_preprocessing_diagnostics(proteins, obs, names):
    mask = obs.assay.eq('CITE').to_numpy()
    raw = proteins[mask].toarray().astype(float)
    log = np.log1p(raw)
    clr = log-log.mean(axis=1, keepdims=True)  # feature-centered log1p CLR, explicit convention
    depth = obs.loc[mask, 'protein_counts'].to_numpy()
    rows = []
    for name, x in [('raw_ADT', raw), ('log1p_ADT', log), ('centered_log1p_CLR', clr)]:
        for j, protein in enumerate(names):
            rows.append(dict(transform=name, protein=protein,
                             spearman_depth=correlation(x[:, j], depth)))
    return pd.DataFrame(rows)


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(Path(output)/'figures'/f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def compute_umap(z, obs):
    import anndata as ad
    import scanpy as sc
    data = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())
    data.obsm['diagnostic'] = z
    sc.pp.neighbors(data, use_rep='diagnostic', n_neighbors=30, random_state=42)
    sc.tl.umap(data, random_state=42)
    return data.obsm['X_umap']


def plot_umap(coords, obs, field, output, name):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, label in enumerate(sorted(obs[field].unique())):
        ix = obs[field].eq(label).to_numpy()
        color = {'CITE': '#35789a', 'Multiome': '#bf663b'}.get(label, plt.get_cmap('tab20')(j % 20))
        ax.scatter(coords[ix, 0], coords[ix, 1], s=2, alpha=.5, rasterized=True,
                   label=label, color=color)
    ax.set(xlabel='UMAP 1', ylabel='UMAP 2', title=name.replace('_', ' '))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=4, fontsize=8)
    save_figure(fig, output, name)


def association_figures(associations, strata, output):
    import matplotlib.pyplot as plt
    for field, name in [('protein_counts', '01_latent_protein_depth'),
                         ('shared_rna_counts', '02_latent_RNA_depth'),
                         ('atac_counts', '03_latent_ATAC_depth')]:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for assay, part in associations[associations.variable.eq(field)].groupby('assay'):
            ax.plot(part.dimension, part.spearman, marker='o', markersize=3, label=assay)
        ax.axhline(0, color='grey', lw=.6)
        ax.set(xlabel='Latent coordinate (zero-based)', ylabel='Spearman rho', ylim=(-1.05, 1.05), title=field)
        ax.legend()
        save_figure(fig, output, name)
    data = strata[strata.variable.eq('protein_counts')].dropna(subset=['spearman'])
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = list(data.groupby('dimension'))
    ax.boxplot([p.spearman for _, p in groups], tick_labels=[str(j) for j, _ in groups])
    ax.set(xlabel='Latent coordinate (zero-based)', ylabel='Within donor × cell-type Spearman rho', ylim=(-1.05, 1.05),
           title='CITE only; at least 50 cells per stratum')
    save_figure(fig, output, '04_stratified_protein_depth')


def comparison_figures(results, comparison, depth_tables, donor_table, retention, local, output,
                       module_status=''):
    import matplotlib.pyplot as plt
    for column, name, title in [('cross_capture_mixing', '09_mixing_comparison', 'Cross-capture neighbors'),
                                 ('cell_type_purity', '10_purity_comparison', 'Cell-type purity')]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(comparison.representation, comparison[column], color='#35789a')
        ax.tick_params(axis='x', rotation=25)
        ax.set(ylabel='Fraction', title=title, ylim=(0, 1))
        save_figure(fig, output, name)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results.items():
        p = result['types'].set_index('cell_type_harmonized').sort_index()
        ax.plot(p.mixing_ratio, p.index, marker='o', label=name, alpha=.8)
    ax.axvline(.5, color='grey', ls='--', label='Concern threshold')
    ax.set(xlabel='Composition-adjusted mixing ratio', title='Per-population alignment')
    ax.legend(fontsize=8)
    save_figure(fig, output, '11_celltype_alignment')
    fig, ax = plt.subplots(figsize=(8, 4))
    p = comparison[comparison.representation.str.startswith('rna_')]
    ax.bar(p.representation, p.cross_capture_mixing, color='#35789a')
    ax.set(ylabel='Cross-capture neighbor fraction', title='Matched PCA feature-panel comparison')
    if p.cross_capture_mixing.isna().any():
        ax.text(.98, .95, 'Broader RNA unavailable: comparison unresolved', transform=ax.transAxes,
                ha='right', va='top', fontsize=9)
    save_figure(fig, output, '12_RNA_panel_comparison')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(retention.module, retention.retained_percent, color='#35789a')
    ax.set(xlabel='Original module genes retained (%)', xlim=(0, 100), title='994-gene panel coverage')
    save_figure(fig, output, '13_module_retention')
    for number, module in [(14, MODULES[0]), (15, MODULES[1])]:
        fig, ax = plt.subplots(figsize=(9, 4))
        part = local[local.module.eq(module)]
        names = list(part.representation.unique())
        for i, name in enumerate(names):
            p = part[part.representation.eq(name)]
            for assay, color in [('CITE', '#35789a'), ('Multiome', '#bf663b')]:
                q = p[p.assay.eq(assay)]
                ax.scatter(np.full(len(q), i)+(-.09 if assay == 'CITE' else .09), q.spearman,
                           color=color, label=assay if i == 0 else None)
        ax.set_xticks(range(len(names)), names, rotation=25)
        suffix = '\nRestricted-panel proxy; full biology unresolved' if 'proxy' in module_status else ''
        ax.set(ylabel='CD14 score vs neighbor-mean Spearman rho', ylim=(-1, 1), title=module+suffix)
        ax.legend()
        save_figure(fig, output, f'{number}_CD14_gradient')
    valid = donor_table[donor_table.sufficient_cells]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ['CITE_protein_counts', 'CITE_shared_rna_counts', 'Multiome_atac_counts']):
        for donor, p in valid.groupby('DonorID'):
            ax.scatter(p[col], p.mixing_ratio, s=22, label=donor)
        ax.set(xscale='log', xlabel=col.replace('_', ' '), ylabel='Mixing ratio')
    axes[-1].legend(bbox_to_anchor=(1, 1), title='Donor', fontsize=7)
    save_figure(fig, output, '16_alignment_vs_depth')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, table in depth_tables.items():
        p = table[table.assay.eq('CITE')]
        q = p.groupby('cell_type_harmonized').absolute_log_depth_gap.median().sort_index()
        axes[0].plot(q, q.index, marker='o', label=name)
        finite = p.absolute_log_depth_gap.dropna().sort_values().to_numpy()
        axes[1].plot(finite, np.arange(1, len(finite)+1)/len(finite), label=name)
    axes[0].set(xlabel='Median absolute log-depth gap', title='Observed CITE neighborhoods')
    axes[1].set(xlabel='Absolute log-depth gap', ylabel='Cumulative fraction',
                title='CITE depth-gap distribution\nMultiome protein depth is unobserved')
    axes[1].legend(fontsize=8)
    save_figure(fig, output, '17_neighbor_depth_dependence')


def decision_tables(comparison, associations, strata_summary, predictions, failure, module_status,
                    retention=None):
    """Conservative exploratory interpretation; thresholds declared in notebook."""
    c = comparison.set_index('representation')
    full, drop, resid = c.loc['joint_full'], c.loc['minus_top1'], c.loc['depth_residualized']
    delta = float(drop.cross_capture_mixing-full.cross_capture_mixing)
    purity_delta = float(drop.cell_type_purity-full.cell_type_purity)
    residual_delta = float(resid.cross_capture_mixing-full.cross_capture_mixing)
    pred = predictions.groupby('representation')[['balanced_accuracy', 'auroc']].mean()
    auc_delta = float(pred.loc['joint_full', 'auroc']-pred.loc['minus_top1', 'auroc'])
    nstrong = int((associations[associations.variable.eq('protein_counts')].spearman.abs() >= .5).sum())
    protein = associations[associations.variable.eq('protein_counts')]
    strongest = int(protein.loc[protein.spearman.abs().idxmax(), 'dimension'])
    within = strata_summary[strata_summary.variable.eq('protein_counts') & strata_summary.dimension.eq(strongest)].iloc[0]
    rows = []
    h1 = ('partially supported' if delta >= .02 and purity_delta >= -.02 else
          'unsupported' if abs(delta) < .02 and abs(residual_delta) < .02 else 'unresolved')
    rows.append(dict(hypothesis='H1: single protein-depth coordinate',
                     evidence_for=f'Deletion mixing change {delta:+.4f}; residualization {residual_delta:+.4f}; held-donor AUROC decrease {auc_delta:+.4f}',
                     evidence_against=f'Purity change {purity_delta:+.4f}; association/removal cannot isolate causality; residualization preserves group offsets',
                     conclusion=h1))
    top = c.loc['minus_top_depth'] if 'minus_top_depth' in c.index else drop
    multi_delta = float(top.cross_capture_mixing-full.cross_capture_mixing)
    h2 = 'partially supported' if nstrong >= 2 and multi_delta >= .02 else 'unresolved' if nstrong >= 2 else 'unsupported'
    rows.append(dict(hypothesis='H2: broader protein-depth encoding',
                     evidence_for=f'{nstrong} dimensions have global CITE |rho| >= 0.5; multi-coordinate mixing change {multi_delta:+.4f}',
                     evidence_against='Feature-depth associations may include biological signal; no observed Multiome protein depths', conclusion=h2))
    broad_done = pd.notna(c.loc['rna_broad', 'cross_capture_mixing'])
    if broad_done:
        rna_delta = float(c.loc['rna_broad', 'cross_capture_mixing']-c.loc['rna_994', 'cross_capture_mixing'])
        rna_purity = float(c.loc['rna_broad', 'cell_type_purity']-c.loc['rna_994', 'cell_type_purity'])
        h3 = 'partially supported' if rna_delta >= .02 and rna_purity >= -.02 else 'unsupported'
        rna_evidence = f'Broad minus restricted RNA mixing {rna_delta:+.4f}; purity {rna_purity:+.4f}'
    else:
        h3, rna_evidence = 'unresolved', 'nb10 full RNA counts unavailable locally; no matched broad-panel experiment'
    rows.append(dict(hypothesis='H3: restricted RNA information', evidence_for=rna_evidence,
                     evidence_against='Module gene retention alone does not establish alignment impairment; PCA is not matched joint-model retraining', conclusion=h3))
    rows.append(dict(hypothesis='H4: population-specific failure',
                     evidence_for=f'{failure.main_failure_mode.nunique()} distinct combinations of descriptive flags',
                     evidence_against='Threshold flags do not prove separate mechanisms; shared technical effects may vary by population',
                     conclusion='partially supported' if failure.main_failure_mode.nunique() > 1 else 'unresolved'))
    has_group = failure.main_failure_mode.str.contains('C:|D:', regex=True).any()
    rows.append(dict(hypothesis='H5: donor/site structure',
                     evidence_for=f'{int(failure.main_failure_mode.str.contains("C:|D:", regex=True).sum())} populations have donor/site enrichment flags',
                     evidence_against='Donor and site are confounded; local enrichment is descriptive and not variance causally explained',
                     conclusion='partially supported' if has_group else 'unsupported'))
    if h1 == 'partially supported' or h2 == 'partially supported':
        action = 'Option B: prioritize protein-depth handling diagnostics before any joint-model retraining.'
    elif h3 == 'partially supported':
        action = 'Option C: broaden shared RNA information before testing a new joint model.'
    elif broad_done:
        action = ('Option E, with a targeted diagnostic next step: retain the joint model only as a descriptive biological representation; '
                  'do not rely on it for cross-capture alignment. Neither deleting depth-associated coordinates nor broadening '
                  'the uncorrected RNA PCA panel demonstrates a sufficient alignment remedy. Before nb15, test capture effects '
                  'within donor/site/population strata and assess population-aware integration against the saved full-module gradients. '
                  'These findings do not establish that a broader panel would be ineffective in a retrained joint model.')
    else:
        action = ('Option F: restore nb10 shared_rna.h5ad and complete the controlled RNA-panel and full-module tests before choosing the nb15 model modification. '
                  'Treat the current joint space as unreliable for cross-capture alignment meanwhile (Option E).')
    coverage = '' if retention is None else '\n\n'+'; '.join(
        f'{r.module}: {int(r.retained)}/{int(r.original_genes)} original module genes retained ({r.retained_percent:.1f}%)'
        for r in retention.itertuples())+'. Gene coverage alone cannot establish loss of alignment information.'
    rna_findings = ('\n\nBroader RNA comparison remains unavailable.' if not broad_done else
                    f'\n\nRNA-only PCA: {int(c.loc["rna_broad", "RNA_genes"]):,} genes yield '
                    f'{c.loc["rna_broad", "cell_type_purity"]:.2%} purity and '
                    f'{c.loc["rna_broad", "cross_capture_mixing"]:.2%} mixing, versus '
                    f'{c.loc["rna_994", "cell_type_purity"]:.2%} and '
                    f'{c.loc["rna_994", "cross_capture_mixing"]:.2%} for 994 genes. '
                    f'The broader panel has {int(c.loc["rna_broad", "problematic_cell_types"])}/'
                    f'{int(c.loc["rna_broad", "evaluable_cell_types"])} population concerns and '
                    f'{int(c.loc["rna_broad", "problematic_donor_celltypes"])}/'
                    f'{int(c.loc["rna_broad", "evaluable_donor_celltypes"])} donor/type concerns.')
    report = (
        '# Main findings\n\n'
        f'Fixed nb13 evaluation: purity {full.cell_type_purity:.4%}, cross-capture mixing {full.cross_capture_mixing:.4%}; '
        f'{int(full.problematic_cell_types)}/{int(full.evaluable_cell_types)} cell types and '
        f'{int(full.problematic_donor_celltypes)}/{int(full.evaluable_donor_celltypes)} donor/type groups have concerns.\n\n'
        f'Coordinate {strongest} (zero-based) has median within-stratum |rho|={within.median_absolute:.3f} '
        f'across {int(within.n_evaluable)} CITE donor/type strata; {nstrong} coordinates have global CITE |rho|≥0.5.\n\n'
        f'Deleting the strongest coordinate changes mixing by {100*delta:+.3f} percentage points and purity by {100*purity_delta:+.3f} points. '
        f'Top-coordinate deletion changes mixing by {100*multi_delta:+.3f} points; centered depth residualization by {100*residual_delta:+.3f} points. '
        f'Mean held-donor assay AUROC is {pred.loc["joint_full", "auroc"]:.6f} in the full space and '
        f'{pred.loc["minus_top1", "auroc"]:.6f} after single-coordinate deletion (decrease {auc_delta:.2g}).'+coverage+rna_findings+'\n\n'
        '# Diagnosis\n\n'
        f'H1 is {h1}; H2 is {h2}; H3 is {h3}. Identity purity and cross-capture alignment measure different properties: '
        'a representation can retain population boundaries while separating captures inside those populations. '
        'The supported diagnosis is persistent capture-specific structure in the remaining representation, '
        'with donor/site-associated and population-dependent patterns; a single protein-depth axis does not explain most of the failure. '
        'The perturbations test neighborhood sensitivity, not biological causality. '
        f'Full-module assessment: {module_status}. '
        + ('These descriptive comparisons do not isolate a unique causal mechanism.\n\n' if broad_done else
           'A unique limiting mechanism cannot be selected while key comparisons remain unresolved.\n\n') +
        '# Recommended next step\n\n'+action+' '
        'Require full-module gradient preservation before accepting any apparent mixing gain. '
        'Do not adopt coordinate deletion or label-informed residualization as the final correction. '
        'Longer training is not indicated by these diagnostics. Cells remain unpaired; missing-modality predictions and direct protein–ATAC relationships remain unvalidated.\n')
    return pd.DataFrame(rows), report


class DiagnosticRun:
    """Section-oriented notebook workflow. Each stage saves its own auditable outputs."""

    def __init__(self, project, output=None):
        self.project = Path(project)
        self.root = self.project/'results/joint_integration/nb13_run01'
        self.nb10 = self.project/'results/shared_rna_integration/run03'
        self.output = Path(output) if output else self.project/'results/joint_model_diagnostics'
        for name in SUBDIRS:
            (self.output/name).mkdir(parents=True, exist_ok=True)
        self.results, self.spaces, self.umaps, self.depth = {}, {}, {}, {}
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                             'pdf.fonttype': 42, 'figure.facecolor': 'white'})

    def save(self, table, path, index=False):
        table.to_csv(self.output/path, index=index)
        return table

    def load(self):
        write_json(self.output/'status.json', {'state': 'running', 'training': False})
        self.bundle = load_joint_latent_outputs(self.root, self.project)
        self.obs = self.bundle['obs']
        self.save(self.obs, 'baseline/cell_metadata.csv', index=True)
        self.save(self.bundle['availability'], 'baseline/metadata_availability.csv')
        self.save(self.bundle['umap'], 'baseline/saved_nb13_umap.csv', index=True)
        write_json(self.output/'baseline/dimensions.json', self.bundle['dimensions'])
        self.obs.groupby(['assay', 'DonorID', 'Site'], observed=True).size().rename('n_cells').to_csv(self.output/'baseline/evaluation_counts.csv')
        write_json(self.output/'manifest.json', dict(seed=42, k=30, min_cells_per_assay=50,
                   source_model_sha256=sha256(self.root/'model_input.h5ad'),
                   evaluation_cells_sha256=sha256(self.root/'evaluation_cells.csv'),
                   versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'scanpy', 'anndata']},
                   scope='Fixed nb13 evaluation cohort, descriptive diagnostic perturbations only',
                   practical_thresholds=dict(mixing_change=.02, purity_loss=.02, strong_global_depth_rho=.5),
                   prior_notebooks_modified=False))
        return self.bundle['dimensions']

    def evaluate(self, name, z, subdir='depth_coordinate', umap=True):
        print('Evaluating', name, z.shape, flush=True)
        result = recompute_neighbor_metrics(z, self.obs)
        self.results[name], self.spaces[name] = result, z
        for key in ['cells', 'types', 'donors']:
            self.save(result[key], f'{subdir}/{name}_{key}.csv', index=key == 'cells')
        np.savez_compressed(self.output/subdir/f'{name}_latent.npz', z=z,
                            cell_id=self.obs.index.to_numpy(dtype=str))
        if umap:
            coords = compute_umap(z, self.obs)
            self.umaps[name] = coords
            self.save(pd.DataFrame(coords, index=self.obs.index, columns=['UMAP1', 'UMAP2']),
                      f'{subdir}/{name}_umap.csv', index=True)
        return result['summary']

    def baseline(self):
        self.evaluate('joint_full', self.bundle['z'], 'baseline', umap=False)
        check = verify_baseline(self.results['joint_full'], self.root, self.output)
        self.save(pd.DataFrame([self.results['joint_full']['summary']]), 'baseline/metrics.csv')
        self.umaps['joint_full'] = compute_umap(self.bundle['z'], self.obs)
        self.save(pd.DataFrame(self.umaps['joint_full'], index=self.obs.index, columns=['UMAP1', 'UMAP2']),
                  'baseline/recomputed_umap.csv', index=True)
        plot_umap(self.umaps['joint_full'], self.obs, 'assay', self.output, '05_full_latent_assay')
        plot_umap(self.umaps['joint_full'], self.obs, 'cell_type_harmonized', self.output, '07_full_latent_celltype')
        return check

    def associations(self):
        self.assoc = compute_latent_technical_associations(self.bundle['z'], self.obs)
        self.save(self.assoc, 'latent_associations/all_dimensions.csv')
        rank = self.assoc[self.assoc.variable.eq('protein_counts')].copy()
        rank['absolute_rho'] = rank.spearman.abs()
        rank = rank.sort_values('absolute_rho', ascending=False)
        self.top = int(rank.iloc[0].dimension)
        self.selected = rank.loc[rank.absolute_rho.ge(.5), 'dimension'].astype(int).tolist()
        self.selected = list(dict.fromkeys([self.top]+self.selected))
        wide = self.assoc.pivot(index='dimension', columns=['variable', 'assay'], values='spearman')
        wide.columns = [' '.join(c)+' rho' for c in wide.columns]
        wide = wide.dropna(axis=1, how='all')
        for field in ['assay', 'DonorID', 'Site', 'cell_type_harmonized']:
            wide[field+' eta2'] = self.assoc[self.assoc.variable.eq(field)].set_index('dimension').eta_squared
        wide = wide.loc[rank.dimension]
        self.save(wide, 'latent_associations/ranked_dimension_table.csv', index=True)
        write_json(self.output/'latent_associations/selected_dimensions.json',
                   dict(zero_based_top=self.top, abs_rho_threshold=.5, dimensions=self.selected))
        return wide

    def strata(self):
        self.stratified = compute_stratified_correlations(self.bundle['z'], self.obs, self.selected)
        self.strata_summary = summarize_correlations(self.stratified)
        self.save(self.stratified, 'latent_associations/within_strata.csv')
        self.save(self.strata_summary, 'latent_associations/strata_summary.csv')
        association_figures(self.assoc, self.stratified, self.output)
        return self.strata_summary

    def perturb(self):
        self.evaluate('minus_top1', drop_latent_dimensions(self.bundle['z'], [self.top]))
        if len(self.selected) > 1:
            self.evaluate('minus_top_depth', drop_latent_dimensions(self.bundle['z'], self.selected[:3]))
        plot_umap(self.umaps['minus_top1'], self.obs, 'assay', self.output, '06_minus_top1_assay')
        plot_umap(self.umaps['minus_top1'], self.obs, 'cell_type_harmonized', self.output, '08_minus_top1_celltype')
        return pd.DataFrame({k: v['summary'] for k, v in self.results.items()}).T

    def residualize(self):
        z, coefficients = residualize_latent_dimension(self.bundle['z'], self.obs, self.top)
        self.save(coefficients, 'depth_coordinate/residualization_coefficients.csv')
        checks = compute_stratified_correlations(z, self.obs, [self.top])
        self.save(checks, 'depth_coordinate/residualized_depth_correlations.csv')
        return self.evaluate('depth_residualized', z)

    def neighborhoods(self):
        summary = []
        for name in ['joint_full', 'minus_top1']:
            table = compute_neighbor_depth_dependence(self.obs, self.results[name]['neighbors'])
            self.depth[name] = table
            self.save(table, f'depth_coordinate/{name}_neighbor_depth.csv', index=True)
            for key, p in table.groupby(['assay']+GROUPS, observed=True):
                summary.append(dict(representation=name, assay=key[0], DonorID=key[1], cell_type=key[2],
                                    n=len(p), depth_neighbor_rho=correlation(p.log_protein_depth, p.neighbor_median_log_depth),
                                    median_depth_gap=p.absolute_log_depth_gap.median(),
                                    random_depth_gap=p.matched_random_depth_gap.median(),
                                    measured_fraction=p.neighbor_median_log_depth.notna().mean(),
                                    same_assay=p.same_assay.mean(), same_type=p.same_cell_type_harmonized.mean(),
                                    same_donor=p.same_DonorID.mean(), same_site=p.same_Site.mean()))
        return self.save(pd.DataFrame(summary), 'depth_coordinate/neighbor_depth_summary.csv')

    def predictability(self):
        # The label-informed residualized representation is deliberately excluded.
        spaces = {k: v for k, v in self.spaces.items() if k in ['joint_full', 'minus_top1', 'minus_top_depth']}
        self.predictions, coef = evaluate_assay_predictability(spaces, self.obs)
        self.save(self.predictions, 'depth_coordinate/assay_prediction_folds.csv')
        self.save(coef, 'depth_coordinate/assay_classifier_coefficients.csv')
        return self.predictions.groupby('representation')[['balanced_accuracy', 'auroc']].agg(['mean', 'std'])

    def rna_panels(self):
        spaces, self.rna_audit, self.broad = compare_rna_feature_panels(self.bundle, self.nb10)
        self.save(self.rna_audit, 'rna_panel/availability.csv')
        if self.broad is not None:
            write_json(self.output/'rna_panel/input_provenance.json', dict(
                path=str(self.nb10/'shared_rna.h5ad'), sha256=sha256(self.nb10/'shared_rna.h5ad'),
                n_cells=self.broad.n_obs, n_genes=self.broad.n_vars,
                shared_counts_verified=True, cell_order_verified=True))
        for name, z in spaces.items():
            self.evaluate(name, z, 'rna_panel')
        # Reproduce nb13 PCA baseline as an additional audit of preprocessing.
        reference = pd.read_csv(self.root/'metric_comparison.csv').set_index('metric').rna_pca
        actual = self.results['rna_994']['summary']
        if abs(actual['cell_type_purity']-reference.cell_type_purity) > .005:
            raise ValueError('RNA PCA baseline differs substantially from nb13; inspect before interpretation')
        return self.rna_audit

    def modules(self):
        self.retention, membership, self.scores, score_corr, self.module_status = module_diagnostics(
            self.bundle, self.project, self.broad)
        self.save(self.retention, 'module_preservation/gene_retention.csv')
        self.save(membership, 'module_preservation/gene_membership.csv')
        self.save(self.scores, 'module_preservation/scores.csv', index=True)
        self.save(score_corr, 'module_preservation/full_restricted_correlations.csv')
        write_json(self.output/'module_preservation/status.json', dict(status=self.module_status))
        return self.retention

    def failures(self):
        self.failure = classify_alignment_failure(self.results['joint_full'], self.obs)
        return self.save(self.failure, 'celltype_failures/classification.csv')

    def donor_groups(self):
        self.donor_table = donor_group_diagnostics(self.bundle['z'], self.obs, self.results['joint_full']['donors'])
        self.save(self.donor_table, 'donor_celltype/groups.csv')
        valid = self.donor_table[self.donor_table.sufficient_cells].copy()
        fields = ['n_CITE', 'n_Multiome', 'CITE_protein_counts', 'CITE_shared_rna_counts', 'Multiome_shared_rna_counts', 'Multiome_atac_counts']
        rows = []
        for field in fields:
            for metric in ['mixing_ratio', 'scaled_centroid_distance']:
                x = valid.groupby('DonorID')[[field, metric]].median().dropna()
                rows.append(dict(level='donor median', variable=field, metric=metric, n=len(x),
                                 spearman=correlation(x[field], x[metric])))
                # Group-level correlations are descriptive; repeated groups share donors.
                rows.append(dict(level='donor x cell type (dependent groups)', variable=field, metric=metric,
                                 n=len(valid), spearman=correlation(valid[field], valid[metric])))
        self.save(pd.DataFrame(rows), 'donor_celltype/depth_alignment_correlations.csv')
        self.save(pd.crosstab(self.obs.DonorID, self.obs.Site), 'donor_celltype/donor_site_confounding.csv', index=True)
        self.save(valid.groupby(['CITE_sites', 'Multiome_sites'])[['mixing_ratio', 'scaled_centroid_distance']].agg(['median', 'count']),
                  'donor_celltype/site_alignment_summary.csv', index=True)
        return pd.DataFrame(rows)

    def gradients(self):
        self.local, self.cross = evaluate_module_preservation(self.spaces, self.obs, self.scores)
        self.save(self.local, 'module_preservation/local_gradients.csv')
        self.save(self.cross, 'module_preservation/cross_capture_gradients.csv')
        return self.local.groupby(['representation', 'module']).spearman.agg(['median', 'count'])

    def proteins(self):
        table = protein_preprocessing_diagnostics(self.bundle['proteins'], self.obs, self.bundle['protein_names'])
        self.save(table, 'depth_coordinate/protein_feature_depth.csv')
        note = (
            'Repository nb11 prepare_cite excludes isotype/control proteins and supplies raw integer ADT counts to totalVI. '
            'nb13 reuses that 134-protein raw panel, with Multiome protein rows masked as missing. '
            'No external ADT normalization or isotype-based background correction precedes training. '
            'MultiVI uses a protein foreground/background count mixture, with learned protein background parameters; '
            'the model implementation has RNA and accessibility library encoders, not a separate observed ADT-total offset. '
            'A background mixture alone should not be equated with removal of library-depth effects. '
            'Feature diagnostics compare raw counts, log1p counts and within-cell centered log1p CLR. '
            'CLR is only a diagnostic feature transform; these values are not valid raw-count inputs to the existing likelihood. '
            'No saved totalVI denoised representation is available locally. No model was retrained.')
        (self.output/'depth_coordinate/protein_preprocessing.md').write_text(note, encoding='utf-8')
        return table.assign(absolute_rho=table.spearman_depth.abs()).groupby('transform').absolute_rho.agg(['median', 'max'])

    def finish(self):
        rows = []
        genes = self.rna_audit.set_index('representation').genes.to_dict()
        for name, result in self.results.items():
            a = compute_latent_technical_associations(self.spaces[name], self.obs)
            depth = a[a.variable.eq('protein_counts')].spearman.abs().max()
            self.save(a, f'latent_associations/{name}_associations.csv')
            gradients = self.local[self.local.representation.eq(name)]
            text = self.module_status
            if 'proxy' not in self.module_status:
                med = gradients.groupby('module').spearman.median()
                ref = self.local[self.local.representation.eq('joint_full')].groupby('module').spearman.median()
                text = 'preserved by exploratory thresholds' if len(med) == 2 and med.ge(.2).all() and (med-ref).ge(-.05).all() else 'mixed/weak; inspect donor gradients'
            rows.append(dict(representation=name, RNA_genes=genes.get(name, len(self.bundle['genes'])),
                             protein_depth_dependence=depth, CD14_biology_preserved=text,
                             **result['summary']))
        if 'rna_broad' not in self.results:
            rows.append(dict(representation='rna_broad', RNA_genes=np.nan, protein_depth_dependence=np.nan,
                             CD14_biology_preserved='unresolved: missing broad RNA counts'))
        self.comparison = pd.DataFrame(rows)
        self.save(self.comparison, 'final_comparison.csv')
        self.decisions, report = decision_tables(self.comparison, self.assoc, self.strata_summary,
                                                 self.predictions, self.failure, self.module_status, self.retention)
        self.save(self.decisions, 'hypothesis_decision_table.csv')
        (self.output/'final_recommendation.md').write_text(report, encoding='utf-8')
        comparison_figures(self.results, self.comparison, self.depth, self.donor_table,
                           self.retention, self.local, self.output, self.module_status)
        write_json(self.output/'status.json', dict(state='partial' if self.broad is None else 'complete',
                   completed='All locally available diagnostics executed; no training',
                   unresolved=[] if self.broad is not None else ['matched broader RNA comparison', 'full-module score and biology preservation'],
                   needed_input=str(self.nb10/'shared_rna.h5ad') if self.broad is None else None))
        return self.comparison, self.decisions, report
