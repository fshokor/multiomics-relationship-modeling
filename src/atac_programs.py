"""Descriptive RNA–ATAC program support within paired Multiome cells only.

Gene activity is used as stored: its upstream transformation is not assumed.
Standardization uses an equal-cell-type reference, separately for each layer.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.rna_concordance import correlations
from src.single_donor_io import load_rna, sha256, write_json
from src.pathway_enrichment import read_gmt


INITIAL_PROGRAMS = [
    ("CD14 monocytes", "Inflammatory Response"),
    ("CD14 monocytes", "TNF-alpha Signaling via NF-kB"),
    ("CD16 monocytes", "Interferon Alpha Response"),
    ("Lymphoid progenitors", "E2F Targets"),
    ("MK/E progenitors", "Myc Targets V1"),
]


def load_inputs(root, requested=None):
    """Check RNA provenance and the requested entries against the saved shortlist."""
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    if status.get('rna_characterization') != 'complete' or status.get('rna_concordance') != 'complete':
        raise ValueError('Complete nb04 and nb05 before nb06')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['artifact_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'Stale RNA artifact: {name}; rerun nb04/nb05')
    followup = pd.read_csv(root / 'rna_concordance/followup_programs.csv')
    if followup.duplicated(['cell_type', 'pathway']).any():
        raise ValueError('Duplicate program keys in followup table')
    indexed = followup.set_index(['cell_type', 'pathway'])
    requested = INITIAL_PROGRAMS if requested is None else list(requested)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Request a nonempty unique program list')
    missing = [key for key in requested if key not in indexed.index]
    if missing:
        raise ValueError(f'Requested programs not in nb05: {missing}')
    selected = indexed.loc[requested].reset_index()
    for key in ['selected_for_followup', 'balanced_direction_supported']:
        if not selected[key].astype(str).str.lower().eq('true').all():
            raise ValueError(f'Every requested program must have {key}=True in nb05')
    for dataset in ['cite', 'multiome']:
        source = pd.read_csv(root / f'gsea/{dataset}_enrichment.csv').set_index(['cell_type', 'pathway'])
        reference = source.loc[requested]
        for field in ['NES', 'q_value']:
            if not np.allclose(reference[field].to_numpy(), selected[f'{field}_{dataset}'], equal_nan=False):
                raise ValueError('Shortlist does not match current nb04 enrichment; rerun nb05')
        if not ((reference.NES > 0) & (reference.q_value <= .05)).all():
            raise ValueError('Initial ATAC programs require positive supported RNA in both captures')
    x, obs, genes = load_rna(root / 'rna_multiome')
    if not obs.index.is_unique or not obs.DonorID.astype(str).eq(str(manifest['donor'])).all():
        raise ValueError('Cached Multiome cells are not a unique single-donor reference')
    if x.shape != (len(obs), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError('RNA cache dimensions or names are invalid')
    universe = pd.read_csv(root / 'rna_concordance/shared_gene_universe.csv')
    universe = set(universe.loc[universe.included_in_comparison, 'gene'])
    return manifest, selected, x, obs, np.asarray(genes), universe


def read_gene_activity(path, cached_obs, wanted, chunk_size=256):
    """Read only requested feature columns; explicitly reorder by RNA cell ID.

    Sparse and dense H5AD arrays are supported. Full-row totals are collected in
    chunks for diagnostics/fallback depth, not treated as raw fragment counts.
    """
    import h5py
    from anndata.io import read_elem, sparse_dataset
    with h5py.File(path, 'r') as handle:
        for key in ['obsm/ATAC_gene_activity', 'uns/ATAC_gene_activity_var_names']:
            if key not in handle:
                raise ValueError(f'Missing {key}')
        source_obs = read_elem(handle['obs'])
        names = pd.Index(np.asarray(read_elem(handle['uns/ATAC_gene_activity_var_names'])).astype(str))
        if not names.is_unique or not source_obs.index.is_unique or not cached_obs.index.is_unique:
            raise ValueError('Unique source cells, cached cells and ATAC gene symbols are required')
        rows = source_obs.index.get_indexer(cached_obs.index)
        if (rows < 0).any():
            raise ValueError('Cached RNA cells missing from source Multiome; cannot pair layers')
        for column in ['DonorID', 'Site', 'cell_type']:
            if not np.array_equal(source_obs.iloc[rows][column].astype(str).to_numpy(), cached_obs[column].astype(str).to_numpy()):
                raise ValueError(f'RNA/source metadata mismatch: {column}')
        node = handle['obsm/ATAC_gene_activity']
        backed = sparse_dataset(node) if isinstance(node, h5py.Group) else node
        if backed.shape != (len(source_obs), len(names)):
            raise ValueError('ATAC matrix shape does not match cell/gene metadata')
        columns = np.flatnonzero(names.isin(wanted))
        if not len(columns):
            raise ValueError('No exact gene-symbol overlap with ATAC gene activity')
        order = np.argsort(rows)
        blocks, totals, zeros = [], [], []
        for start in range(0, len(rows), chunk_size):
            block = sparse.csr_matrix(backed[rows[order[start:start + chunk_size]], :])
            block.sum_duplicates()
            block.eliminate_zeros()
            if not np.isfinite(block.data).all() or (block.data < 0).any():
                raise ValueError('Expected finite nonnegative gene activity; inspect its upstream representation')
            totals.append(np.asarray(block.sum(axis=1)).ravel())
            zeros.append(1 - block.getnnz(axis=1) / block.shape[1])
            blocks.append(block[:, columns])
        inverse = np.argsort(order)
        values = sparse.vstack(blocks, format='csr')[inverse]
        total = np.concatenate(totals)[inverse]
        zero = np.concatenate(zeros)[inverse]
    qc = pd.DataFrame({'atac_activity_total': total, 'atac_zero_fraction': zero,
                       'atac_nonzero_library': total > 0}, index=cached_obs.index)
    return values, names[columns].to_numpy(), qc


def balanced_standardize(matrix, labels, eps=1e-8):
    """Feature z-score with equal weight per retained cell type, all cells within type."""
    x = np.asarray(matrix, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if len(x) != len(labels) or not len(x) or not np.isfinite(x).all():
        raise ValueError('Invalid standardization inputs')
    groups = sorted(set(labels))
    mean = np.mean([x[labels == ct].mean(axis=0) for ct in groups], axis=0)
    second = np.mean([(x[labels == ct] ** 2).mean(axis=0) for ct in groups], axis=0)
    sd = np.sqrt(np.maximum(0, second - mean ** 2))
    valid = sd > eps
    z = np.full_like(x, np.nan)
    z[:, valid] = (x[:, valid] - mean[valid]) / sd[valid]
    return z, mean, sd, valid


def relative_state(rna, atac, threshold=.5):
    if not np.isfinite(rna) or not np.isfinite(atac):
        return 'insufficient evidence'
    if rna >= threshold and atac >= threshold:
        return 'relatively high RNA and ATAC'
    if rna <= -threshold and atac >= threshold:
        return 'relatively high ATAC / low RNA; potentially permissive'
    if rna >= threshold and atac <= -threshold:
        return 'relatively high RNA / low ATAC; candidate decoupling'
    if rna <= -threshold and atac <= -threshold:
        return 'relatively low RNA and ATAC'
    return 'intermediate / mixed'


def residual_correlations(rna, atac, obs):
    """OLS residual association after log-depth and site adjustment; descriptive only."""
    columns = [np.ones(len(obs))]
    for key in ['rna_library_counts', 'atac_depth_covariate']:
        values = np.asarray(obs[key], float)
        if not np.isfinite(values).all() or (values < 0).any():
            return np.nan, np.nan
        values = np.log1p(values)
        if np.std(values) > 1e-10:
            columns.append((values - values.mean()) / values.std())
    sites = pd.get_dummies(obs.Site.astype(str), drop_first=True, dtype=float)
    columns.extend(sites.to_numpy().T)
    design = np.column_stack(columns)
    if len(obs) <= np.linalg.matrix_rank(design) + 3:
        return np.nan, np.nan
    r = rna - design @ np.linalg.lstsq(design, rna, rcond=None)[0]
    a = atac - design @ np.linalg.lstsq(design, atac, rcond=None)[0]
    if np.std(r) < 1e-10 or np.std(a) < 1e-10:
        return np.nan, np.nan
    return correlations(r, a)


def association(rna, atac, obs, minimum=50):
    if len(obs) < minimum:
        return dict(n_cells=len(obs), tested=False, pearson=np.nan, spearman=np.nan,
                    adjusted_pearson=np.nan, residual_spearman=np.nan)
    p, s = correlations(rna, atac)
    ap, rs = residual_correlations(rna, atac, obs)
    return dict(n_cells=len(obs), tested=True, pearson=p, spearman=s,
                adjusted_pearson=ap, residual_spearman=rs)


def analyze_programs(rna, atac, genes, obs, selected, gene_sets, universe,
                     min_genes=10, min_coverage=.2, min_cells=50, threshold=.5):
    """Same genes/cells in both layers; all retained types define the reference.

    Coverage denominator is the original GMT set. Module score = mean per-gene z;
    it is not itself unit variance and is not an absolute activity measurement.
    """
    if rna.shape != atac.shape or rna.shape != (len(obs), len(genes)):
        raise ValueError('RNA/ATAC matrix dimensions must match paired cells and genes')
    if min_genes < 1 or not 0 <= min_coverage <= 1 or min_cells < 3 or threshold <= 0:
        raise ValueError('Invalid score thresholds')
    rna, atac = np.asarray(rna, float), np.asarray(atac, float)
    labels = obs.cell_type_harmonized.to_numpy()
    rz, rm, rs, rv = balanced_standardize(rna, labels)
    az, am, ass, av = balanced_standardize(atac, labels)
    lz, _, _, _ = balanced_standardize(np.log1p(atac), labels)
    lookup = {g: i for i, g in enumerate(genes)}
    usable = set(np.asarray(genes)[rv & av]) & set(universe)
    feature_stats = pd.DataFrame(dict(gene=genes, rna_reference_mean=rm, rna_reference_sd=rs,
                                     atac_reference_mean=am, atac_reference_sd=ass,
                                     usable_in_both=rv & av))
    coverage, cell_scores, summaries, associations, gene_rows, states, memberships = [], [], [], [], [], [], []
    for number, row in enumerate(selected.itertuples(), start=1):
        key = dict(program_id=f'P{number:02d}', cell_type=row.cell_type, pathway=row.pathway)
        members = set(gene_sets[row.pathway])
        available = sorted(members & set(lookup) & set(universe))
        used = sorted(members & usable)
        edge = set(str(row.shared_leading_edge).split(';')) - {'', 'nan'}
        edge_used = sorted(edge & set(used))
        fraction = len(used) / len(members)
        eligible = len(used) >= min_genes and fraction >= min_coverage
        coverage.append(dict(**key, n_original_genes=len(members), n_shared_rna_genes=len(members & set(universe)),
                             n_measured_in_both=len(available), n_variable_in_both=len(used), coverage=fraction,
                             n_shared_leading_edge=len(edge), n_covered_leading_edge=len(edge_used),
                             analyzed=eligible, reason='adequate coverage' if eligible else 'insufficient joint coverage',
                             used_genes=';'.join(used), missing_atac_or_rna=';'.join(sorted(members - set(available)))))
        for g in sorted(members):
            memberships.append(dict(**key, gene=g, in_shared_rna=g in universe, measured_in_both=g in lookup,
                                    used_in_module=g in used, shared_leading_edge=g in edge))
        if not eligible:
            continue
        cols = [lookup[g] for g in used]
        rscore, ascore = rz[:, cols].mean(axis=1), az[:, cols].mean(axis=1)
        logscore = lz[:, cols].mean(axis=1)
        ecol = [lookup[g] for g in edge_used]
        er = rz[:, ecol].mean(axis=1) if len(ecol) >= 3 else np.full(len(obs), np.nan)
        ea = az[:, ecol].mean(axis=1) if len(ecol) >= 3 else np.full(len(obs), np.nan)
        frame = obs[['DonorID', 'Site', 'cell_type_harmonized', 'rna_library_counts', 'atac_depth_covariate']].copy()
        frame['cell_id'] = obs.index
        frame['program_id'], frame['pathway'], frame['target_cell_type'] = key['program_id'], row.pathway, row.cell_type
        frame['rna_score'], frame['atac_score'], frame['atac_log1p_score'] = rscore, ascore, logscore
        frame['rna_edge_score'], frame['atac_edge_score'] = er, ea
        cell_scores.append(frame.reset_index(drop=True))
        scopes = [('pooled_reference', 'ALL', np.ones(len(obs), bool))]
        for ct in sorted(set(labels)):
            mask = labels == ct
            scopes.append(('cell_type', ct, mask))
            rmean, amean = float(rscore[mask].mean()), float(ascore[mask].mean())
            summaries.append(dict(**key, evaluated_cell_type=ct, n_cells=int(mask.sum()),
                                  rna_mean=rmean, atac_mean=amean, rna_median=float(np.median(rscore[mask])),
                                  atac_median=float(np.median(ascore[mask])),
                                  state=relative_state(rmean, amean, threshold) if mask.sum() >= min_cells else 'insufficient cells'))
            if ct == row.cell_type:
                for t in sorted({.25, .5, .75, threshold}):
                    states.append(dict(**key, threshold=t, n_cells=int(mask.sum()),
                                       state=relative_state(rmean, amean, t) if mask.sum() >= min_cells else 'insufficient cells'))
        target = labels == row.cell_type
        for site in sorted(obs.Site.astype(str).unique()):
            scopes.append(('target_by_site', site, target & (obs.Site.to_numpy() == site)))
        for scope, group, mask in scopes:
            record = dict(**key, scope=scope, group=group,
                          **association(rscore[mask], ascore[mask], obs.iloc[np.flatnonzero(mask)], min_cells))
            record['rna_mean'] = float(rscore[mask].mean()) if mask.any() else np.nan
            record['atac_mean'] = float(ascore[mask].mean()) if mask.any() else np.nan
            record['state'] = relative_state(record['rna_mean'], record['atac_mean'], threshold) if mask.sum() >= min_cells else 'insufficient cells'
            record['log1p_atac_spearman'] = correlations(rscore[mask], logscore[mask])[1] if mask.sum() >= min_cells else np.nan
            record['leading_edge_spearman'] = correlations(er[mask], ea[mask])[1] if mask.sum() >= min_cells else np.nan
            record['rna_score_vs_rna_depth'] = correlations(rscore[mask], obs.rna_library_counts.to_numpy()[mask])[1]
            record['atac_score_vs_atac_depth'] = correlations(ascore[mask], obs.atac_depth_covariate.to_numpy()[mask])[1]
            associations.append(record)
        for gene in edge_used:
            j = lookup[gene]
            record = dict(**key, gene=gene, **association(rna[target, j], atac[target, j], obs.loc[target], min_cells),
                          rna_mean=float(rna[target, j].mean()), atac_mean=float(atac[target, j].mean()),
                          rna_detection=float((rna[target, j] > 0).mean()), atac_nonzero_fraction=float((atac[target, j] > 0).mean()))
            gene_rows.append(record)
    # Fixed columns make absence of usable coverage a valid, readable outcome.
    return {
        'coverage': pd.DataFrame(coverage), 'gene_membership': pd.DataFrame(memberships),
        'feature_standardization': feature_stats,
        'cell_scores': pd.concat(cell_scores, ignore_index=True) if cell_scores else pd.DataFrame(columns=['cell_id', 'program_id', 'rna_score', 'atac_score']),
        'population_summary': pd.DataFrame(summaries, columns=list(summaries[0]) if summaries else ['program_id', 'cell_type', 'pathway', 'evaluated_cell_type', 'n_cells', 'rna_mean', 'atac_mean', 'state']),
        'associations': pd.DataFrame(associations, columns=list(associations[0]) if associations else ['program_id', 'cell_type', 'pathway', 'scope', 'group', 'n_cells', 'pearson', 'spearman', 'adjusted_pearson']),
        'leading_edge_genes': pd.DataFrame(gene_rows, columns=list(gene_rows[0]) if gene_rows else ['program_id', 'cell_type', 'pathway', 'gene', 'n_cells', 'pearson', 'spearman']),
        'state_threshold_sensitivity': pd.DataFrame(states, columns=list(states[0]) if states else ['program_id', 'cell_type', 'pathway', 'threshold', 'n_cells', 'state']),
    }


def run_atac(root, multiome_path, requested=None, min_genes=10, min_coverage=.2, min_cells=50, threshold=.5, seed=42):
    root, multiome_path = Path(root), Path(multiome_path)
    output = root / 'atac'
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    write_json(output / 'status.json', {'state': 'in_progress'})
    manifest, selected, x, obs, rna_genes, universe = load_inputs(root, requested)
    if multiome_path.stat().st_size != manifest['inputs']['multiome']['file_bytes']:
        raise ValueError('Source Multiome size differs from nb04; verify source and rerun RNA analysis')
    sets = read_gmt(root / 'gsea/used_collection.gmt')
    wanted = set().union(*(set(sets[p]) for p in selected.pathway)) & universe & set(rna_genes)
    a, genes, qc = read_gene_activity(multiome_path, obs, wanted)
    obs = obs.join(qc, validate='one_to_one')
    obs['included_atac'] = obs.atac_nonzero_library
    obs[['DonorID', 'Site', 'cell_type_harmonized', 'atac_activity_total', 'atac_zero_fraction', 'included_atac']].to_csv(output / 'cell_qc.csv', index_label='cell_id')
    before = obs.groupby('cell_type_harmonized').size()
    valid = obs.included_atac.to_numpy()
    if not valid.any():
        raise ValueError('No cells with nonzero ATAC activity')
    obs, a, x = obs.loc[valid].copy(), a[valid], x[valid]
    if obs.cell_type_harmonized.nunique() < 2:
        raise ValueError('At least two retained populations are needed for the reference')
    count_audit = pd.DataFrame({'n_rna_reference': before, 'n_atac_usable': obs.groupby('cell_type_harmonized').size()}).fillna(0)
    count_audit.to_csv(output / 'cell_counts.csv', index_label='cell_type')
    if any((obs.cell_type_harmonized == ct).sum() == 0 for ct in selected.cell_type):
        raise ValueError('A target cell type has no usable ATAC cells; inspect cell_counts.csv')
    depth_source = 'atac_activity_total (processed-score total, not raw depth)'
    obs['atac_depth_covariate'] = obs.atac_activity_total
    for key in ['ATAC_nCount_peaks', 'ATAC_atac_fragments']:
        if key in obs:
            depth = pd.to_numeric(obs[key], errors='coerce')
            if np.isfinite(depth).all() and (depth >= 0).all() and depth.max() > depth.min():
                obs['atac_depth_covariate'] = depth
                depth_source = key
                break
    indices = pd.Index(rna_genes).get_indexer(genes)
    rna, atac = x[:, indices].toarray(), a.toarray()
    del x, a
    tables = analyze_programs(rna, atac, genes, obs, selected, sets, universe,
                              min_genes, min_coverage, min_cells, threshold)
    selected.to_csv(output / 'selected_rna_programs.csv', index=False)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    from src.atac_plots import make_figures
    figures = make_figures(tables, rna, atac, genes, obs, output / 'figures', seed)
    targets = tables['population_summary']
    targets = targets[targets.cell_type == targets.evaluated_cell_type]
    targets.to_csv(output / 'target_program_summary.csv', index=False)
    sources = ['manifest.json', 'status.json', 'rna_concordance/followup_programs.csv',
               'rna_concordance/balanced_pathway_sensitivity.csv', 'gsea/used_collection.gmt',
               'rna_multiome/rna_log1p_cp10k.npz', 'rna_multiome/cells.csv', 'rna_multiome/genes.csv']
    provenance = dict(donor=manifest['donor'], source_path=str(multiome_path), source_bytes=multiome_path.stat().st_size,
                      source_mtime_ns=multiome_path.stat().st_mtime_ns, upstream_sha256={p: sha256(root / p) for p in sources},
                      source_representation='ATAC gene activity as stored; no assumed count or log scale',
                      score='Mean gene z-score; equal weight per retained cell type; identical genes in both layers',
                      depth_covariate=depth_source, min_genes=min_genes, min_coverage=min_coverage,
                      min_cells=min_cells, state_threshold=threshold, seed=seed, figures=figures,
                      n_programs_requested=len(selected), n_programs_analyzed=int(tables['coverage'].analyzed.sum()),
                      n_cells=len(obs), limitations=['single donor', 'relative activity, not absolute accessibility',
                      'no causal or temporal priming inference', 'RNA site robustness remains provisional',
                      'one balanced RNA draw checked direction, not replicated significance',
                      'no CITE/Multiome cell matching', 'cell associations are descriptive, no individual-level inference'])
    write_json(output / 'manifest.json', provenance)
    write_json(output / 'status.json', {'state': 'complete', 'biological_review': 'required',
                                      'n_programs_analyzed': provenance['n_programs_analyzed']})
    return tables, targets, provenance
