"""Fixed-target cross-donor validation and separate raw RNA pseudobulk exports."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from src.single_donor_io import inspect_h5ad, read_rna_counts, sha256, write_json
from src.celltype_harmonization import mapping_table, apply_mapping
from src.expression_programs import normalize_counts
from src.pathway_enrichment import read_gmt, preranked
from src.atac_programs import read_gene_activity, analyze_programs, balanced_standardize
from src.protein_programs import read_adt_counts, analyze_protein, panel_mapping, centered_log_adt, association

PROGRAMS = ['Inflammatory Response', 'TNF-alpha Signaling via NF-kB']
TARGET = 'CD14 monocytes'


def audit_inputs(paths, discovery_root, output, min_cells=50):
    """Metadata-only eligibility audit; freeze common reference before outcomes."""
    output, discovery_root = Path(output), Path(discovery_root)
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((discovery_root / 'manifest.json').read_text())
    for p in ['rna_concordance/shared_gene_universe.csv', 'gsea/used_collection.gmt']:
        if sha256(discovery_root / p) != original['artifact_sha256'][p]:
            raise ValueError(f'Discovery fingerprint mismatch: {p}')
    inspected = {d: inspect_h5ad(p) for d, p in paths.items()}
    mapping = mapping_table(inspected['cite'][0].cell_type.unique(), inspected['multiome'][0].cell_type.unique())
    # Require the same discovery mapping policy; custom overrides need explicit porting.
    known = discovery_root / 'donor_selection/celltype_mapping.csv'
    if known.exists():
        old = pd.read_csv(known)
        compare = old.merge(mapping, on=['dataset', 'original_label'], suffixes=('_old', '_new'))
        if not compare.harmonized_label_old.fillna('').eq(compare.harmonized_label_new.fillna('')).all():
            raise ValueError('Discovery mapping override differs; port reviewed mapping before validation')
    metadata = {d: apply_mapping(parts[0], mapping, d) for d, parts in inspected.items()}
    mapping.to_csv(output / 'mapping_audit.csv', index=False)
    counts = pd.concat([obs.groupby(['DonorID', 'Site', 'cell_type_harmonized'], dropna=False).size().rename('n_cells').reset_index().assign(dataset=d) for d, obs in metadata.items()], ignore_index=True)
    counts.to_csv(output / 'donor_site_coverage.csv', index=False)
    donors = sorted(set().union(*(set(o.DonorID) for o in metadata.values())))
    rows, type_sets = [], {}
    for donor in donors:
        by = {d: obs.loc[obs.DonorID == donor, 'cell_type_harmonized'].value_counts() for d, obs in metadata.items()}
        types = set.intersection(*(set(v[v >= min_cells].index) for v in by.values())) & set(original['retained_cell_types'])
        eligible = TARGET in types and len(types) >= 2
        rows.append(dict(donor=donor, role='discovery' if donor == str(original['donor']) else 'validation',
                         cite_cd14=int(by['cite'].get(TARGET, 0)), multiome_cd14=int(by['multiome'].get(TARGET, 0)),
                         eligible=eligible, reason='adequate target and comparator coverage' if eligible else 'missing capture, target or comparator coverage'))
        if eligible:
            type_sets[donor] = types
    eligibility = pd.DataFrame(rows)
    eligibility.to_csv(output / 'donor_eligibility.csv', index=False)
    if not type_sets:
        raise ValueError('No eligible donors; inspect donor_eligibility.csv')
    reference = sorted(set.intersection(*type_sets.values()))
    if TARGET not in reference or len(reference) < 2:
        raise ValueError('No common comparator reference across eligible donors; review coverage before changing protocol')
    universe_table = pd.read_csv(discovery_root / 'rna_concordance/shared_gene_universe.csv')
    universe = sorted(universe_table.loc[universe_table.included_in_comparison, 'gene'])
    for d, (_, var, _) in inspected.items():
        if not set(universe) <= set(var.index[var.feature_types == 'GEX']):
            raise ValueError(f'{d} lacks frozen universe genes')
    protocol = dict(discovery_donor=str(original['donor']), programs=PROGRAMS, target=TARGET,
                    reference_types=reference, min_cells=min_cells, universe=universe,
                    gmt_sha256=sha256(discovery_root / 'gsea/used_collection.gmt'),
                    discovery_manifest_sha256=sha256(discovery_root / 'manifest.json'),
                    inputs={d: parts[2] for d, parts in inspected.items()},
                    reference_note='Common discovery-listed types meeting cell threshold in every eligible donor/capture; may differ from nb04 reference')
    write_json(output / 'protocol.json', protocol)
    return inspected, metadata, eligibility, protocol


def pseudobulk(counts, obs, genes, output, min_cells=50):
    """Sum integer RNA counts, keeping donor/site/type distinct; mean log is separate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = sparse.csr_matrix(counts)
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.allclose(counts.data, np.rint(counts.data), atol=1e-6, rtol=0):
        raise ValueError('Pseudobulk requires finite nonnegative integer counts')
    counts = counts.astype(np.int64)
    norm = normalize_counts(counts)
    rows, totals, means = [], [], []
    group_frame = obs.reset_index(drop=True)
    for key, indices in group_frame.groupby(['DonorID', 'Site', 'cell_type_harmonized'], sort=True).indices.items():
        total = sparse.csr_matrix(counts[indices].sum(axis=0))
        totals.append(total)
        means.append(sparse.csr_matrix(norm[indices].mean(axis=0)))
        rows.append(dict(sample_id=f'sample_{len(rows):05}', donor=key[0], site=key[1], cell_type=key[2],
                         n_cells=len(indices), eligible_min_cells=len(indices) >= min_cells, library_counts=int(total.sum())))
    summed = sparse.vstack(totals, format='csr')
    assert int(summed.sum()) == int(counts.sum())
    sparse.save_npz(output / 'raw_count_sums.npz', summed)
    sparse.save_npz(output / 'mean_cell_log1p_cp10k.npz', sparse.vstack(means, format='csr'))
    # Descriptive visualization only; not a substitute for edgeR/DESeq2 offsets.
    sparse.save_npz(output / 'sum_log1p_cp10k.npz', normalize_counts(summed))
    pd.DataFrame(rows).to_csv(output / 'samples.csv', index=False)
    pd.Series(genes, name='gene').to_csv(output / 'genes.csv', index=False)


def rna_enrichment(x, obs, genes, reference, universe, sets, min_cells, permutations, seed):
    """All Hallmark tests per ranking, so q-values aren't corrected for two targets only."""
    cols = pd.Index(genes).get_indexer(universe)
    output, coverage = [], []
    for site in ['ALL'] + sorted(obs.Site.unique()):
        mask = np.ones(len(obs), bool) if site == 'ALL' else obs.Site.eq(site).to_numpy()
        counts = obs.loc[mask, 'cell_type_harmonized'].value_counts()
        valid = all(counts.get(ct, 0) >= min_cells for ct in reference)
        coverage.append(dict(site=site, tested=valid, n_target=int(counts.get(TARGET, 0)),
                             missing_reference_types=';'.join(ct for ct in reference if counts.get(ct, 0) < min_cells)))
        if not valid:
            continue
        means = {ct: np.asarray(x[mask & obs.cell_type_harmonized.eq(ct).to_numpy()][:, cols].mean(axis=0)).ravel() for ct in reference}
        rank = means[TARGET] - np.mean([v for k, v in means.items() if k != TARGET], axis=0)
        result = preranked(pd.Series(rank, index=universe), sets, permutations=permutations, seed=seed)
        result['site'] = site
        output.append(result)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(), pd.DataFrame(coverage)


def leave_one_out(x, obs, genes, counts, adts, sets, universe, min_cells):
    bio = ~panel_mapping(adts, genes).is_control.to_numpy()
    counts, adts = counts[:, bio], np.asarray(adts)[bio]
    obs = obs.copy()
    obs['adt_library_counts'] = counts.sum(axis=1)
    keep = obs.adt_library_counts.to_numpy() > 0
    obs, x, counts = obs.loc[keep], x[keep], counts[keep]
    y = centered_log_adt(counts)
    rows = []
    for pathway in PROGRAMS:
        members = sorted(set(sets[pathway]) & set(universe) & set(genes))
        z, _, _, valid = balanced_standardize(x[:, pd.Index(genes).get_indexer(members)].toarray(), obs.cell_type_harmonized)
        z, members = z[:, valid], np.asarray(members)[valid]
        for adt, gene in [('CD14', 'CD14'), ('CD88', 'C5AR1'), ('CD54', 'ICAM1')]:
            if gene not in sets[pathway]:
                continue
            usable = adt in adts and gene in members and len(members) >= 11
            for site in ['ALL'] + sorted(obs.Site.unique()):
                mask = obs.cell_type_harmonized.eq(TARGET).to_numpy()
                if site != 'ALL':
                    mask &= obs.Site.eq(site).to_numpy()
                row = dict(pathway=pathway, adt=adt, gene=gene, site=site, n_cells=int(mask.sum()),
                           full_adjusted=np.nan, loo_adjusted=np.nan, status='insufficient coverage')
                if usable:
                    values = y[mask, list(adts).index(adt)]
                    full = association(z[mask].mean(axis=1), values, obs.loc[mask], min_cells)
                    loo = association(z[mask][:, members != gene].mean(axis=1), values, obs.loc[mask], min_cells)
                    row.update(full_adjusted=full['adjusted_pearson'], loo_adjusted=loo['adjusted_pearson'],
                               status='tested' if np.isfinite(loo['adjusted_pearson']) else 'insufficient cells/variation')
                rows.append(row)
    return pd.DataFrame(rows)


def run_cross_donor(paths, discovery_root, output, min_cells=50, permutations=1000, seed=42):
    output, discovery_root = Path(output), Path(discovery_root)
    if (output / 'status.json').exists():
        raise FileExistsError('Use a new output directory for another run; completed or partial donor results are preserved')
    start = time.monotonic()
    inspected, metadata, eligibility, protocol = audit_inputs(paths, discovery_root, output, min_cells)
    write_json(output / 'status.json', {'state': 'in_progress'})
    protocol.update(permutations=permutations, seed=seed)
    write_json(output / 'protocol.json', protocol)
    sets = read_gmt(discovery_root / 'gsea/used_collection.gmt')
    universe, reference = protocol['universe'], protocol['reference_types']
    chosen = pd.DataFrame({'cell_type': TARGET, 'pathway': PROGRAMS, 'shared_leading_edge': ''})
    donor_results = []
    for donor in eligibility.loc[eligibility.eligible, 'donor']:
        folder = output / f'donor_{donor}'
        folder.mkdir(exist_ok=True)
        print(f'[{(time.monotonic()-start)/60:.1f} min] Donor {donor}', flush=True)
        donor_rna, enrichment = {}, {}
        for d, path in paths.items():
            all_obs = metadata[d]
            mask = all_obs.DonorID.eq(donor) & all_obs.cell_type_harmonized.notna()
            raw, genes = read_rna_counts(path, np.flatnonzero(mask), inspected[d][1])
            obs = all_obs.loc[mask].copy()
            print(f'  {d}: raw RNA read; exporting site-resolved pseudobulk', flush=True)
            pseudobulk(raw, obs, genes, folder / f'pseudobulk_{d}', min_cells)
            retain = obs.cell_type_harmonized.isin(reference).to_numpy()
            raw, obs = raw[retain], obs.loc[retain].copy()
            obs['rna_library_counts'] = np.asarray(raw.sum(axis=1)).ravel()
            x = normalize_counts(raw)
            del raw
            donor_rna[d] = (x, obs, genes)
            enrich, coverage = rna_enrichment(x, obs, genes, reference, universe, sets, min_cells, permutations, seed)
            enrich.to_csv(folder / f'{d}_enrichment.csv', index=False)
            coverage.to_csv(folder / f'{d}_rna_site_coverage.csv', index=False)
            enrichment[d] = enrich
        x, obs, genes = donor_rna['multiome']
        wanted = set().union(*(set(sets[p]) for p in PROGRAMS)) & set(universe)
        activity, a_genes, qc = read_gene_activity(paths['multiome'], obs, wanted)
        obs = obs.join(qc, validate='one_to_one')
        obs[['DonorID', 'Site', 'cell_type_harmonized', 'atac_nonzero_library']].to_csv(folder / 'atac_cell_qc.csv')
        keep = obs.atac_nonzero_library.to_numpy()
        obs, x, activity = obs.loc[keep].copy(), x[keep], activity[keep]
        if not set(reference) <= set(obs.cell_type_harmonized):
            raise ValueError('ATAC QC removed an entire reference type; review donor QC')
        depth = 'atac_activity_total'
        for candidate in ['ATAC_nCount_peaks', 'ATAC_atac_fragments']:
            if candidate in obs:
                v = pd.to_numeric(obs[candidate], errors='coerce')
                if np.isfinite(v).all() and (v >= 0).all() and v.nunique() > 1:
                    depth = candidate
                    break
        obs['atac_depth_covariate'] = pd.to_numeric(obs[depth])
        atac = analyze_programs(x[:, pd.Index(genes).get_indexer(a_genes)].toarray(), activity.toarray(), a_genes, obs, chosen, sets, set(universe), min_cells=min_cells)
        for name in ['coverage', 'associations', 'population_summary', 'gene_membership']:
            atac[name].to_csv(folder / f'atac_{name}.csv', index=False)
        x, obs, genes = donor_rna['cite']
        counts, adts = read_adt_counts(paths['cite'], obs, progress=lambda m: None)
        protein, _ = analyze_protein(x, genes, counts, adts, obs, chosen, sets, set(universe), min_cells=min_cells, progress=lambda m: print('  '+m, flush=True))
        for name in ['coverage', 'associations', 'direct_gene_associations', 'population_summary', 'cell_qc', 'mapping']:
            protein[name].to_csv(folder / f'protein_{name}.csv', index=False)
        loo = leave_one_out(x, obs, genes, counts, adts, sets, universe, min_cells)
        loo.to_csv(folder / 'protein_leave_one_out.csv', index=False)
        for pathway in PROGRAMS:
            r = dict(donor=donor, role='discovery' if donor == protocol['discovery_donor'] else 'validation', pathway=pathway, atac_depth_covariate=depth)
            for d in paths:
                e = enrichment[d].query('site == "ALL" and pathway == @pathway').iloc[0]
                r.update({f'{d}_NES': e.NES, f'{d}_q': e.q_value})
            primary = atac['associations']
            a = primary[(primary.pathway == pathway) & (primary.scope == 'cell_type') & (primary.group == TARGET)]
            r['atac_adjusted'] = a.adjusted_pearson.iloc[0] if len(a) else np.nan
            for adt in ['CD14', 'CD88', 'CD54']:
                p = protein['associations']
                p = p[(p.pathway == pathway) & (p.adt == adt) & (p.scope == 'target')]
                r[f'{adt}_module_adjusted'] = p.adjusted_pearson.iloc[0] if len(p) else np.nan
                l = loo[(loo.pathway == pathway) & (loo.adt == adt) & (loo.site == 'ALL')]
                r[f'{adt}_loo_adjusted'] = l.loo_adjusted.iloc[0] if len(l) else np.nan
            donor_results.append(r)
        write_json(folder / 'status.json', {'state': 'complete', 'donor': donor})
    result = pd.DataFrame(donor_results)
    result.to_csv(output / 'donor_evidence.csv', index=False)
    summaries = []
    for pathway, part in result[result.role == 'validation'].groupby('pathway'):
        for field in ['atac_adjusted', 'CD14_module_adjusted', 'CD88_module_adjusted', 'CD54_module_adjusted']:
            values = part[field].dropna()
            summaries.append(dict(pathway=pathway, measurement=field, n_eligible_validation_donors=len(part), n_testable_donors=len(values), n_positive_donors=int((values > 0).sum()), median=values.median(), minimum=values.min(), maximum=values.max()))
    summary = pd.DataFrame(summaries, columns=['pathway','measurement','n_eligible_validation_donors','n_testable_donors','n_positive_donors','median','minimum','maximum'])
    summary.to_csv(output / 'validation_donor_summary.csv', index=False)
    from src.cross_donor_plots import plot_evidence
    plot_evidence(result, output)
    write_json(output / 'manifest.json', dict(protocol_sha256=sha256(output / 'protocol.json'), elapsed_minutes=(time.monotonic()-start)/60,
               source_sha256={d: sha256(p) for d, p in paths.items()},
               limitations=['No donor-level hypothesis test', 'Site subgroups are not independent donors', 'Fixed discovery-selected targets', 'No disease/control contrast'],
               outputs_sha256={str(p.relative_to(output)): sha256(p) for p in output.rglob('*.csv')}))
    write_json(output / 'status.json', {'state': 'complete', 'biological_review': 'required'})
    return result, summary
