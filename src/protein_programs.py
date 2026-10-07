"""CITE RNA–ADT support for reviewed programs; no cross-capture cell matching."""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.analysis.cd_gene_mapping import CD_TO_GENE
from src.atac_programs import balanced_standardize
from src.rna_concordance import correlations
from src.pathway_enrichment import read_gmt
from src.single_donor_io import load_rna, sha256, write_json


# Override older representative-chain/epitope mappings before exact-name matching.
AMBIGUOUS = {
    'CD3': 'Multi-subunit complex; CD3D is only a representative chain',
    'CD8': 'Antigen label alone does not resolve chain/complex specificity',
    'CD16': 'FCGR3A/FCGR3B specificity requires antibody clone information',
    'CD32': 'FCGR2 family specificity requires antibody clone information',
    'CD57': 'Carbohydrate epitope, not a direct B3GAT1 protein measurement',
    'CD45RA': 'PTPRC isoform; total-gene RNA does not distinguish this epitope',
    'CD45RO': 'PTPRC isoform; total-gene RNA does not distinguish this epitope',
    'HLA-DR': 'Multi-chain/family antigen; no unique gene match',
    'HLA-A-B-C': 'Multiple HLA proteins; no unique gene match',
}


def panel_mapping(adt_names, rna_genes):
    """Conservative exact-label mapping. No suffix stripping or guessed symbols."""
    rna_genes = set(rna_genes)
    rows = []
    for name in adt_names:
        control = bool(re.search(r'isotype|control|igg[12]', name, flags=re.I))
        gene, reason = None, ''
        if control:
            reason = 'Isotype/control label; excluded from biological ADTs'
        elif name in AMBIGUOUS:
            reason = AMBIGUOUS[name]
        elif name.startswith(('TCR', 'CD158')) or (name in CD_TO_GENE and CD_TO_GENE[name] is None):
            reason = 'Clone/family-specific or unresolved antigen; not a direct molecular match'
        elif name in CD_TO_GENE:
            gene, reason = CD_TO_GENE[name], 'Existing curated CD-to-gene alias; conservative ambiguity overrides applied'
        elif name in rna_genes:
            gene, reason = name, 'Exact antigen/gene symbol match'
        else:
            reason = 'No documented unique mapping; inspect antibody metadata before adding one'
        rows.append(dict(adt=name, gene=gene, is_control=control,
                         direct_match=gene is not None and gene in rna_genes,
                         mapping_status='direct' if gene in rna_genes else 'RNA unmeasured' if gene else 'unresolved',
                         reason=reason))
    return pd.DataFrame(rows)


def phenotype_candidates(selected):
    """Nonspecific surface context, not pathway membership or direct validation.

    References support antigen function, not an expected direction of correlation
    with the Hallmark score. No forced proxies for intracellular E2F/Myc programs.
    """
    rows = []
    for row in selected.itertuples():
        if row.pathway not in ['Inflammatory Response', 'TNF-alpha Signaling via NF-kB', 'Interferon Alpha Response']:
            continue
        for adt, reason, source in [
            ('CD86', 'Costimulatory surface phenotype; exploratory and not specific to this RNA pathway',
             'https://www.uniprot.org/uniprotkb/P42081/entry'),
            ('HLA-DR', 'Antigen-presentation surface phenotype; multi-chain antigen, not a direct HLA-DRA match or interferon-specific readout',
             'https://www.uniprot.org/entry/P01903'),
        ]:
            rows.append(dict(cell_type=row.cell_type, pathway=row.pathway, adt=adt,
                             rationale=reason, reference=source, expected_direction='unspecified'))
    return pd.DataFrame(rows, columns=['cell_type', 'pathway', 'adt', 'rationale', 'reference', 'expected_direction'])


def load_protein_inputs(root, progress=print):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    if status.get('rna_characterization') != 'complete' or status.get('rna_concordance') != 'complete':
        raise ValueError('Complete nb04 and nb05 first')
    if json.loads((root / 'atac/status.json').read_text()).get('state') != 'complete':
        raise ValueError('Complete and review nb06 before nb07')
    atac = json.loads((root / 'atac/manifest.json').read_text())
    if str(atac['donor']) != str(manifest['donor']):
        raise ValueError('nb06 donor differs from current RNA run')
    # No need to read or hash the full Multiome RNA cache for a CITE-only stage.
    hashes = {p: v for p, v in manifest['artifact_sha256'].items()
              if p.startswith(('rna_cite/', 'gsea/', 'rna_concordance/'))}
    for path, digest in hashes.items():
        progress(f'Checking RNA artifact: {path}')
        if sha256(root / path) != digest:
            raise ValueError(f'Stale artifact {path}; use consistent nb04/05 results')
    for path in ['manifest.json', 'rna_concordance/followup_programs.csv', 'gsea/used_collection.gmt']:
        if sha256(root / path) != atac['upstream_sha256'][path]:
            raise ValueError('nb06 comes from a different RNA run')
    selected = pd.read_csv(root / 'atac/selected_rna_programs.csv')
    keys = ['cell_type', 'pathway']
    if selected.empty or selected.duplicated(keys).any():
        raise ValueError('Expected unique selected nb06 programs')
    followup = pd.read_csv(root / 'rna_concordance/followup_programs.csv').set_index(keys)
    requested = list(zip(selected.cell_type, selected.pathway))
    current = followup.loc[requested]
    if not current.selected_for_followup.astype(str).str.lower().eq('true').all():
        raise ValueError('nb06 panel no longer matches the current RNA shortlist')
    for d in ['cite', 'multiome']:
        for field in ['NES', 'q_value']:
            if not np.allclose(selected[f'{field}_{d}'], current[f'{field}_{d}'], equal_nan=False):
                raise ValueError('nb06 panel enrichment differs from current shortlist')
    x, obs, genes = load_rna(root / 'rna_cite')
    if x.shape != (len(obs), len(genes)) or not obs.index.is_unique or len(set(genes)) != len(genes):
        raise ValueError('Invalid CITE RNA cache dimensions or identifiers')
    if not obs.DonorID.astype(str).eq(str(manifest['donor'])).all():
        raise ValueError('CITE cache contains a different donor')
    universe = pd.read_csv(root / 'rna_concordance/shared_gene_universe.csv')
    return manifest, selected, x, obs, np.asarray(genes), set(universe.loc[universe.included_in_comparison, 'gene'])


def read_adt_counts(path, obs, chunk_size=1024, progress=print):
    """Align CITE cells by ID, select ADT by position, retain raw count integrity."""
    import h5py
    from anndata.io import read_elem, sparse_dataset
    with h5py.File(path, 'r') as f:
        source, var = read_elem(f['obs']), read_elem(f['var'])
        if not source.index.is_unique or not obs.index.is_unique:
            raise ValueError('Duplicate CITE cell IDs')
        rows = source.index.get_indexer(obs.index)
        if (rows < 0).any():
            raise ValueError('RNA cells missing from CITE source')
        for col in ['DonorID', 'Site', 'cell_type']:
            if not np.array_equal(source.iloc[rows][col].astype(str), obs[col].astype(str)):
                raise ValueError(f'CITE metadata mismatch: {col}')
        cols = np.flatnonzero(var.feature_types.to_numpy() == 'ADT')
        names = var.index[cols].astype(str).to_numpy()
        if not len(cols) or len(set(names)) != len(names):
            raise ValueError('ADT features must exist and be unique within ADT')
        node = f['layers/counts']
        backed = sparse_dataset(node) if isinstance(node, h5py.Group) else node
        order = np.argsort(rows)
        blocks = []
        for start in range(0, len(rows), chunk_size):
            block = sparse.csr_matrix(backed[rows[order[start:start + chunk_size]], :])[:, cols]
            blocks.append(block.toarray())
            progress(f'ADT cells read: {min(start + chunk_size, len(rows)):,}/{len(rows):,}')
    counts = np.vstack(blocks)[np.argsort(order)].astype(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.allclose(counts, np.rint(counts), atol=1e-6, rtol=0):
        raise ValueError('ADT counts must be finite nonnegative integers')
    return counts, names


def centered_log_adt(counts):
    """Explicit per-cell centered log1p: log(1+x) - mean_panel(log(1+x))."""
    logged = np.log1p(np.asarray(counts, float))
    return logged - logged.mean(axis=1, keepdims=True)


def association(x, y, obs, minimum=50):
    """Descriptive raw/residual effects; no cell-level inferential p-values."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, sub = np.asarray(x)[finite], np.asarray(y)[finite], obs.iloc[np.flatnonzero(finite)]
    result = dict(n_cells=len(sub), tested=False, pearson=np.nan, spearman=np.nan,
                  adjusted_pearson=np.nan, residual_spearman=np.nan)
    if len(sub) < minimum or np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return result
    result.update(tested=True, pearson=correlations(x, y)[0], spearman=correlations(x, y)[1])
    columns = [np.ones(len(sub))]
    for col in ['rna_library_counts', 'adt_library_counts']:
        values = np.log1p(sub[col].to_numpy(float))
        if np.std(values) > 1e-10:
            columns.append((values - values.mean()) / values.std())
    columns.extend(pd.get_dummies(sub.Site.astype(str), drop_first=True, dtype=float).to_numpy().T)
    design = np.column_stack(columns)
    if len(sub) > np.linalg.matrix_rank(design) + 3:
        rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        if np.std(rx) > 1e-10 and np.std(ry) > 1e-10:
            result['adjusted_pearson'], result['residual_spearman'] = correlations(rx, ry)
    return result


def analyze_protein(rna, genes, counts, adts, obs, selected, sets, universe,
                    min_cells=50, min_rna_genes=10, progress=print):
    if rna.shape != (len(obs), len(genes)) or counts.shape != (len(obs), len(adts)):
        raise ValueError('Paired RNA/ADT dimensions do not match metadata')
    mapping = panel_mapping(adts, genes)
    biological = ~mapping.is_control.to_numpy()
    if not biological.any():
        raise ValueError('No biological ADTs')
    obs = obs.copy()
    obs['adt_library_counts'] = counts[:, biological].sum(axis=1)
    keep = obs.adt_library_counts.to_numpy() > 0
    qc = obs[['DonorID', 'Site', 'cell_type_harmonized', 'rna_library_counts', 'adt_library_counts']].copy()
    qc['included_protein'] = keep
    if not keep.any():
        raise ValueError('All CITE cells have zero biological ADT counts')
    obs, rna, counts = obs.loc[keep], rna[keep], counts[keep]
    adts = np.asarray(adts)[biological]
    counts = counts[:, biological]
    transformed, logged = centered_log_adt(counts), np.log1p(counts)
    labels = obs.cell_type_harmonized.to_numpy()
    if len(set(labels)) < 2:
        raise ValueError('At least two cell types needed for a relative reference')
    # Never densify the complete donor RNA: each module is processed separately.
    gene_lookup = {g: i for i, g in enumerate(genes)}
    adt_lookup = {g: i for i, g in enumerate(adts)}
    pz, _, _, pv = balanced_standardize(transformed, labels)
    phenotypes = phenotype_candidates(selected)
    coverage, associations, direct, summaries, evidence, score_frames, plot_pairs = [], [], [], [], [], [], []
    for number, row in enumerate(selected.itertuples(), 1):
        key = dict(program_id=f'P{number:02d}', cell_type=row.cell_type, pathway=row.pathway)
        progress(f'Scoring {key["program_id"]}: {row.cell_type} | {row.pathway}')
        members = set(sets[row.pathway])
        available = sorted(members & set(genes) & universe)
        rx = rna[:, [gene_lookup[g] for g in available]].toarray()
        rz, _, _, rv = balanced_standardize(rx, labels)
        used = np.asarray(available)[rv].tolist()
        valid_module = len(used) >= min_rna_genes
        score = rz[:, rv].mean(axis=1) if valid_module else np.full(len(obs), np.nan)
        direct_map = mapping[mapping.direct_match & mapping.gene.isin(members) & ~mapping.is_control]
        direct_genes = set(direct_map.gene)
        coverage.append(dict(**key, n_pathway_genes=len(members), n_rna_module_genes=len(used),
                             rna_module_tested=valid_module, n_direct_adts=len(direct_map), n_direct_genes=len(direct_genes),
                             direct_gene_coverage=len(direct_genes) / len(members),
                             protein_coverage_status='measured subset only' if direct_genes else 'no direct ADT coverage',
                             rna_genes=';'.join(used)))
        frame = obs[['DonorID', 'Site', 'cell_type_harmonized']].copy()
        frame['cell_id'] = obs.index
        for k, value in key.items():
            frame['target_cell_type' if k == 'cell_type' else k] = value
        frame['rna_score'] = score
        score_frames.append(frame.reset_index(drop=True))
        candidates = []
        for m in direct_map.itertuples():
            candidates.append((m.adt, m.gene, 'direct molecular match', 'Unique mapped antigen gene is in pathway', 'src/analysis/cd_gene_mapping.py + mapping audit'))
        for p in phenotypes[(phenotypes.cell_type == row.cell_type) & (phenotypes.pathway == row.pathway)].itertuples():
            if p.adt not in set(direct_map.adt):
                candidates.append((p.adt, None, 'phenotypic context', p.rationale, p.reference))
        target = labels == row.cell_type
        if not target.any():
            raise ValueError(f'No ADT-usable cells for {row.cell_type}')
        for adt, gene, tier, reason, reference in candidates:
            measured = adt in adt_lookup
            evidence.append(dict(**key, adt=adt, gene=gene, evidence_type=tier, measured=measured, rationale=reason, reference=reference))
            if not measured:
                continue
            j = adt_lookup[adt]
            summaries.append(dict(**key, adt=adt, gene=gene, evidence_type=tier, n_cells=int(target.sum()),
                                  rna_module_mean=float(np.nanmean(score[target])) if valid_module else np.nan,
                                  adt_mean_centered_log=float(transformed[target, j].mean()),
                                  adt_mean_z=float(pz[target, j].mean()) if pv[j] else np.nan,
                                  adt_nonzero_fraction=float((counts[target, j] > 0).mean()),
                                  interpretation='measured subset/context only; no whole-pathway validation'))
            scopes = [('target', 'ALL', target)] + [('target_by_site', site, target & (obs.Site.to_numpy() == site))
                                                   for site in sorted(obs.Site.unique())]
            for scope, group, mask in scopes:
                result = dict(**key, adt=adt, gene=gene, evidence_type=tier, scope=scope, group=group,
                              **association(score[mask], transformed[mask, j], obs.loc[mask], min_cells))
                result['log1p_adt_spearman'] = association(score[mask], logged[mask, j], obs.loc[mask], min_cells)['spearman']
                associations.append(result)
                if gene is not None:
                    expression = np.asarray(rna[:, gene_lookup[gene]].toarray()).ravel()
                    direct.append(dict(**key, adt=adt, gene=gene, scope=scope, group=group,
                                       **association(expression[mask], transformed[mask, j], obs.loc[mask], min_cells),
                                       rna_mean=float(expression[mask].mean()) if mask.any() else np.nan,
                                       adt_mean=float(transformed[mask, j].mean()) if mask.any() else np.nan))
            if len([p for p in plot_pairs if p['program_id'] == key['program_id']]) < 3:
                plot_pairs.append(dict(**key, adt=adt, evidence_type=tier, rna=score[target], adt_values=transformed[target, j],
                                       gene=gene, gene_rna=np.asarray(rna[target, gene_lookup[gene]].toarray()).ravel() if gene else None))
    columns = ['program_id', 'cell_type', 'pathway', 'adt', 'gene', 'evidence_type', 'scope', 'group', 'n_cells', 'tested', 'pearson', 'spearman', 'adjusted_pearson', 'residual_spearman', 'log1p_adt_spearman']
    tables = dict(mapping=mapping, phenotype_candidates=phenotypes, coverage=pd.DataFrame(coverage),
                  evidence=pd.DataFrame(evidence, columns=['program_id','cell_type','pathway','adt','gene','evidence_type','measured','rationale','reference']),
                  associations=pd.DataFrame(associations, columns=columns),
                  direct_gene_associations=pd.DataFrame(direct, columns=['program_id','cell_type','pathway','adt','gene','scope','group','n_cells','tested','pearson','spearman','adjusted_pearson','residual_spearman','rna_mean','adt_mean']),
                  population_summary=pd.DataFrame(summaries, columns=['program_id','cell_type','pathway','adt','gene','evidence_type','n_cells','rna_module_mean','adt_mean_centered_log','adt_mean_z','adt_nonzero_fraction','interpretation']),
                  cell_scores=pd.concat(score_frames, ignore_index=True), cell_qc=qc.reset_index(names='cell_id'))
    # No collapsed protein score: a few ADTs cannot represent an entire pathway.
    return tables, plot_pairs


def run_protein(root, cite_path, min_cells=50, progress=True):
    root, cite_path = Path(root), Path(cite_path)
    start = time.monotonic()
    def log(message):
        if progress:
            print(f'[{(time.monotonic()-start)/60:.1f} min] {message}', flush=True)
    output = root / 'protein'
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    write_json(output / 'status.json', {'state': 'in_progress'})
    manifest, selected, x, obs, genes, universe = load_protein_inputs(root, log)
    if cite_path.stat().st_size != manifest['inputs']['cite']['file_bytes']:
        raise ValueError('CITE source size differs from nb04')
    log('Reading paired ADT counts')
    counts, adts = read_adt_counts(cite_path, obs, progress=log)
    sets = read_gmt(root / 'gsea/used_collection.gmt')
    tables, plot_pairs = analyze_protein(x, genes, counts, adts, obs, selected, sets, universe, min_cells=min_cells, progress=log)
    log('Saving protein tables')
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    selected.to_csv(output / 'selected_rna_programs.csv', index=False)
    from src.protein_plots import make_figures
    log('Rendering figures')
    figures = make_figures(tables, plot_pairs, output / 'figures')
    sources = ['manifest.json', 'atac/manifest.json', 'atac/selected_rna_programs.csv',
               'rna_concordance/followup_programs.csv', 'gsea/used_collection.gmt']
    provenance = dict(donor=manifest['donor'], source=str(cite_path), source_bytes=cite_path.stat().st_size,
                      normalization='ADT raw counts: per-cell log1p minus mean log1p over biological panel',
                      sensitivity='uncentered log1p ADT counts; depth/site-adjusted correlations',
                      score='RNA module: equal-cell-type-reference gene z-scores, same shared RNA universe as nb04',
                      min_cells=min_cells, figures=figures, elapsed_minutes=(time.monotonic()-start)/60,
                      upstream_sha256={p: sha256(root / p) for p in sources},
                      limitations=['No empty-droplet/background correction', 'Limited surface panel', 'Single donor',
                                   'No whole-pathway protein validation', 'Phenotypic marker direction unspecified',
                                   'No cross-capture cell pairing', 'Association is not causality'])
    write_json(output / 'manifest.json', provenance)
    write_json(output / 'status.json', {'state': 'complete', 'review': 'required'})
    log('Complete; review coverage before interpreting protein support')
    return tables, provenance
