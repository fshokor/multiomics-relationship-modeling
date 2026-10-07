"""Population-level evidence synthesis; never pair CITE with Multiome cells."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.single_donor_io import sha256, write_json

KEYS = ['cell_type', 'pathway']


def site_summary(frame):
    """Retain missing and negative results; sign counts are not significance tests."""
    rows = []
    keys = KEYS + (['adt'] if 'adt' in frame else [])
    for values, part in frame[frame.scope == 'target_by_site'].groupby(keys, sort=False):
        valid = pd.to_numeric(part.adjusted_pearson, errors='coerce').dropna()
        rows.append(dict(zip(keys, values), n_sites=len(part), n_testable_sites=len(valid),
                         n_positive_sites=int((valid > 0).sum()), n_negative_sites=int((valid < 0).sum()),
                         site_adjusted_min=valid.min(), site_adjusted_max=valid.max(),
                         site_values='; '.join(f'{r.group}: {r.adjusted_pearson:.3f}' if pd.notna(r.adjusted_pearson)
                                              else f'{r.group}: untested' for r in part.itertuples())))
    return pd.DataFrame(rows, columns=keys + ['n_sites', 'n_testable_sites', 'n_positive_sites', 'n_negative_sites', 'site_adjusted_min', 'site_adjusted_max', 'site_values'])


def unique(frame, keys, name):
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate keys in {name}')
    return frame


def synthesize(selected, atac_coverage, atac_target, atac_assoc, protein_coverage,
               protein_assoc, direct, protein_population, threshold=.5):
    """Keep RNA enrichment, relative abundance and within-type effects distinct."""
    unique(selected, KEYS, 'selected panel')
    summary = selected.copy()
    for name, table, fields in [
        ('atac', atac_coverage, ['n_variable_in_both', 'coverage', 'analyzed']),
        ('activity', atac_target, ['rna_mean', 'atac_mean', 'state']),
        ('protein', protein_coverage, ['n_direct_genes', 'direct_gene_coverage', 'protein_coverage_status'])]:
        unique(table, KEYS, name)
        view = table[KEYS + fields].rename(columns={c: f'{name}_{c}' for c in fields})
        summary = summary.merge(view, on=KEYS, how='left', validate='one_to_one')
    primary = atac_assoc[(atac_assoc.scope == 'cell_type') & (atac_assoc.group == atac_assoc.cell_type)]
    unique(primary, KEYS, 'ATAC target associations')
    summary = summary.merge(primary[KEYS + ['n_cells', 'pearson', 'spearman', 'adjusted_pearson']].rename(
        columns={c: f'atac_{c}' for c in ['n_cells', 'pearson', 'spearman', 'adjusted_pearson']}), on=KEYS, how='left', validate='one_to_one')
    summary = summary.merge(site_summary(atac_assoc).rename(columns=lambda c: c if c in KEYS else 'atac_' + c), on=KEYS, how='left', validate='one_to_one')
    details = protein_assoc[protein_assoc.scope == 'target'].copy()
    unique(details, KEYS + ['adt'], 'protein target')
    details = details.merge(site_summary(protein_assoc), on=KEYS + ['adt'], how='left', validate='one_to_one')
    gene = direct[direct.scope == 'target'][KEYS + ['adt', 'adjusted_pearson']].rename(columns={'adjusted_pearson': 'same_gene_adjusted_pearson'})
    details = details.merge(unique(gene, KEYS + ['adt'], 'same gene'), on=KEYS + ['adt'], how='left', validate='one_to_one')
    details = details.merge(protein_population[KEYS + ['adt', 'rna_module_mean', 'adt_mean_z', 'adt_nonzero_fraction']], on=KEYS + ['adt'], how='left', validate='one_to_one')
    details['relative_protein_state'] = details.adt_mean_z.map(lambda v: 'unmeasured/constant' if pd.isna(v) else 'high' if v >= threshold else 'low' if v <= -threshold else 'intermediate')
    interpretations = []
    for row in summary.itertuples():
        part = details[(details.cell_type == row.cell_type) & (details.pathway == row.pathway)]
        measured = part[part.evidence_type == 'direct molecular match']
        if row.protein_n_direct_genes == 0:
            note = 'Protein unmeasured; no protein-state or decoupling conclusion'
        else:
            positive = measured[(measured.n_testable_sites >= 2) & (measured.n_positive_sites == measured.n_testable_sites) & (measured.adjusted_pearson > 0)]
            note = ('Positive adjusted module association in all testable sites: ' + ', '.join(positive.adt)
                    if len(positive) else 'No direct ADT has positive adjusted module association in all testable sites')
            note += '; measured subset only; sign consistency is descriptive'
        interpretations.append(note)
    summary['protein_interpretation'] = interpretations
    summary['rna_site_replication'] = 'pending separate cross-capture pathway check'
    summary['causal_or_temporal_conclusion'] = 'not established'
    return summary, details


def run_summary(root, threshold=.5):
    root = Path(root)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Relative state threshold must be positive')
    manifests = {p: json.loads((root / p).read_text()) for p in ['manifest.json', 'atac/manifest.json', 'protein/manifest.json']}
    donor = str(manifests['manifest.json']['donor'])
    status = json.loads((root / 'status.json').read_text())
    if any(status.get(k) != 'complete' for k in ['rna_characterization', 'rna_concordance']):
        raise ValueError('Complete nb04/05 first')
    for stage in ['atac', 'protein']:
        if json.loads((root / stage / 'status.json').read_text()).get('state') != 'complete':
            raise ValueError(f'Complete {stage} stage first')
        manifest = manifests[f'{stage}/manifest.json']
        if str(manifest['donor']) != donor:
            raise ValueError('Donor mismatch between stages')
        for path, digest in manifest['upstream_sha256'].items():
            # Only summary inputs and run identities; no raw matrix loading/hashing.
            if path.endswith('.npz') or path.startswith('rna_multiome/'):
                continue
            if sha256(root / path) != digest:
                raise ValueError(f'Stale {stage} provenance: {path}')
    paths = ['atac/selected_rna_programs.csv', 'protein/selected_rna_programs.csv',
             'atac/coverage.csv', 'atac/target_program_summary.csv', 'atac/associations.csv',
             'protein/coverage.csv', 'protein/associations.csv', 'protein/direct_gene_associations.csv',
             'protein/population_summary.csv', 'atac/state_threshold_sensitivity.csv']
    frames = {p: pd.read_csv(root / p) for p in paths}
    selected = frames[paths[0]]
    other = frames[paths[1]]
    pd.testing.assert_frame_equal(selected.sort_values(KEYS).reset_index(drop=True), other.sort_values(KEYS).reset_index(drop=True))
    panel = set(map(tuple, selected[KEYS].to_numpy()))
    for path in ['atac/coverage.csv', 'protein/coverage.csv']:
        if set(map(tuple, frames[path][KEYS].to_numpy())) != panel:
            raise ValueError(f'Panel mismatch: {path}')
    summary, detail = synthesize(selected, *(frames[p] for p in paths[2:9]), threshold=threshold)
    output = root / 'multilayer'
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'program_evidence.csv', index=False)
    detail.to_csv(output / 'protein_evidence.csv', index=False)
    site_summary(frames['protein/direct_gene_associations.csv']).to_csv(output / 'same_gene_site_sensitivity.csv', index=False)
    frames[paths[-1]].to_csv(output / 'atac_state_sensitivity.csv', index=False)
    # Preserve full signed/site data for audit, including untested rows.
    for stage in ['atac', 'protein']:
        frames[f'{stage}/associations.csv'].to_csv(output / f'{stage}_associations.csv', index=False)
    from src.multimodal_plots import make_figures
    figures = make_figures(summary, detail, output / 'figures')
    provenance = dict(donor=donor, relative_protein_threshold=threshold, figures=figures,
                      upstream_sha256={p: sha256(root / p) for p in paths + list(manifests)},
                      integrity_limit='Older stages do not hash their own output tables; fingerprints record the inputs used here, not proof against prior edits.',
                      unresolved=['Cross-capture RNA pathway site replication', 'Unavailable nb05 NES audit', 'Multi-donor validation'],
                      interpretation='Population-level synthesis; no cross-capture cell matching or composite activation score')
    write_json(output / 'manifest.json', provenance)
    write_json(output / 'status.json', {'state': 'complete', 'review': 'required'})
    return summary, detail, provenance
