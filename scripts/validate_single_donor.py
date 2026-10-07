"""Run remaining single-donor checks without changing upstream results."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.single_donor_io import load_rna, sha256, write_json
from src.pathway_enrichment import read_gmt, preranked, enrichment_score
from src.atac_programs import INITIAL_PROGRAMS, balanced_standardize
from src.protein_programs import read_adt_counts, panel_mapping, centered_log_adt, association


def main():
    root = ROOT / 'results/single_donor'
    out = root / 'validation'
    out.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    for path, digest in manifest['artifact_sha256'].items():
        if sha256(root / path) != digest:
            raise ValueError(f'Fingerprint mismatch: {path}')
    print('All manifest artifact fingerprints match.', flush=True)
    data = {d: load_rna(root / f'rna_{d}') for d in ['cite', 'multiome']}
    for x, obs, genes in data.values():
        assert obs.DonorID.eq(str(manifest['donor'])).all()
        assert x.shape == (len(obs), len(genes)) and obs.index.is_unique
    sets = read_gmt(root / 'gsea/used_collection.gmt')
    u = pd.read_csv(root / 'rna_concordance/shared_gene_universe.csv')
    universe = set(u.loc[u.included_in_comparison, 'gene'])
    selected = list(INITIAL_PROGRAMS)

    # Audit each missing score using recorded null counts, then an independent
    # larger targeted null simulation. Do not silently replace old q-values.
    audits = []
    for d in data:
        enrichment = pd.read_csv(root / f'gsea/{d}_enrichment.csv')
        programs = pd.read_csv(root / f'rna_{d}/gene_programs.csv')
        for row in enrichment[enrichment.NES.isna()].itertuples():
            rank = programs[programs.cell_type == row.cell_type].set_index('gene').loc[sorted(universe), 'specificity']
            rank = rank.sort_index().sort_values(ascending=False, kind='stable')
            hits = np.flatnonzero(rank.index.isin(sets[row.pathway]))
            es, _ = enrichment_score(rank.to_numpy(), hits)
            assert np.isclose(es, row.ES)
            rng = np.random.default_rng(20261007)
            null = np.array([enrichment_score(rank.to_numpy(), rng.choice(len(rank), len(hits), replace=False))[0] for _ in range(10000)])
            side = null[null >= 0] if es >= 0 else null[null < 0]
            audits.append(dict(dataset=d, cell_type=row.cell_type, pathway=row.pathway, ES=es,
                               original_n_same_sign=row.n_null_same_sign, original_permutations=row.permutations,
                               extra_permutations=10000, extra_n_same_sign=len(side),
                               exploratory_NES=es / np.abs(side).mean() if len(side) else np.nan,
                               selected_for_followup=(row.cell_type, row.pathway) in selected,
                               cause='No same-sign null draws; normalization and conditional p-value undefined' if row.n_null_same_sign == 0 else 'Requires further audit'))
            print('Audited missing NES:', d, row.cell_type, row.pathway, len(side), flush=True)
    pd.DataFrame(audits).to_csv(out / 'missing_nes_audit.csv', index=False)

    # Same-site captures use exactly the same comparator types, >=50 cells in
    # each capture, and the frozen shared gene universe. BH covers all eligible
    # Hallmark sets for each target/capture/site, not only the selected pathways.
    sites = sorted(set(data['cite'][1].Site) & set(data['multiome'][1].Site))
    site_rows, coverage = [], []
    for site in sites:
        counts = {d: obs[obs.Site == site].cell_type_harmonized.value_counts() for d, (_, obs, _) in data.items()}
        types = sorted(set(counts['cite'][counts['cite'] >= 50].index) & set(counts['multiome'][counts['multiome'] >= 50].index))
        for ct, pathway in selected:
            coverage.append(dict(site=site, cell_type=ct, pathway=pathway, n_cite=int(counts['cite'].get(ct, 0)), n_multiome=int(counts['multiome'].get(ct, 0)), tested=ct in types and len(types) >= 2, reference_types=';'.join(types)))
        for d, (x, obs, genes) in data.items():
            gi = pd.Index(genes).get_indexer(sorted(universe))
            assert (gi >= 0).all()
            means = {ct: np.asarray(x[(obs.Site.eq(site) & obs.cell_type_harmonized.eq(ct)).to_numpy()][:, gi].mean(axis=0)).ravel() for ct in types}
            for ct in sorted({ct for ct, _ in selected} & set(types)):
                if len(types) < 2:
                    continue
                specificity = means[ct] - np.mean([v for k, v in means.items() if k != ct], axis=0)
                result = preranked(pd.Series(specificity, index=sorted(universe)), sets, permutations=1000, seed=42)
                result['dataset'], result['site'], result['cell_type'] = d, site, ct
                result['n_cells'] = counts[d][ct]
                site_rows.append(result)
                print('Site enrichment:', site, d, ct, flush=True)
    full = pd.concat(site_rows, ignore_index=True)
    full.to_csv(out / 'site_enrichment_all_sets.csv', index=False)
    pd.DataFrame(coverage).to_csv(out / 'site_coverage.csv', index=False)
    panel = pd.DataFrame(selected, columns=['cell_type', 'pathway'])
    selected_sites = full.merge(panel, on=['cell_type', 'pathway'])
    paired = selected_sites[selected_sites.dataset == 'cite'].merge(selected_sites[selected_sites.dataset == 'multiome'], on=['site', 'cell_type', 'pathway'], suffixes=('_cite', '_multiome'))
    paired['positive_both'] = (paired.NES_cite > 0) & (paired.NES_multiome > 0)
    paired['q05_positive_both'] = paired.positive_both & (paired.q_value_cite <= .05) & (paired.q_value_multiome <= .05)
    paired.to_csv(out / 'site_pathway_comparison.csv', index=False)

    # Reproduce nb07 full-module effects before removing one gene. Use the same
    # per-gene reference moments and subtract only the matched gene contribution.
    x, obs, genes = data['cite']
    path = ROOT / 'data/benchmark/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad'
    assert path.stat().st_size == manifest['inputs']['cite']['file_bytes']
    counts, adts = read_adt_counts(path, obs)
    mapping = panel_mapping(adts, genes)
    bio = ~mapping.is_control.to_numpy()
    counts, adts = counts[:, bio], adts[bio]
    obs = obs.copy()
    obs['adt_library_counts'] = counts.sum(axis=1)
    keep = obs.adt_library_counts.to_numpy() > 0
    obs, x, counts = obs.loc[keep], x[keep], counts[keep]
    adt_values = centered_log_adt(counts)
    targets = [('CD14', 'CD14'), ('CD88', 'C5AR1'), ('CD54', 'ICAM1')]
    members = sorted(set(sets['Inflammatory Response']) & universe & set(genes))
    z, _, _, valid = balanced_standardize(x[:, pd.Index(genes).get_indexer(members)].toarray(), obs.cell_type_harmonized.to_numpy())
    members, z = np.asarray(members)[valid], z[:, valid]
    full_score = z.mean(axis=1)
    rows = []
    target = obs.cell_type_harmonized.eq('CD14 monocytes').to_numpy()
    existing = pd.read_csv(root / 'protein/associations.csv')
    for adt, gene in targets:
        assert gene in members
        loo = z[:, members != gene].mean(axis=1)
        y = adt_values[:, list(adts).index(adt)]
        for site in ['ALL'] + sorted(obs.Site.unique()):
            mask = target if site == 'ALL' else target & obs.Site.eq(site).to_numpy()
            original = association(full_score[mask], y[mask], obs.loc[mask])
            removed = association(loo[mask], y[mask], obs.loc[mask])
            old = existing[(existing.pathway == 'Inflammatory Response') & existing.adt.eq(adt) & existing.group.eq(site)].iloc[0]
            assert np.isclose(original['adjusted_pearson'], old.adjusted_pearson, atol=1e-8)
            rows.append(dict(adt=adt, removed_gene=gene, site=site, n_genes_full=len(members), n_genes_loo=len(members)-1,
                             n_cells=original['n_cells'], full_adjusted=original['adjusted_pearson'], loo_adjusted=removed['adjusted_pearson'],
                             full_spearman=original['spearman'], loo_spearman=removed['spearman'],
                             delta_adjusted=removed['adjusted_pearson']-original['adjusted_pearson']))
    pd.DataFrame(rows).to_csv(out / 'protein_leave_one_gene_out.csv', index=False)
    write_json(out / 'manifest.json', dict(donor=manifest['donor'], seed=42, site_permutations=1000,
               missing_nes_seed=20261007, missing_nes_permutations=10000, min_cells=50,
               root_manifest_sha256=sha256(root / 'manifest.json'), cite_source_sha256=sha256(path),
               limitations=['Single donor', 'Site reference populations differ between sites', 'Gene-set permutations do not preserve gene correlation', 'LOO preserves other correlated pathway genes'],
               outputs_sha256={p.name: sha256(p) for p in out.glob('*.csv')}))
    print('Validation complete:', out, flush=True)


if __name__ == '__main__':
    main()
