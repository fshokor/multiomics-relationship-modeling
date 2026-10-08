"""Descriptive RNA-bridge diagnostics on fixed evaluation cells, never cell matching."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
from scipy.stats import spearmanr


def evaluation_indices(obs, cap=20000, seed=42):
    """Equal maximum quota per donor/assay; no labels used to select cells."""
    rng = np.random.default_rng(seed)
    groups = list(obs.groupby(['assay', 'DonorID'], observed=True).indices.values())
    quota = max(1, cap // len(groups))
    return np.sort(np.concatenate([rng.choice(g, min(len(g), quota), replace=False) for g in groups]))


def knn_indices(z, k=30):
    if len(z) < 3 or not np.isfinite(z).all():
        raise ValueError('Need at least three finite cells for neighborhoods')
    k = min(k, len(z)-1)
    candidates = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(z).kneighbors(z, return_distance=False)
    # Explicit self removal also handles duplicate coordinates and distance ties.
    return np.vstack([row[row != i][:k] for i, row in enumerate(candidates)])


def category_metrics(labels, neighbors):
    labels = np.asarray(labels, str)
    categories, encoded = np.unique(labels, return_inverse=True)
    fractions = np.column_stack([(labels[neighbors] == c).mean(axis=1) for c in categories])
    entropy = -(fractions * np.log(np.maximum(fractions, 1e-15))).sum(axis=1)
    if len(categories) > 1:
        entropy /= np.log(len(categories))
    same = (labels[neighbors] == labels[:, None]).mean(axis=1)
    majority = categories[fractions.argmax(axis=1)] == labels
    return same, entropy, majority.astype(float)


def baseline_metrics(z, obs, k=30, silhouette_cap=3000, seed=42):
    """Unweighted cell summaries on the exact same evaluation sample in both spaces."""
    neighbors = knn_indices(z, k)
    cells = pd.DataFrame(index=obs.index)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(obs), min(len(obs), silhouette_cap), replace=False))
    for column, prefix in [('cell_type_harmonized','cell_type'), ('assay','assay'), ('DonorID','donor'), ('Site','site')]:
        values = obs[column].astype(object).fillna('Unresolved').astype(str).to_numpy()
        same, entropy, majority = category_metrics(values, neighbors)
        cells[prefix+'_purity'] = same
        cells[prefix+'_entropy'] = entropy
        cells[prefix+'_knn_label_agreement'] = majority
        cells[prefix+'_silhouette'] = np.nan
        if 1 < len(set(values[sub])) < len(sub):
            cells.loc[obs.index[sub], prefix+'_silhouette'] = silhouette_samples(z[sub], values[sub])
    cells['opposite_assay_fraction'] = 1-cells.assay_purity
    cells['cell_type_harmonized'] = obs.cell_type_harmonized.astype(object).fillna('Unresolved').astype(str)
    return cells, neighbors


def centroid_distance(z, assay):
    a, b = z[np.asarray(assay) == 'CITE'], z[np.asarray(assay) == 'Multiome']
    if not len(a) or not len(b):
        return np.nan, np.nan
    distance = float(np.linalg.norm(a.mean(axis=0)-b.mean(axis=0)))
    radius = np.sqrt(((a-a.mean(axis=0))**2).sum(axis=1).mean()/2 + ((b-b.mean(axis=0))**2).sum(axis=1).mean()/2)
    return distance, distance/radius if radius > 1e-12 else np.nan


def group_alignment(z, obs, columns, min_cells=50, k=30):
    rows = []
    for key, indices in obs.groupby(columns, observed=True).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        part = obs.iloc[indices]
        counts = part.assay.value_counts()
        na, nb = int(counts.get('CITE',0)), int(counts.get('Multiome',0))
        valid = min(na, nb) >= min_cells
        row = dict(zip(columns, key), n_CITE=na, n_Multiome=nb, sufficient_cells=valid,
                   cross_assay_fraction=np.nan, expected_cross_fraction=np.nan, mixing_ratio=np.nan,
                   centroid_distance=np.nan, scaled_centroid_distance=np.nan, alignment_status='insufficient cells')
        if valid:
            local = z[indices]
            neighbors = knn_indices(local, k)
            assay = part.assay.to_numpy()
            cross = (assay[neighbors] != assay[:,None]).mean(axis=1)
            # Equal weight to query assays; expected value accounts for panel imbalance.
            mixing = .5*(cross[assay=='CITE'].mean()+cross[assay=='Multiome'].mean())
            expected = .5*(nb/(na+nb-1)+na/(na+nb-1))
            distance, scaled = centroid_distance(local, assay)
            ratio = mixing/expected
            status = ('supported' if ratio >= .8 and scaled <= .5 else
                      'concern' if ratio < .5 or scaled > 1 else 'partially supported')
            row.update(cross_assay_fraction=mixing, expected_cross_fraction=expected,
                       mixing_ratio=ratio, centroid_distance=distance,
                       scaled_centroid_distance=scaled, alignment_status=status)
        rows.append(row)
    return pd.DataFrame(rows)


def gradient_agreement(z, obs, scores, min_cells=50, k=30):
    """Cross-assay neighbor score prediction within CD14/donor; no paired-cell claim."""
    rows = []
    for donor, ids in obs.groupby('DonorID', observed=True).indices.items():
        ids = ids[obs.iloc[ids].cell_type_harmonized.eq('CD14 monocytes').to_numpy()]
        for direction in ['CITE','Multiome']:
            query = ids[obs.iloc[ids].assay.eq(direction).to_numpy()]
            ref = ids[~obs.iloc[ids].assay.eq(direction).to_numpy()]
            for module in scores.columns:
                row = dict(donor=donor, query_assay=direction, module=module, n_query=len(query), n_reference=len(ref),
                           spearman=np.nan, shuffled_spearman=np.nan, status='insufficient cells/variation')
                if min(len(query), len(ref)) >= min_cells:
                    ni = NearestNeighbors(n_neighbors=min(k,len(ref)), n_jobs=-1).fit(z[ref]).kneighbors(z[query],return_distance=False)
                    truth = scores.iloc[query][module].to_numpy()
                    reference = scores.iloc[ref][module].to_numpy()
                    pred = reference[ni].mean(axis=1)
                    if np.isfinite(truth).all() and np.isfinite(pred).all() and np.std(truth)>1e-10 and np.std(pred)>1e-10:
                        shuffled = np.random.default_rng(42).permutation(reference)[ni].mean(axis=1)
                        row.update(spearman=float(spearmanr(truth,pred).statistic),
                                   shuffled_spearman=float(spearmanr(truth,shuffled).statistic) if np.std(shuffled)>1e-10 else np.nan,
                                   status='tested; descriptive, not independent validation')
                rows.append(row)
    return pd.DataFrame(rows)


def bridge_decision(comparison, type_table, donor_table, gradients):
    """Predeclared exploratory tolerances; retain mixed evidence, no automatic approval."""
    rows=[]
    def add(name, result, status):
        rows.append(dict(criterion=name,result=result,status=status))
    c=comparison.set_index('metric')
    mix=float(c.loc['opposite_assay_fraction','integrated']-c.loc['opposite_assay_fraction','uncorrected'])
    add('Global assay mixing',f'Change {mix:.3f}; check within-type composition-adjusted ratios', 'supported' if mix>=.05 else 'partially supported' if mix>=-.02 else 'concern')
    purity=float(c.loc['cell_type_purity','integrated']-c.loc['cell_type_purity','uncorrected'])
    sil=float(c.loc['cell_type_silhouette','integrated']-c.loc['cell_type_silhouette','uncorrected'])
    add('Cell identity preservation',f'Purity change {purity:.3f}; silhouette change {sil:.3f}', 'supported' if purity>=-.02 and sil>=-.05 else 'concern')
    tested=type_table[type_table.sufficient_cells]
    concerns=tested[tested.alignment_status=='concern'].cell_type_harmonized.tolist()
    add('Population alignment',f'Concern populations: {concerns}; untestable: {len(type_table)-len(tested)}', 'concern' if concerns else 'supported' if len(tested)==len(type_table) else 'partially supported')
    valid=donor_table[donor_table.sufficient_cells]
    median=valid.scaled_centroid_distance.median()
    rna=valid.rna_profile_pearson.median()
    add('Donor/type agreement',f'{len(valid)}/{len(donor_table)} groups testable; median scaled distance {median:.3f}; RNA correlation {rna:.3f}', 'supported' if len(valid) and median<=.5 and rna>=.5 else 'partially supported' if len(valid) else 'concern')
    for module, part in gradients.groupby('module'):
        values=part.spearman_integrated.dropna()
        delta=part.spearman_integrated-part.spearman_uncorrected
        add('CD14 gradient: '+module,f'{len(values)} directional donor tests; median rho {values.median():.3f}; median change {delta.median():.3f}', 'supported' if len(values) and values.median()>=.2 and delta.median()>=-.05 else 'partially supported' if len(values) and values.median()>0 else 'concern')
    return pd.DataFrame(rows)
