"""Capture-structure investigation following nb14; no generative model training.

All learned perturbations are fitted outside the held-out donor. Their transformed
folds are evaluated independently, never concatenated into a purported shared space.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from src.rna_integration_metrics import knn_indices, group_alignment
from src.joint_model_diagnostics import correlation
from src.single_donor_io import write_json, sha256

STRATA = ['DonorID', 'Site', 'cell_type_harmonized']
MIN_CELLS = 50


def matched_groups(obs, minimum=MIN_CELLS):
    counts = obs.groupby(STRATA+['assay'], observed=True).size().unstack(fill_value=0)
    counts = counts.reindex(columns=['CITE', 'Multiome'], fill_value=0)
    counts['eligible'] = counts.min(axis=1).ge(minimum)
    keys = pd.MultiIndex.from_frame(obs[STRATA])
    mask = keys.isin(counts.index[counts.eligible])
    return mask, counts.reset_index()


def balanced_indices(obs, seed, cap=250):
    """Exact equal assay counts inside eligible donor/site/population strata."""
    rng = np.random.default_rng(seed)
    chosen = []
    for _, ids in obs.groupby(STRATA, observed=True).indices.items():
        a = ids[obs.iloc[ids].assay.eq('CITE').to_numpy()]
        b = ids[obs.iloc[ids].assay.eq('Multiome').to_numpy()]
        n = min(len(a), len(b), cap)
        if n >= MIN_CELLS:
            chosen.extend(rng.choice(a, n, replace=False))
            chosen.extend(rng.choice(b, n, replace=False))
    return np.sort(chosen)


def equal_stratum_weights(obs):
    """Each donor/site/type/assay receives equal total training weight."""
    counts = obs.groupby(STRATA+['assay'], observed=True).assay.transform('size').to_numpy(float)
    w = 1/counts
    return w/w.mean()


def fit_capture_classifier(z, obs):
    scaler = StandardScaler().fit(z)
    model = LogisticRegression(C=1, max_iter=3000)
    model.fit(scaler.transform(z), obs.assay.eq('CITE').astype(int),
              sample_weight=equal_stratum_weights(obs))
    return scaler, model


def classifier_metrics(scaler, model, z, obs):
    y = obs.assay.eq('CITE').astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        return dict(auroc=np.nan, balanced_accuracy=np.nan, orientation_free_auroc_descriptive=np.nan)
    p = model.predict_proba(scaler.transform(z))[:, 1]
    auc = roc_auc_score(y, p)
    # Post-hoc orientation-free score is descriptive, not held-out predictive accuracy.
    return dict(auroc=auc, balanced_accuracy=balanced_accuracy_score(y, p >= .5),
                orientation_free_auroc_descriptive=max(auc, 1-auc))


def project_out(z, basis):
    if basis.size == 0:
        return np.asarray(z, float).copy()
    return z-(z@basis.T)@basis


def learn_capture_basis(z, obs, n_directions=3):
    """Sequential classifier normals in raw Euclidean units; training data only."""
    basis = np.empty((0, z.shape[1]))
    for _ in range(n_directions):
        projected = project_out(z, basis)
        scaler, model = fit_capture_classifier(projected, obs)
        direction = model.coef_[0]/scaler.scale_
        if len(basis):
            direction -= (direction@basis.T)@basis
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            break
        basis = np.vstack([basis, direction/norm])
    return basis


def simple_metrics(z, obs):
    nn = knn_indices(z)
    types, assay = obs.cell_type_harmonized.to_numpy(), obs.assay.to_numpy()
    groups = group_alignment(z, obs, ['Site', 'cell_type_harmonized'])
    valid = groups[groups.sufficient_cells]
    return dict(purity=float((types[nn] == types[:, None]).mean()),
                mixing=float((assay[nn] != assay[:, None]).mean()),
                median_group_mixing_ratio=valid.mixing_ratio.median(),
                concern_groups=int(valid.alignment_status.eq('concern').sum()),
                evaluable_groups=len(valid)), groups


def conditional_capture_effects(z, obs):
    """Equal stratum/assay weighted, group-intercept-adjusted partial R².

    Common capture slope after stratum intercepts vs a separate slope per stratum.
    Their difference measures heterogeneous associations, not causal attribution.
    """
    mask, _ = matched_groups(obs)
    obs = obs.loc[mask].reset_index(drop=True)
    z = z[mask]
    w = equal_stratum_weights(obs)
    y = obs.assay.eq('CITE').to_numpy(float)
    zr, yr = z.copy(), y.copy()
    individual_ss = np.zeros(z.shape[1])
    offsets = []
    for key, ids in obs.groupby(STRATA, observed=True).indices.items():
        a, b = ids[y[ids] == 1], ids[y[ids] == 0]
        delta = z[a].mean(axis=0)-z[b].mean(axis=0)
        mean = .5*(z[a].mean(axis=0)+z[b].mean(axis=0))
        zr[ids] -= mean
        yr[ids] -= .5
        individual_ss += w[ids].sum()*.25*delta**2
        offsets.append(dict(zip(STRATA, key), **{f'z{j}': d for j, d in enumerate(delta)}))
    denom = np.sum(w*yr**2)
    beta = np.sum(w[:, None]*zr*yr[:, None], axis=0)/denom
    ss = np.sum(w[:, None]*zr**2, axis=0)
    common_r2 = np.divide(beta**2*denom, ss, out=np.full_like(ss, np.nan), where=ss > 0)
    varying_r2 = np.divide(individual_ss, ss, out=np.full_like(ss, np.nan), where=ss > 0)
    table = pd.DataFrame(dict(dimension=np.arange(z.shape[1]), common_capture_beta=beta,
                             common_capture_partial_r2=common_r2, stratum_specific_capture_partial_r2=varying_r2,
                             residual_variance=ss/w.sum()))
    return table, pd.DataFrame(offsets)


def held_donor_predictability(spaces, obs):
    rows = []
    mask, _ = matched_groups(obs)
    for name, z in spaces.items():
        for donor in sorted(obs.DonorID.unique()):
            train = mask & obs.DonorID.ne(donor).to_numpy()
            test = mask & obs.DonorID.eq(donor).to_numpy()
            if min(train.sum(), test.sum()) < 100:
                continue
            scaler, model = fit_capture_classifier(z[train], obs.loc[train])
            rows.append(dict(space=name, scope='all matched populations', donor=donor, n_test=int(test.sum()),
                             **classifier_metrics(scaler, model, z[test], obs.loc[test])))
            for ct in sorted(obs.loc[test, 'cell_type_harmonized'].unique()):
                tr = train & obs.cell_type_harmonized.eq(ct).to_numpy()
                te = test & obs.cell_type_harmonized.eq(ct).to_numpy()
                if min(tr.sum(), te.sum()) < 100 or obs.loc[tr].DonorID.nunique() < 2:
                    continue
                scaler, model = fit_capture_classifier(z[tr], obs.loc[tr])
                rows.append(dict(space=name, scope=ct, donor=donor, n_test=int(te.sum()),
                                 **classifier_metrics(scaler, model, z[te], obs.loc[te])))
    return pd.DataFrame(rows)


def module_gradients(z, obs, scores, permutations=100):
    """Cross-assay CD14 transfer within the same donor/site, repeated nulls."""
    from sklearn.neighbors import NearestNeighbors
    rows = []
    for key, ids in obs.groupby(['DonorID', 'Site'], observed=True).indices.items():
        ids = ids[obs.iloc[ids].cell_type_harmonized.eq('CD14 monocytes').to_numpy()]
        for assay in ['CITE', 'Multiome']:
            query = ids[obs.iloc[ids].assay.eq(assay).to_numpy()]
            ref = ids[obs.iloc[ids].assay.ne(assay).to_numpy()]
            if min(len(query), len(ref)) < MIN_CELLS:
                continue
            nn = NearestNeighbors(n_neighbors=min(30, len(ref))).fit(z[ref]).kneighbors(z[query], return_distance=False)
            local_nn = knn_indices(z[query])
            for module in scores:
                truth, reference = scores.iloc[query][module].to_numpy(), scores.iloc[ref][module].to_numpy()
                actual = correlation(truth, reference[nn].mean(axis=1))
                rng = np.random.default_rng(42)
                null = np.array([correlation(truth, rng.permutation(reference)[nn].mean(axis=1))
                                 for _ in range(permutations)])
                rows.append(dict(donor=key[0], site=key[1], query_assay=assay, module=module,
                                 n_query=len(query), n_reference=len(ref), cross_rho=actual,
                                 local_rho=correlation(truth, truth[local_nn].mean(axis=1)),
                                 null_median=np.nanmedian(null), null_q95=np.nanquantile(null, .95),
                                 excess_over_null=actual-np.nanmedian(null),
                                 above_null_q95=bool(actual > np.nanquantile(null, .95))))
    return pd.DataFrame(rows)


def projection_experiment(z, obs, scores, output):
    rows, grouped, gradients, directions = [], [], [], []
    eligible, _ = matched_groups(obs)
    for fold, donor in enumerate(sorted(obs.DonorID.unique())):
        print('Held-donor projection:', donor, flush=True)
        train = eligible & obs.DonorID.ne(donor).to_numpy()
        # Evaluate all cells in held-out donor for purity; group metrics require matched sites.
        test = obs.DonorID.eq(donor).to_numpy()
        basis = learn_capture_basis(z[train], obs.loc[train])
        rng = np.random.default_rng(42+fold)
        random_basis = np.linalg.qr(rng.normal(size=(z.shape[1], 3)))[0].T
        arms = {'original': np.empty((0, z.shape[1])), 'remove_capture_rank1': basis[:1],
                'remove_capture_rank3': basis, 'remove_random_rank3': random_basis}
        for arm, q in arms.items():
            ztest = project_out(z[test], q)
            metrics, groups = simple_metrics(ztest, obs.loc[test])
            centered = z[test]-z[test].mean(axis=0)
            within = z[test].copy()
            for _, ids in obs.loc[test].groupby(['Site','cell_type_harmonized'], observed=True).indices.items():
                within[ids] -= within[ids].mean(axis=0)
            metrics['removed_total_variance_fraction'] = float(np.sum((centered@q.T)**2)/np.sum(centered**2))
            metrics['removed_within_stratum_variance_fraction'] = float(np.sum((within@q.T)**2)/np.sum(within**2))
            # Refit the classifier on projected training data, then test residual separability.
            scaler, model = fit_capture_classifier(project_out(z[train], q), obs.loc[train])
            et = eligible[test]
            prediction = classifier_metrics(scaler, model, ztest[et], obs.loc[test].iloc[np.flatnonzero(et)])
            rows.append(dict(donor=donor, arm=arm, n_test=int(test.sum()), **metrics, **prediction))
            grouped.append(groups.assign(donor=donor, arm=arm))
            gradients.append(module_gradients(ztest, obs.loc[test], scores.loc[test]).assign(arm=arm))
            for j, direction in enumerate(q):
                directions.append(dict(donor=donor, arm=arm, direction=j,
                                       **{f'z{k}': v for k, v in enumerate(direction)}))
    return pd.DataFrame(rows), pd.concat(grouped), pd.concat(gradients), pd.DataFrame(directions)


def load_inputs(project):
    import h5py
    from anndata.io import read_elem, sparse_dataset
    project = Path(project)
    root = project/'results/joint_model_diagnostics'
    obs = pd.read_csv(root/'baseline/cell_metadata.csv', index_col=0, dtype={'DonorID': str, 'Site': str})
    spaces = {}
    for name, path in [('joint', 'baseline/joint_full_latent.npz'),
                        ('rna_994', 'rna_panel/rna_994_latent.npz'), ('rna_broad', 'rna_panel/rna_broad_latent.npz')]:
        saved = np.load(root/path)
        if not np.array_equal(saved['cell_id'], obs.index.to_numpy(str)):
            raise ValueError('Saved latent ordering mismatch: '+name)
        spaces[name] = saved['z']
    path = project/'results/shared_rna_integration/run03/shared_rna.h5ad'
    with h5py.File(path, 'r') as f:
        source_obs, var = read_elem(f['obs']), read_elem(f['var'])
        ids = source_obs.index.get_indexer(obs.index)
        if (ids < 0).any():
            raise ValueError('Missing RNA cells')
        ds = sparse_dataset(f['layers/counts'])
        counts = sparse.vstack([ds[ids[i:i+1024], :] for i in range(0, len(ids), 1024)], format='csr')
        for col in ['cell_type_original', 'full_gex_counts', 'shared_gex_counts']:
            obs[col] = source_obs.iloc[ids][col].to_numpy()
    obs['shared_detected_genes'] = np.asarray((counts > 0).sum(axis=1)).ravel()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    obs['shared_mito_fraction'] = np.asarray(counts[:,var.index.str.upper().str.startswith('MT-')].sum(axis=1)).ravel()/totals
    obs['shared_ribosomal_fraction'] = np.asarray(counts[:,var.index.str.match(r'^RP[SL]\d')].sum(axis=1)).ravel()/totals
    scores = pd.read_csv(root/'module_preservation/scores.csv', index_col=0).reindex(obs.index)
    if scores.isna().any().any():
        raise ValueError('Full module scores are missing')
    module_status = json.loads((root/'module_preservation/status.json').read_text())['status']
    if 'proxy' in module_status:
        raise ValueError('Require completed full-RNA module scores')
    return obs, spaces, counts, var.index, scores


def rna_capture_effects(counts, genes, obs):
    """Site-matched donor/type pseudobulk contrasts; donors, not cells, replicated.

    Report descriptive effect sizes/sign consistency without cell-level DE p-values.
    CPM is relative RNA abundance and cannot distinguish technical from biological
    capture effects. Equal-site averaging avoids treating sites as extra donors.
    """
    eligible, _ = matched_groups(obs)
    rows, effects, qc, expression, detection = [], [], [], [], []
    for key, ids in obs.groupby(STRATA, observed=True).indices.items():
        if not eligible[ids].all():
            continue
        vectors, expr, detect = {}, {}, {}
        for assay in ['CITE', 'Multiome']:
            ix = ids[obs.iloc[ids].assay.eq(assay).to_numpy()]
            total = np.asarray(counts[ix].sum(axis=0)).ravel().astype(float)
            expr[assay] = 1e6*total/total.sum()
            vectors[assay] = np.log2(1+expr[assay])
            detect[assay] = np.asarray((counts[ix] > 0).mean(axis=0)).ravel()
            row = dict(zip(STRATA, key), assay=assay, n_cells=len(ix))
            for field in ['full_gex_counts', 'shared_gex_counts', 'shared_detected_genes', 'shared_mito_fraction',
                          'shared_ribosomal_fraction', 'protein_counts', 'atac_counts']:
                row[field] = obs.iloc[ix][field].median()
            qc.append(row)
        effects.append(vectors['CITE']-vectors['Multiome'])
        expression.append(np.stack([expr['CITE'],expr['Multiome']]))
        detection.append(np.stack([detect['CITE'],detect['Multiome']]))
        rows.append(dict(zip(STRATA, key)))
    effects = np.asarray(effects)
    expression, detection = np.asarray(expression), np.asarray(detection)
    index = pd.DataFrame(rows)
    donor_effects, donor_rows, donor_expression, donor_detection = [], [], [], []
    for key, ids in index.groupby(['DonorID', 'cell_type_harmonized']).indices.items():
        donor_effects.append(effects[ids].mean(axis=0))
        donor_expression.append(expression[ids].mean(axis=0))
        donor_detection.append(detection[ids].mean(axis=0))
        donor_rows.append(dict(DonorID=key[0], cell_type_harmonized=key[1], n_sites=len(ids)))
    donor_effects = np.asarray(donor_effects)
    donor_expression, donor_detection = np.asarray(donor_expression), np.asarray(donor_detection)
    donor_index = pd.DataFrame(donor_rows)
    summaries = []
    for ct, ids in donor_index.groupby('cell_type_harmonized').indices.items():
        if len(ids) < 4:
            continue
        d = donor_effects[ids]
        median = np.median(d, axis=0)
        agreement = (np.sign(d) == np.sign(median)[None, :]).mean(axis=0)
        summaries.append(pd.DataFrame(dict(gene=genes, cell_type=ct, n_donors=len(ids),
                                           median_log2_CPM1_difference=median,
                                           donor_sign_agreement=agreement,
                                           absolute_median_effect=abs(median),
                                           median_CITE_CPM=np.median(donor_expression[ids,0,:],axis=0),
                                           median_Multiome_CPM=np.median(donor_expression[ids,1,:],axis=0),
                                           median_CITE_detection=np.median(donor_detection[ids,0,:],axis=0),
                                           median_Multiome_detection=np.median(donor_detection[ids,1,:],axis=0))))
    summary = pd.concat(summaries, ignore_index=True).sort_values('absolute_median_effect', ascending=False)
    # Cosines of independently estimated capture effects across donor/site strata.
    geometry = []
    for i, row in index.iterrows():
        train = index.DonorID.ne(row.DonorID).to_numpy()
        same = train & index.cell_type_harmonized.eq(row.cell_type_harmonized).to_numpy()
        for scope, mask in [('all_populations', train), ('same_population', same)]:
            if not mask.any():
                continue
            # First average sites within donor/type, then give donor/type contrasts equal weight.
            di = donor_index.DonorID.ne(row.DonorID).to_numpy()
            if scope == 'same_population':
                di &= donor_index.cell_type_harmonized.eq(row.cell_type_harmonized).to_numpy()
            ref = donor_effects[di].mean(axis=0)
            cosine = np.dot(effects[i], ref)/(np.linalg.norm(effects[i])*np.linalg.norm(ref))
            geometry.append(dict(**row.to_dict(), reference=scope, cosine=cosine))
    return summary, pd.DataFrame(qc), pd.DataFrame(geometry), donor_effects, donor_index


def paired_donor_deltas(table):
    """Equal donor weight; descriptive bootstrap intervals, no cell-level inference."""
    rows = []
    baseline = table[table.arm.eq('original')].set_index('donor')
    for arm, part in table[~table.arm.eq('original')].groupby('arm'):
        part = part.set_index('donor')
        for metric in ['purity','mixing','median_group_mixing_ratio','auroc']:
            delta = (part[metric]-baseline[metric]).dropna()
            rng = np.random.default_rng(42)
            boot = rng.choice(delta.to_numpy(), size=(2000,len(delta)), replace=True).mean(axis=1)
            rows.append(dict(arm=arm, metric=metric, n_donors=len(delta), mean_delta=delta.mean(),
                             median_delta=delta.median(), bootstrap_low=np.quantile(boot,.025),
                             bootstrap_high=np.quantile(boot,.975), positive_donors=int(delta.gt(0).sum())))
    return pd.DataFrame(rows)


def write_figures(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig, name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(run.output/'figures'/f'{name}.{ext}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4))
    for name, part in run.composition_summary.groupby('space'):
        ax.plot(part.balance_seed.astype(str), part.median_mixing_ratio, marker='o', label=name)
    ax.axhline(.5,color='grey',ls='--')
    ax.set(xlabel='-1: all matched cells; 42–44: balanced sample seed',ylabel='Median within-stratum mixing ratio',
           title='Same donor, site and population; equal capture counts')
    ax.legend(); save(fig,'01_composition_control')
    fig, ax = plt.subplots(figsize=(9,6))
    for name, part in run.predictions.groupby('space'):
        p = part.groupby('scope').auroc.median().sort_index()
        ax.plot(p,p.index,marker='o',label=name)
    ax.set(xlabel='Median held-donor assay AUROC',xlim=(.45,1.01),title='Capture predictability within populations')
    ax.legend(); save(fig,'02_population_capture_predictability')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(run.effects.dimension,run.effects.common_capture_partial_r2,marker='o',label='Common capture effect')
    ax.plot(run.effects.dimension,run.effects.stratum_specific_capture_partial_r2,marker='o',label='Stratum-specific capture effects')
    ax.set(xlabel='Joint latent coordinate (zero-based)',ylabel='Conditional partial R²',
           title='Capture association after donor/site/population intercepts')
    ax.legend(); save(fig,'03_conditional_capture_effects')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax, table, title in zip(axes,[run.offset_geometry,run.rna_geometry],['Joint centroid offsets','RNA pseudobulk contrasts']):
        for ref, p in table.groupby('reference'):
            values=p.cosine.sort_values().to_numpy()
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=ref)
        ax.set(xlabel='Cosine to other-donor reference',ylabel='Cumulative fraction',title=title)
        ax.legend(fontsize=8)
    save(fig,'04_capture_direction_generalization')
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    arms=list(run.projections.arm.unique())
    for ax,metric,title in zip(axes,['mixing','purity','auroc'],['Within-held-donor mixing','Cell-type purity','Residual assay predictability']):
        for donor,p in run.projections.groupby('donor'):
            p=p.set_index('arm').reindex(arms)
            ax.plot(range(len(arms)),p[metric],marker='o',alpha=.7,label=donor)
        ax.set_xticks(range(len(arms)),[a.replace('remove_','') for a in arms],rotation=30,ha='right')
        ax.set(title=title,ylabel=metric)
    axes[-1].legend(bbox_to_anchor=(1,1),fontsize=7,title='Held donor')
    save(fig,'05_projection_diagnostics')
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax,module in zip(axes,run.gradients.module.unique()):
        part=run.gradients[run.gradients.module.eq(module)]
        for donor,p in part.groupby('donor'):
            v=p.groupby('arm').excess_over_null.mean().reindex(arms)
            ax.plot(range(len(arms)),v,marker='o',label=donor,alpha=.7)
        ax.axhline(0,color='grey',lw=.8)
        ax.set_xticks(range(len(arms)),[a.replace('remove_','') for a in arms],rotation=30,ha='right')
        ax.set(ylabel='Cross-capture rho minus median shuffled rho',title=module)
    axes[-1].legend(bbox_to_anchor=(1,1),fontsize=7,title='Donor')
    save(fig,'06_CD14_cross_capture_gradients')
    fig, ax = plt.subplots(figsize=(10,5))
    p=run.rna_effects[run.rna_effects.cell_type.eq('CD14 monocytes') & run.rna_effects.donor_sign_agreement.ge(.75)].head(20)
    ax.barh(p.gene.iloc[::-1],p.median_log2_CPM1_difference.iloc[::-1],color='#35789a')
    ax.set(xlabel='Median donor log2(CPM+1) difference: CITE − Multiome',title='Consistent CD14 RNA capture differences (descriptive)')
    save(fig,'07_CD14_capture_genes')


def investigation_report(run):
    strict = run.composition_summary.query('space == "joint" and balance_seed == -1').iloc[0]
    balanced = run.composition_summary.query('space == "joint" and balance_seed >= 0')
    within = run.predictions.query('space == "joint" and scope != "all matched populations"')
    delta = run.deltas.set_index(['arm','metric'])
    mix = delta.loc[('remove_capture_rank3','mixing')]
    purity = delta.loc[('remove_capture_rank3','purity')]
    rand = delta.loc[('remove_random_rank3','mixing')]
    post = run.projections.query('arm == "remove_capture_rank3"').auroc.median()
    removed = run.projections.query('arm == "remove_capture_rank3"').removed_within_stratum_variance_fraction.median()
    oriented = run.projections.query('arm == "remove_capture_rank3"').orientation_free_auroc_descriptive.median()
    qc_prediction = run.predictions.query('space == "rna_QC_only" and scope != "all matched populations"').auroc.median()
    program = run.gradients.groupby(['arm','module','donor'])[['local_rho','excess_over_null']].mean()
    original = program.loc['original']
    perturbed = program.loc['remove_capture_rank3']
    change = (perturbed-original).groupby('module').mean()
    biology_ok = change.local_rho.ge(-.05).all() and change.excess_over_null.ge(-.05).all()
    learned_help = mix.mean_delta >= .02 and purity.mean_delta >= -.02 and mix.mean_delta > rand.mean_delta+.01
    consistent = run.rna_effects.query('donor_sign_agreement >= .75 and absolute_median_effect >= 1')
    f = run.offset_geometry.groupby('reference').cosine.median()
    r = run.rna_geometry.groupby('reference').cosine.median()
    result = [
        dict(question='Does donor/site/population composition explain the failure?',
             evidence=f'{int(strict.concerns)}/{int(strict.evaluable)} same-site donor/population groups remain concerning; balanced median mixing ratio {balanced.median_mixing_ratio.median():.3f}',
             conclusion='Composition is insufficient' if strict.concerns/strict.evaluable > .5 else 'Composition contributes; compare matched groups carefully'),
        dict(question='Does capture separation generalize inside populations?',
             evidence=f'{len(within)} held-donor/population tests; median AUROC {within.auroc.median():.4f}; minimum {within.auroc.min():.4f}',
             conclusion='Persistent within-population capture structure' if within.auroc.median() >= .8 else 'Heterogeneous or weak generalization'),
        dict(question='Is a small learned capture subspace sufficient?',
             evidence=f'Rank-3 mixing change {mix.mean_delta:+.4f}; random rank-3 {rand.mean_delta:+.4f}; purity {purity.mean_delta:+.4f}; refit AUROC {post:.4f}',
             conclusion='Candidate geometric contribution, not a validated correction' if learned_help else 'No sufficient global low-rank remedy'),
        dict(question='Are capture effects population dependent?',
             evidence=f'Joint offset cosine, same population {f.get("same_population",np.nan):.3f} vs all {f.get("all_populations",np.nan):.3f}; RNA {r.get("same_population",np.nan):.3f} vs {r.get("all_populations",np.nan):.3f}',
             conclusion='Descriptive heterogeneity; do not assign one mechanism to all populations'),
        dict(question='Do RNA capture differences recur across donors?',
             evidence=f'{consistent.gene.nunique()} distinct genes meet |median log2(CPM+1) difference|≥1 and ≥75% sign agreement in at least one population with ≥4 donors',
             conclusion='Relative-expression differences; technical versus biological origin remains unresolved'),
        dict(question='Are full CD14 gradients retained after rank-3 perturbation?',
             evidence='Equal-donor module changes: '+change.round(3).to_json(),
             conclusion='Passes exploratory ≤0.05 loss screen' if biology_ok else 'Biological tradeoff detected; inspect module/donor results')]
    decision = pd.DataFrame(result)
    if learned_help and biology_ok:
        next_step = ('Before nb15, test a capture-penalized model with population-conditioned alignment, using held-out donors and full RNA module gradients as acceptance criteria. '
                     'The learned projection is only evidence of a geometric contribution, not the proposed final correction.')
    else:
        next_step = ('Before nb15, prioritize a controlled comparison of population-conditioned RNA-bridge alignment against the existing joint model, '
                     'with donor-held-out evaluation and exact site/population overlap. Do not use global coordinate removal or simply increase training duration. '
                     'Audit recurring RNA capture differences and within-population subtype composition before deciding which assay-associated signals may safely be removed.')
    report = (
        '# Main findings\n\n'
        f'- Exact donor/site/population comparisons: **{int(strict.concerns)}/{int(strict.evaluable)} concern groups**; '
        f'composition-adjusted median mixing ratio **{strict.median_mixing_ratio:.3f}**. '
        f'Equal-count sampling over three seeds yields median **{balanced.median_mixing_ratio.median():.3f}**.\n'
        f'- Within-population held-donor capture prediction: median AUROC **{within.auroc.median():.4f}** across {len(within)} tests.\n'
        f'- Removing three classifier directions learned on other donors changes within-held-donor mixing by **{100*mix.mean_delta:+.2f} percentage points** '
        f'(descriptive donor-bootstrap 95% interval {100*mix.bootstrap_low:+.2f} to {100*mix.bootstrap_high:+.2f}); '
        f'purity changes **{100*purity.mean_delta:+.2f} points**. Random rank-3 removal changes mixing by {100*rand.mean_delta:+.2f} points. '
        f'Refitted capture classifier median AUROC remains **{post:.4f}**. '
        f'These directions account for only **{100*removed:.4f}%** of within-stratum latent variance (median held donor).\n'
        f'- RNA depth/detection/mitochondrial/ribosomal summaries alone give within-population held-donor median AUROC **{qc_prediction:.4f}**. '
        'These quantities can reflect both measurement and biology; they are not automatically safe regression targets.\n\n'
        '# Diagnosis\n\n'
        + ('Capture separation persists inside the same donor, site and population, so donor/site composition alone is not the explanation. '
           if strict.concerns/strict.evaluable > .5 else 'Composition controls materially alter the diagnosis; inspect the matched-group tables. ')
        + 'RNA differences between captures also recur across donors. Broad identities can remain distinct while each population retains capture-associated structure. '
        'The observations do not distinguish technical assay effects from unmeasured state/subtype selection; capture is not randomized. '
        'Strong classifier signals can occupy very low-variance directions and therefore contribute little to Euclidean neighborhood distances. '
        f'After projection, some held-donor classifier relations reverse: median orientation-free separability is {oriented:.4f}; '
        'this post-hoc statistic is descriptive, not predictive accuracy. Low or below-chance AUROC alone does not prove successful alignment. '
        'Directional heterogeneity and full-module checks must guide any proposed integration.\n\n'
        '# Recommended next step\n\n'+next_step+'\n\n'
        'Accept a future model only if same-site donor/type mixing improves, cell-type purity loses no more than 0.02, '
        'and full inflammatory/TNF gradients show no more than 0.05 loss relative to the matched baseline. '
        'These are exploratory prespecified screens, not proof of equivalence. No generative model was trained, cells were not paired, '
        'and no missing-modality prediction or molecular causality was validated.\n')
    return decision, report


class CaptureInvestigation:
    def __init__(self, project):
        self.project = Path(project)
        self.output = self.project/'results/joint_model_diagnostics/capture_structure'
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'figures').mkdir(exist_ok=True)

    def save(self, table, name):
        table.to_csv(self.output/(name+'.csv'), index=False)
        return table

    def load(self):
        write_json(self.output/'status.json', dict(state='running', joint_model_training=False))
        self.obs, self.spaces, self.counts, self.genes, self.scores = load_inputs(self.project)
        self.eligible, coverage = matched_groups(self.obs)
        self.save(coverage, 'matched_stratum_coverage')
        self.obs.to_csv(self.output/'cell_metadata.csv')
        write_json(self.output/'manifest.json', dict(seed=42, minimum_cells_per_capture=MIN_CELLS,
                   balance_seeds=[42,43,44], balance_cap=250, permutations=100,
                   source_RNA_sha256=sha256(self.project/'results/shared_rna_integration/run03/shared_rna.h5ad'),
                   source_joint_sha256=sha256(self.project/'results/joint_model_diagnostics/baseline/joint_full_latent.npz'),
                   test_policy='Held-out donor, within-site/type comparisons; folds never pooled into one corrected space',
                   n_evaluation_cells=len(self.obs), n_matched_cells=int(self.eligible.sum())))
        return coverage

    def composition(self):
        tables, summary = [], []
        for name, z in self.spaces.items():
            strict = group_alignment(z, self.obs, STRATA)
            tables.append(strict.assign(space=name, balance_seed=-1))
            for seed in [42,43,44]:
                ids = balanced_indices(self.obs, seed)
                part = group_alignment(z[ids], self.obs.iloc[ids], STRATA)
                tables.append(part.assign(space=name, balance_seed=seed))
            for seed, part in [(int(t.balance_seed.iloc[0]), t) for t in tables if t.space.iloc[0] == name]:
                valid = part[part.sufficient_cells]
                summary.append(dict(space=name, balance_seed=seed, evaluable=len(valid),
                                    concerns=int(valid.alignment_status.eq('concern').sum()),
                                    median_mixing_ratio=valid.mixing_ratio.median(),
                                    median_scaled_centroid=valid.scaled_centroid_distance.median()))
        self.group_alignment = pd.concat(tables)
        self.save(self.group_alignment, 'site_matched_alignment')
        self.composition_summary = self.save(pd.DataFrame(summary), 'composition_summary')
        return self.composition_summary

    def predict(self):
        spaces = dict(self.spaces)
        spaces['rna_QC_only'] = np.column_stack([
            np.log1p(self.obs.shared_gex_counts), np.log1p(self.obs.shared_detected_genes),
            self.obs.shared_mito_fraction, self.obs.shared_ribosomal_fraction])
        self.predictions = self.save(held_donor_predictability(spaces, self.obs), 'held_donor_predictability')
        return self.predictions.groupby(['space','scope'])[['auroc','balanced_accuracy']].agg(['median','min','count'])

    def geometry(self):
        self.effects, offsets = conditional_capture_effects(self.spaces['joint'], self.obs)
        self.save(self.effects, 'conditional_capture_effects')
        self.save(offsets, 'joint_capture_offsets')
        cols = [c for c in offsets if c.startswith('z')]
        rows = []
        for i, row in offsets.iterrows():
            for scope in ['all_populations', 'same_population']:
                ref = offsets[offsets.DonorID.ne(row.DonorID)]
                if scope == 'same_population':
                    ref = ref[ref.cell_type_harmonized.eq(row.cell_type_harmonized)]
                if not len(ref):
                    continue
                v = row[cols].to_numpy(float)
                direction = ref.groupby(['DonorID', 'cell_type_harmonized'])[cols].mean().mean().to_numpy()
                rows.append(dict(DonorID=row.DonorID, Site=row.Site, cell_type=row.cell_type_harmonized,
                                 reference=scope, cosine=np.dot(v,direction)/(np.linalg.norm(v)*np.linalg.norm(direction))))
        self.offset_geometry = self.save(pd.DataFrame(rows), 'joint_offset_generalization')
        return self.effects.sort_values('stratum_specific_capture_partial_r2', ascending=False)

    def perturb(self):
        self.projections, groups, self.gradients, directions = projection_experiment(
            self.spaces['joint'], self.obs, self.scores, self.output)
        for frame, name in [(self.projections, 'held_donor_projection_metrics'), (groups, 'projection_group_metrics'),
                             (self.gradients, 'projection_module_gradients'), (directions, 'training_only_projection_directions')]:
            self.save(frame, name)
        return self.projections.groupby('arm')[['purity','mixing','median_group_mixing_ratio','auroc']].agg(['median','min','max'])

    def rna(self):
        self.rna_effects, self.qc, self.rna_geometry, donor_effects, donor_index = rna_capture_effects(self.counts, self.genes, self.obs)
        self.save(self.rna_effects, 'RNA_capture_gene_effects')
        self.save(self.qc, 'matched_stratum_QC')
        self.save(self.rna_geometry, 'RNA_capture_effect_generalization')
        self.save(donor_index, 'RNA_donor_contrast_index')
        np.savez_compressed(self.output/'RNA_donor_contrasts.npz', effects=donor_effects, genes=self.genes.to_numpy(str))
        consistent = self.rna_effects[self.rna_effects.donor_sign_agreement.ge(.75) & self.rna_effects.absolute_median_effect.ge(1)]
        self.save(consistent, 'RNA_consistent_capture_genes')
        membership = pd.read_csv(self.project/'results/joint_model_diagnostics/module_preservation/gene_membership.csv')
        program = self.rna_effects[self.rna_effects.cell_type.eq('CD14 monocytes')].merge(membership, on='gene')
        self.save(program, 'CD14_program_capture_effects')
        # Original-label composition is an audit, not a matched taxonomy claim.
        original = self.obs.groupby(STRATA+['assay','cell_type_original'], observed=True).size().rename('n_cells').reset_index()
        self.save(original, 'original_label_composition')
        return consistent.groupby('cell_type').agg(n_genes=('gene','size'), n_donors=('n_donors','first'))

    def finish(self):
        self.deltas = self.save(paired_donor_deltas(self.projections), 'paired_donor_projection_deltas')
        self.decisions, report = investigation_report(self)
        self.save(self.decisions, 'investigation_decisions')
        (self.output/'investigation_report.md').write_text(report, encoding='utf-8')
        write_figures(self)
        write_json(self.output/'status.json', dict(state='complete', no_generative_model_training=True,
                   n_cells=len(self.obs), n_genes=len(self.genes), n_matched_cells=int(self.eligible.sum()),
                   limitation='Descriptive capture-associated effects; technical vs biological causes not identified'))
        return self.deltas, self.decisions, report
