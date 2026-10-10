"""Read-only review of saved nb13 coordinates; never load counts or train."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from anndata.io import read_elem
root=Path(__file__).resolve().parents[1]/'results/joint_integration/nb13_run01'
with h5py.File(root/'model_input.h5ad','r') as f:
    obs=read_elem(f['obs'])
    z=read_elem(f['obsm']['X_joint'])
ev=pd.read_csv(root/'evaluation_cells.csv',index_col='cell_id',dtype={'DonorID':str})
ids=obs.index.get_indexer(ev.index)
assert (ids>=0).all() and ev.index.is_unique
z=z[ids]
assert np.isfinite(z).all()
cite=ev.assay.eq('CITE').to_numpy();c=ev.loc[cite].copy();zc=z[cite]
rhos=np.array([spearmanr(zc[:,j],c.protein_counts).statistic for j in range(zc.shape[1])])
j=int(np.nanargmax(np.abs(rhos)));c['coordinate']=zc[:,j]
rows=[]
for fields in [['DonorID'],['cell_type_harmonized'],['DonorID','cell_type_harmonized']]:
    for key,g in c.groupby(fields,observed=True):
        key=key if isinstance(key,tuple) else (key,)
        if len(g)<50:continue
        rows.append(dict(grouping='+'.join(fields),group=' | '.join(map(str,key)),n_cells=len(g),
            spearman=float(spearmanr(g.coordinate,g.protein_counts).statistic)))
table=pd.DataFrame(rows)
out=root/'review';out.mkdir(exist_ok=True)
table.to_csv(out/'protein_depth_within_groups.csv',index=False)
# Global ranks residualized on donor x cell-type group intercepts: descriptive adjustment.
c['depth_rank']=rankdata(c.protein_counts);c['coordinate_rank']=rankdata(c.coordinate)
fields=['DonorID','cell_type_harmonized'];res=c[['depth_rank','coordinate_rank']]-c.groupby(fields,observed=True)[['depth_rank','coordinate_rank']].transform('mean')
adjusted=float(np.corrcoef(res.to_numpy().T)[0,1])
metrics=pd.read_csv(root/'joint_cell_metrics.csv',index_col='cell_id')
c['opposite_assay_fraction']=metrics.loc[c.index,'opposite_assay_fraction']
mix=[]
for key,g in c.groupby(fields,observed=True):
    if len(g)<50:continue
    rho=spearmanr(g.protein_counts,g.opposite_assay_fraction).statistic if g.opposite_assay_fraction.nunique()>1 else np.nan
    mix.append(dict(donor=key[0],cell_type=key[1],n_cells=len(g),depth_vs_mixing_spearman=rho,mean_cross_assay=float(g.opposite_assay_fraction.mean())))
pd.DataFrame(mix).to_csv(out/'protein_depth_vs_mixing.csv',index=False)
summary=dict(n_evaluation_cells=len(ev),n_cite=len(c),coordinate_zero_based=j,global_spearman=float(rhos[j]),
    donor_type_rank_residual_correlation=adjusted,group_summaries={})
for grouping,g in table.groupby('grouping'):
    summary['group_summaries'][grouping]=dict(n_groups=len(g),median_rho=float(g.spearman.median()),min_rho=float(g.spearman.min()),max_rho=float(g.spearman.max()),n_above_abs_08=int((g.spearman.abs()>.8).sum()))
summary['depth_mixing']=dict(n_groups=len(mix),median_rho=float(pd.DataFrame(mix).depth_vs_mixing_spearman.median()))
(out/'depth_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('Donors:',table[table.grouping.eq('DonorID')].to_string(index=False))
