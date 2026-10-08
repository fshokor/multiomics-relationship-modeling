"""RNA integration figures with fixed colors, explicit scales and sampled scatter."""
import numpy as np
import matplotlib.pyplot as plt
from src.single_donor_plots import save


def make_figures(data,scores,comparison,types,donors,gradients,criteria,output):
    folder=output/'figures'
    folder.mkdir(exist_ok=True)
    names=[]
    def finish(fig,name):
        save(fig,folder/name)
        names.append(name+'.png')
    counts=data.obs.groupby(['cell_type_harmonized','assay'],observed=True).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(10,max(5,len(counts)*.35)))
    counts.plot.barh(ax=ax)
    ax.set_xlabel('Cells in the shared RNA cohort')
    finish(fig,'cell_counts')
    ids=np.random.default_rng(42).permutation(data.n_obs)[:min(50000,data.n_obs)]
    for field in ['cell_type_harmonized','assay','DonorID','Site']:
        categories=sorted(data.obs[field].astype(str).unique())
        colors=plt.get_cmap('tab20',len(categories))
        fig,axes=plt.subplots(1,2,figsize=(16,6))
        for ax,space in zip(axes,['uncorrected','integrated']):
            xy=data.obsm['X_umap_'+space]
            for i,label in enumerate(categories):
                chosen=ids[data.obs.iloc[ids][field].astype(str).eq(label).to_numpy()]
                ax.scatter(xy[chosen,0],xy[chosen,1],s=2,alpha=.5,color=colors(i),label=label,rasterized=True)
            ax.set_title(space+' RNA — '+field)
            ax.set_xticks([]); ax.set_yticks([])
        axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),markerscale=3,fontsize=7)
        finish(fig,'umap_'+field)
    metrics=['cell_type_purity','cell_type_knn_label_agreement','cell_type_silhouette','opposite_assay_fraction','assay_entropy','donor_entropy','site_entropy']
    view=comparison.set_index('metric').loc[metrics,['uncorrected','integrated']]
    fig,ax=plt.subplots(figsize=(11,5))
    view.plot.barh(ax=ax)
    ax.set_title('Fixed evaluation cells; donor/site mixing is diagnostic only')
    finish(fig,'metric_comparison')
    for value,name in [('mixing_ratio','celltype_assay_mixing'),('scaled_centroid_distance','celltype_centroid_distance')]:
        view=types.pivot(index='cell_type_harmonized',columns='space',values=value)
        fig,ax=plt.subplots(figsize=(10,max(5,len(view)*.35)))
        view.plot.barh(ax=ax)
        ax.set_title('Composition-adjusted assay mixing ratio' if value=='mixing_ratio' else 'Centroid distance / within-assay RMS radius')
        ax.set_xlabel('Missing bars mean insufficient evaluation cells')
        finish(fig,name)
    xy=data.obsm['X_umap_integrated']
    for number,module in enumerate(scores.columns,1):
        values=scores[module].to_numpy()
        finite=np.isfinite(values)
        fig,ax=plt.subplots(figsize=(8,6))
        selected=ids[finite[ids]]
        if len(selected):
            low,high=np.nanpercentile(values,[1,99])
            artist=ax.scatter(xy[selected,0],xy[selected,1],c=values[selected],s=2,cmap='viridis',vmin=low,vmax=high,rasterized=True)
            fig.colorbar(artist,ax=ax,label='Expression-derived RNA score (1–99% color limits)')
        ax.set_title(module); ax.set_xticks([]); ax.set_yticks([])
        finish(fig,f'module_{number}_umap')
        for field in ['assay','DonorID','cell_type_harmonized']:
            # Donor distributions are split by assay within CD14; other panels use all cells.
            frame=data.obs.copy()
            frame['score']=values
            if field=='DonorID':
                frame=frame[frame.cell_type_harmonized=='CD14 monocytes']
                frame['group']=frame.DonorID.astype(str)+' | '+frame.assay.astype(str)
            else:
                frame['group']=frame[field].astype(str)
            groups=[(name,p.score.dropna().to_numpy()) for name,p in frame.groupby('group',observed=True) if p.score.notna().any()]
            fig,ax=plt.subplots(figsize=(12,max(4,len(groups)*.32)))
            if groups:
                ax.boxplot([v for _,v in groups],vert=False,showfliers=False,labels=[n for n,_ in groups])
            ax.set_xlabel('Expression-derived module score; boxes describe cells, not donor uncertainty')
            ax.set_title(module+(' — CD14 donor × assay' if field=='DonorID' else ' — '+field))
            finish(fig,f'module_{number}_distribution_{field}')
    return names
