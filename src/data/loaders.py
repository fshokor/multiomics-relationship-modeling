"""
loaders.py
----------
Load and align paired h5ad modalities from the NeurIPS 2021 benchmark.
"""
import scanpy as sc
import anndata as ad
import warnings
warnings.filterwarnings("ignore")


def load_cite(cite_path: str) -> ad.AnnData:
    """
    Load the combined CITE-seq h5ad (RNA + ADT in one object).
    The benchmark file stores both modalities in a single AnnData:
      adata.X         → RNA counts
      adata.obsm['protein_expression'] → ADT (140 proteins)
    """
    adata = sc.read_h5ad(cite_path)
    return adata


def load_multiome(multiome_path: str) -> ad.AnnData:
    """
    Load the combined Multiome h5ad (RNA + ATAC in one object).
      adata.X         → RNA counts
      adata.obsm['atac'] or separate layers → ATAC peaks
    Check adata.obsm.keys() after loading to confirm the key name.
    """
    adata = sc.read_h5ad(multiome_path)
    return adata


def check_obs_alignment(adata_a: ad.AnnData, adata_b: ad.AnnData) -> bool:
    """Confirm cell barcodes are identical and in the same order."""
    match = list(adata_a.obs_names) == list(adata_b.obs_names)
    n_common = len(set(adata_a.obs_names) & set(adata_b.obs_names))
    print(f"  Barcodes identical and ordered: {match}")
    print(f"  Common barcodes: {n_common} / {len(adata_a.obs_names)}")
    return match
