"""Memory-bounded access to benchmark counts without loading all ATAC/obsm data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def inspect_h5ad(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        for key in ("DonorID", "Site", "cell_type"):
            if key not in obs or obs[key].isna().any():
                raise ValueError(f"{path}: missing or null metadata {key}")
        if "feature_types" not in var or "counts" not in f.get("layers", {}):
            raise ValueError(f"{path}: expected feature_types and layers/counts")
        for key in ("DonorID", "Site", "cell_type"):
            obs[key] = obs[key].astype(str)
        if not obs.index.is_unique or not var.index.is_unique:
            raise ValueError("Duplicate observation/feature names require explicit resolution")
        audit = {"path": str(Path(path).resolve()), "file_bytes": Path(path).stat().st_size,
                 "file_mtime_ns": Path(path).stat().st_mtime_ns, "shape": [len(obs), len(var)],
                 "feature_types": var.feature_types.value_counts().to_dict(),
                 "layers": list(f["layers"]), "obsm": list(f.get("obsm", {})),
                 "X_encoding": str(f["X"].attrs.get("encoding-type", "unknown")),
                 "rna_source": "layers/counts restricted to GEX before normalization"}
    return obs, var, audit


def read_rna_counts(path, rows, var, chunk_size=256):
    import h5py
    from anndata.io import sparse_dataset
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0 or (np.diff(rows) <= 0).any():
        raise ValueError("Row positions must be nonempty, increasing and unique")
    cols = np.flatnonzero(var.feature_types.to_numpy() == "GEX")
    if not len(cols):
        raise ValueError("No GEX features")
    with h5py.File(path, "r") as f:
        node = f["layers/counts"]
        backed = sparse_dataset(node) if isinstance(node, h5py.Group) else node
        blocks = []
        for start in range(0, len(rows), chunk_size):
            block = backed[rows[start:start + chunk_size], :]
            blocks.append(sparse.csr_matrix(block)[:, cols])
    return sparse.vstack(blocks, format="csr"), var.index[cols].astype(str).to_numpy()


def output_dirs(root):
    dirs = {name: Path(root) / name for name in ("donor_selection", "rna_cite", "rna_multiome",
            "rna_concordance", "gsea", "atac", "protein", "multilayer")}
    for path in dirs.values():
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_rna(path, x, obs, genes):
    sparse.save_npz(path / "rna_log1p_cp10k.npz", x)
    obs.to_csv(path / "cells.csv", index_label="cell_id")
    pd.Series(genes, name="gene").to_csv(path / "genes.csv", index=False)


def load_rna(path):
    return (sparse.load_npz(path / "rna_log1p_cp10k.npz"),
            pd.read_csv(path / "cells.csv", index_col="cell_id", dtype={"DonorID": str, "Site": str}),
            pd.read_csv(path / "genes.csv").gene.to_numpy())
