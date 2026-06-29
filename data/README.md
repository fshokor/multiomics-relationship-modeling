# Data

`data/benchmark/` is git-ignored. Place or symlink your h5ad files here:

```
data/benchmark/
├── GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad      (~5 GB)
└── GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad  (~5 GB)
```

To symlink from an existing location (recommended if files are large):
```bash
mkdir -p data/benchmark
ln -s /path/to/your/existing/files/cite.h5ad data/benchmark/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
ln -s /path/to/your/existing/files/multiome.h5ad data/benchmark/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
```

To download fresh:
```bash
python scripts/download_benchmark.py
```
