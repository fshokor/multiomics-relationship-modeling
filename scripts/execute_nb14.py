"""Execute only nb14, saving completed cells even on failure."""
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMBA_NUM_THREADS', '4')
os.environ.setdefault('MPLBACKEND', 'Agg')
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root/'notebooks/nb14_joint_model_diagnostics.ipynb'
nb = nbformat.read(path, as_version=4)

def progress(cell, cell_index, **kwargs):
    print(f'Cell {cell_index}: {cell.source.splitlines()[0][:100]}', flush=True)

client = NotebookClient(nb, timeout=3600, resources={'metadata': {'path': str(root)}},
                        on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print('Saved executed nb14', flush=True)
