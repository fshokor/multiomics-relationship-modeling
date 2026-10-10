"""Execute the capture investigation, preserving completed cell outputs on error."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
from pathlib import Path
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
path=root/'notebooks/nb14b_capture_structure_investigation.ipynb'
nb=nbformat.read(path,4)
def progress(cell,cell_index,**kwargs):
    print(f'Cell {cell_index}: {cell.source.splitlines()[0][:100]}',flush=True)
client=NotebookClient(nb,timeout=3600,resources={'metadata':{'path':str(root)}},on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Saved executed investigation',flush=True)
