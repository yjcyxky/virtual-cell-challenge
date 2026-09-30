"""Append bounded CSR batches into a standard, atomically published H5AD."""
from pathlib import Path

import anndata as ad
from anndata.io import write_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .counts import validate_counts


class CountWriter:
    def __init__(self, path, var):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix('.partial.h5ad')
        ad.AnnData(sparse.csr_matrix((0, len(var)), dtype=np.int32),
                   obs=pd.DataFrame(index=pd.Index([], dtype=str)), var=var).write_h5ad(self.temporary)
        self.handle = h5py.File(self.temporary, 'r+')
        del self.handle['X']
        x = self.handle.create_group('X')
        x.attrs.update({'encoding-type': 'csr_matrix', 'encoding-version': '0.1.0', 'shape': (0, len(var))})
        self.data = x.create_dataset('data', (0,), maxshape=(None,), chunks=(262144,), dtype='i4',
                                    compression='gzip', compression_opts=1)
        self.indices = x.create_dataset('indices', (0,), maxshape=(None,), chunks=(262144,), dtype='i4',
                                       compression='gzip', compression_opts=1)
        self.ptr = x.create_dataset('indptr', data=np.array([0], dtype=np.int64), maxshape=(None,),
                                   chunks=True, compression='gzip', compression_opts=1)
        self.rows = self.nnz = 0
        self.columns = len(var)

    def append(self, matrix):
        matrix = sparse.csr_matrix(matrix)
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
        validate_counts(matrix)
        if matrix.shape[1] != self.columns or matrix.data.max(initial=0) > np.iinfo(np.int32).max:
            raise ValueError('Count store shape or integer capacity exceeded')
        stop = self.nnz + matrix.nnz
        self.data.resize((stop,)); self.indices.resize((stop,))
        self.data[self.nnz:stop] = matrix.data
        self.indices[self.nnz:stop] = matrix.indices
        self.ptr.resize((self.rows + matrix.shape[0] + 1,))
        self.ptr[self.rows+1:] = matrix.indptr[1:] + self.nnz
        self.rows += matrix.shape[0]; self.nnz = stop

    def finish(self, obs):
        if len(obs) != self.rows or obs.index.has_duplicates:
            raise ValueError('Count store observation identity mismatch')
        self.handle['X'].attrs['shape'] = (self.rows, self.columns)
        del self.handle['obs']
        write_elem(self.handle, 'obs', obs)
        self.handle.close()
        self.temporary.replace(self.path)
        return self.path


def copy_panel(bank, targets, destination, chunk=1024):
    """Materialize an exact panel, preserving every unique reference identity."""
    selected = bank.obs.target_gene.isin([*targets, 'non-targeting']).to_numpy()
    rows = np.flatnonzero(selected)
    writer = CountWriter(destination, bank.var.copy())
    for start in range(0, len(rows), chunk):
        writer.append(bank.X[rows[start:start+chunk]])
    return writer.finish(bank.obs.iloc[rows].copy())
