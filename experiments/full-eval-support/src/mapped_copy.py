"""Experiment-local memory adapter for official anchor row copies.

The scorer still chooses the cells and computes every statistic. Only the
storage of ``real[integer_rows].copy()`` changes: an independent writable CSR
matrix is copied in row chunks to temporary memory maps. Other selections use
AnnData's normal implementation. No global NumPy or scorer functions change.
"""
from functools import partial
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import shutil
import tempfile
import weakref

import anndata as ad
import numpy as np
from scipy import sparse


def _copy_selected_rows(view_ref, source, rows, directory, chunk_rows, filename=None):
    view = view_ref()
    if view is None:
        raise RuntimeError('The row view was released before its copy')
    if filename is not None:
        return ad.AnnData.copy(view, filename=filename)
    # The frozen reference constructor has only X/obs/var. Refuse to silently
    # drop new metadata or layers if a future caller expands that contract.
    # This pinned AnnData exposes X as layers[None]; it is not an extra layer.
    if (view.raw is not None or any(key is not None for key in view.layers)
            or len(view.obsm) or len(view.varm)
            or len(view.obsp) or len(view.varp) or len(view.uns)):
        raise ValueError('Mapped anchor copy requires the registered X/obs/var-only reference')
    if not sparse.isspmatrix_csr(source):
        raise TypeError('Mapped anchor copy requires a CSR reference')
    counts = np.asarray(source.indptr[rows + 1] - source.indptr[rows], dtype=np.int64)
    nnz = int(counts.sum())
    # The decoded source has int64 indices. SciPy's native copy narrows the
    # result to int32 when its shape/nnz fit; reproduce that final copy dtype.
    if source.indptr.dtype != np.dtype('int64') or source.indices.dtype != np.dtype('int64'):
        raise ValueError('Mapped anchor copy requires the frozen int64 reference indices')
    index_dtype = np.int64 if max(nnz, len(rows), source.shape[1]) > np.iinfo(np.int32).max else np.int32
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # AnnData may participate in cycles; reclaim unreachable earlier halves
    # before allocating another pair of large temporary files.
    gc.collect()
    local = Path(tempfile.mkdtemp(prefix='anchor-copy-', dir=directory))
    arrays = []
    try:
        for key, dtype, shape in [('data', source.data.dtype, (nnz,)),
                                  ('indices', index_dtype, (nnz,)),
                                  ('indptr', index_dtype, (len(rows) + 1,))]:
            arrays.append(np.lib.format.open_memmap(local / (key + '.npy'), mode='w+',
                                                   dtype=dtype, shape=shape))
        data, indices, indptr = arrays
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        for start in range(0, len(rows), chunk_rows):
            stop = min(start + chunk_rows, len(rows))
            block = source[rows[start:stop]]
            left, right = int(indptr[start]), int(indptr[stop])
            if block.nnz != right - left:
                raise ValueError('CSR row copy changed the stored element count')
            data[left:right] = block.data
            indices[left:right] = block.indices
        for array in arrays:
            array.flush()
        matrix = sparse.csr_matrix((len(rows), source.shape[1]), dtype=source.dtype)
        matrix.data, matrix.indices, matrix.indptr = arrays
        matrix.has_sorted_indices = source.has_sorted_indices
        matrix.has_canonical_format = source.has_canonical_format
        result = ad.AnnData(matrix, obs=view.obs.copy(deep=True), var=view.var.copy(deep=True))
        # POSIX mappings remain usable if a downstream matrix outlives AnnData.
        # Only this newly-created copy's temporary directory is removed.
        weakref.finalize(result, shutil.rmtree, local, ignore_errors=True)
        print(f'MAPPED ANCHOR COPY cells={len(rows)} nnz={nnz}', flush=True)
        return result
    except BaseException:
        shutil.rmtree(local, ignore_errors=True)
        raise


class MappedCopyAnnData(ad.AnnData):
    """AnnData with bounded, independent copies for official row permutations."""

    def __init__(self, *args, copy_directory, copy_chunk_rows=2048, **kwargs):
        super().__init__(*args, **kwargs)
        self._vcc_copy_directory = Path(copy_directory)
        self._vcc_copy_chunk_rows = int(copy_chunk_rows)
        if self._vcc_copy_chunk_rows < 1:
            raise ValueError('copy_chunk_rows must be positive')

    def __getitem__(self, index):
        view = super().__getitem__(index)
        # _disjoint_halves uses a 1-D integer permutation and retains all genes.
        # Boolean/control/column selections keep the native copy behavior.
        if (isinstance(index, np.ndarray) and index.ndim == 1
                and np.issubdtype(index.dtype, np.integer)):
            rows = np.asarray(index, dtype=np.int64).copy()
            rows[rows < 0] += self.n_obs
            view.copy = partial(_copy_selected_rows, weakref.ref(view), self.X, rows,
                                self._vcc_copy_directory, self._vcc_copy_chunk_rows)
        return view


@contextmanager
def _anchor_reference(path, directory, *, original):
    with original(path, directory) as reference:
        yield MappedCopyAnnData(reference.X, obs=reference.obs.copy(), var=reference.var.copy(),
                               copy_directory=Path(directory) / 'anchor-copies')


def memory_bounded_bundle(*args, **kwargs):
    """Reuse our frozen bundle adapter with a different reference storage class.

    The reference factory belongs to our I/O adapter, not cell-eval2. Its
    temporary replacement is process-local and restored even on failure. The
    official baseline, split, scoring, anchor and bundle functions are untouched.
    This entry is used only by the single-threaded Experiment driver.
    """
    import vcc_task.bounded_bundle as adapter
    from vcc_task.common import ROOT, verified

    original = adapter.mapped_reference
    original_baseline = adapter.official_baseline
    registered = json.loads((ROOT / 'docs/research/experiment_dag.json').read_text())
    node = registered['nodes'][os.environ['VCC_RESEARCH_NODE']]
    preserved = {r['path']: r for r in node.get('evaluation_repair', {}).get('preserved_refs', [])}
    real_path = Path(args[0]).resolve()

    def baseline(real, profile, path, chunk_targets, verify_full):
        path = Path(path).resolve()
        equivalence = path.parent / 'baseline-equivalence.json'
        names = [str(p.relative_to(ROOT)) for p in (real_path, path, equivalence)]
        if all(name in preserved for name in names):
            for name in names:
                verified(preserved[name])
            record = json.loads(equivalence.read_text())
            if (record['baseline_ref'] != preserved[names[1]] or record['seed'] != 0
                    or record['chunk_targets'] != chunk_targets
                    or verify_full and not record['canonical_full_checked']):
                raise ValueError('Preserved official baseline differs from the frozen invocation')
            print('REUSED FROZEN OFFICIAL BASELINE', path, flush=True)
            return record
        return original_baseline(real, profile, path, chunk_targets, verify_full)

    adapter.mapped_reference = partial(_anchor_reference, original=original)
    adapter.official_baseline = baseline
    try:
        return adapter.bounded_bundle(*args, **kwargs)
    finally:
        adapter.mapped_reference = original
        adapter.official_baseline = original_baseline
