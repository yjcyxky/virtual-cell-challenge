"""Keep raw score inputs in read-only memory maps through the official API.

Our score wrapper still validates the original files and builds run metadata
from those paths. Only its call to compute_metrics receives the equivalent
AnnData matrices. No official function, normalization or score is replaced.
"""
from contextlib import contextmanager, ExitStack
import gc
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
from scipy import sparse

from vcc_task.bounded_bundle import mapped_reference


@contextmanager
def mapped_score_input(path, directory):
    """Keep the source dtypes, as this pinned AnnData's to_memory does."""
    directory = Path(directory)
    with h5py.File(path, 'r') as handle:
        index_dtypes = {key: handle['X/' + key].dtype for key in ('indices', 'indptr')}
    with mapped_reference(path, directory) as source:
        matrix = sparse.csr_matrix(source.shape, dtype=source.X.dtype)
        matrix.data = source.X.data
        for key in ('indices', 'indptr'):
            original = getattr(source.X, key)
            index_dtype = index_dtypes[key]
            if original.dtype == np.dtype(index_dtype):
                mapped = original
            else:
                target = directory / (key + '-native.npy')
                output = np.lib.format.open_memmap(target, mode='w+', dtype=index_dtype,
                                                   shape=original.shape)
                for start in range(0, len(original), 1_048_576):
                    output[start:start + 1_048_576] = original[start:start + 1_048_576]
                output.flush()
                del output
                mapped = np.load(target, mmap_mode='r')
            setattr(matrix, key, mapped)
        matrix.has_sorted_indices = source.X.has_sorted_indices
        matrix.has_canonical_format = source.X.has_canonical_format
        value = ad.AnnData(matrix, obs=source.obs.copy(), var=source.var.copy())
        try:
            yield value
        finally:
            del value, matrix
            gc.collect()


def memory_score_frozen(*args, **kwargs):
    """Call our unchanged scoring wrapper with memory-mapped raw inputs."""
    import vcc_task.frozen_scoring as scoring

    original = scoring.compute_metrics

    def compute(prediction, real, *, config, **options):
        if not isinstance(prediction, (str, Path)) or not isinstance(real, (str, Path)):
            raise TypeError('Registered frozen scores require file inputs')
        directory = Path(config.outdir) / 'decoded-score-cache'
        with ExitStack() as stack:
            mapped = {}
            inputs = []
            for side, path in [('prediction', prediction), ('real', real)]:
                path = Path(path).resolve()
                if path not in mapped:
                    mapped[path] = stack.enter_context(mapped_score_input(path, directory / side))
                inputs.append(mapped[path])
            print('MAPPED OFFICIAL SCORE', inputs[0].shape, inputs[1].shape, flush=True)
            result = original(*inputs, config=config, **options)
            del inputs, mapped
            gc.collect()
        return result

    # The alias belongs to our single-threaded adapter, not the official package.
    scoring.compute_metrics = compute
    try:
        return scoring.score_frozen(*args, **kwargs)
    finally:
        scoring.compute_metrics = original
