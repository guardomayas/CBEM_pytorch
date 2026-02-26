import h5py
import numpy as np

def _is_ref_dtype(dset):
    return h5py.check_dtype(ref=dset.dtype) is not None

def _maybe_squeeze_scalar(x):
    if isinstance(x, np.ndarray) and x.shape == (1, 1):
        return x[0, 0]
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x

def _read_matlab_string(arr):
    # MATLAB v7.3 often stores strings as uint16 codes
    if arr.dtype == np.uint16:
        chars = arr.flatten()
        return "".join(chr(c) for c in chars if c != 0)
    # bytes/strings
    if arr.dtype.kind in ("S", "U"):
        return "".join(arr.astype(str).flatten())
    return arr

def _reshape_list(flat_items, shape):
    """Reshape flat list into nested python lists with given shape (MATLAB column-major)."""
    # MATLAB uses column-major ordering; h5py reads in row-major.
    # In practice for 1xN or Nx1 cells it usually doesn't matter, but we preserve shape anyway.
    if len(shape) == 0:
        return flat_items[0]
    if len(shape) == 1:
        return flat_items
    # build nested lists
    out = []
    idx = 0
    nrows, ncols = shape[0], shape[1]
    # h5py gives shape as (rows, cols) typically
    for r in range(nrows):
        row = []
        for c in range(ncols):
            row.append(flat_items[idx])
            idx += 1
        out.append(row)
    return out

def _read_item(f, obj):
    # resolve references
    if isinstance(obj, h5py.Reference):
        if not obj:  # null ref
            return None
        obj = f[obj]

    # group -> struct-like
    if isinstance(obj, h5py.Group):
        out = {}
        for k in obj.keys():
            out[k] = _read_item(f, obj[k])
        return out

    # dataset
    dset = obj

    # reference dataset -> cell array or struct array
    if _is_ref_dtype(dset):
        refs = dset[()]
        # scalar ref
        if refs.shape == ():
            return _read_item(f, refs)

        flat = refs.reshape(-1)
        items = [_read_item(f, r) for r in flat]
        # return nested python lists instead of np.array reshape (avoids broadcasting issues)
        return _reshape_list(items, refs.shape)

    # numeric / char
    arr = dset[()]
    if isinstance(arr, np.ndarray) and arr.dtype == np.uint16:
        return _read_matlab_string(arr)
    return _maybe_squeeze_scalar(np.array(arr))

def load_mat_v73(filename, varname):
    with h5py.File(filename, "r") as f:
        return _read_item(f, f[varname])
    
def flatten_cell(cell):
    # cell is nested list; flatten recursively, keeping non-lists as leaves
    out = []
    def rec(x):
        if isinstance(x, list):
            for xi in x:
                rec(xi)
        else:
            out.append(x)
    rec(cell)
    return out