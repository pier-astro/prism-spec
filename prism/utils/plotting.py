import numpy as np

__all__ = ['plot_outline', 'mask_to_outline']

def _segments_to_paths(segments, close_paths=True):
    """
    Convert a set of disconnected outline segments into connected polygonal paths.

    Parameters
    ----------
    segments : ndarray, shape (n_segments, 2, 2)
        Array of line segments where each row represents a segment with two
        endpoints ``[[x0, y0], [x1, y1]]``.
    close_paths : bool, optional
        If True (default), closed loops keep their starting vertex repeated
        at the end. Open segment chains are left open so callers can detect
        malformed inputs instead of silently drawing a fabricated edge.

    Returns
    -------
    paths : list of ndarray
        A list of 2D arrays where each array contains vertices of one
        connected outline. Each vertex is a pair ``[x, y]``. Closed paths
        repeat the first vertex at the end when ``close_paths`` is True.
    """
    import numpy as _np
    from collections import defaultdict

    segs = _np.asarray(segments, dtype=float)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2):
        raise ValueError(
            "segments must have shape (n_segments, 2, 2) to form paths"
        )
    if len(segs) == 0:
        return []

    # Pixel-edge vertices lie on an integer or half-integer grid, so convert
    # them to an exact integer lattice for robust hashing and matching.
    lattice = _np.rint(2.0 * segs).astype(int)

    def _key(pt):
        return (int(pt[0]), int(pt[1]))

    point_to_segments = defaultdict(list)
    for idx, seg in enumerate(lattice):
        point_to_segments[_key(seg[0])].append(idx)
        point_to_segments[_key(seg[1])].append(idx)

    unused = set(range(len(segs)))
    paths = []

    while unused:
        i0 = unused.pop()
        seg = lattice[i0]
        start = _key(seg[0])
        end = _key(seg[1])
        path = [start, end]
        current = end

        while True:
            candidates = point_to_segments[current]
            next_idx = None
            for j in candidates:
                if j in unused:
                    next_idx = j
                    break
            if next_idx is None:
                break
            unused.remove(next_idx)
            s = lattice[next_idx]
            a = _key(s[0])
            b = _key(s[1])
            current = b if a == current else a
            path.append(current)
            if current == start:
                break

        paths.append(0.5 * _np.asarray(path, dtype=float))
    return paths

def mask_to_outline(mask):
    """
    Return pixel-exact outline segments for a 2D boolean mask.

    Parameters
    ----------
    mask : array-like of bool, shape (ny, nx)
        True values define the selected region(s).

    Returns
    -------
    segments : ndarray, shape (n_segments, 2, 2)
        Line segments in matplotlib format:
        [[[x0, y0], [x1, y1]], ...]

    Examples
    --------
    Basic example with a rectangular region:

    >>> mask = np.zeros((6, 6), dtype=bool)
    >>> mask[2:4, 1:5] = True
    >>> segments = mask_to_outline(mask)
    >>> segments.shape[1:]
    (2, 2)

    Example with two disconnected regions:

    >>> mask = np.zeros((8, 8), dtype=bool)
    >>> mask[1:3, 1:3] = True
    >>> mask[5:7, 5:7] = True
    >>> segments = mask_to_outline(mask)

    Minimal circular mask example:

    >>> ny, nx = 11, 11
    >>> y, x = np.indices((ny, nx))
    >>> cy, cx = 5, 5
    >>> r = 2.0
    >>> mask = (x - cx)**2 + (y - cy)**2 <= r**2
    >>> segments = mask_to_outline(mask)
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    y, x = np.indices(mask.shape)

    # Pad with False so borders are naturally handled
    padded = np.pad(mask, 1, mode="constant", constant_values=False)

    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    segments = []

    # Bottom edges
    m = center & ~up
    if np.any(m):
        segments.append(np.stack([
            np.stack([x[m] - 0.5, y[m] - 0.5], axis=1),
            np.stack([x[m] + 0.5, y[m] - 0.5], axis=1),
        ], axis=1))

    # Top edges
    m = center & ~down
    if np.any(m):
        segments.append(np.stack([
            np.stack([x[m] - 0.5, y[m] + 0.5], axis=1),
            np.stack([x[m] + 0.5, y[m] + 0.5], axis=1),
        ], axis=1))

    # Left edges
    m = center & ~left
    if np.any(m):
        segments.append(np.stack([
            np.stack([x[m] - 0.5, y[m] - 0.5], axis=1),
            np.stack([x[m] - 0.5, y[m] + 0.5], axis=1),
        ], axis=1))

    # Right edges
    m = center & ~right
    if np.any(m):
        segments.append(np.stack([
            np.stack([x[m] + 0.5, y[m] - 0.5], axis=1),
            np.stack([x[m] + 0.5, y[m] + 0.5], axis=1),
        ], axis=1))

    if not segments:
        return np.empty((0, 2, 2), dtype=float)

    return np.concatenate(segments, axis=0)


def plot_outline(
    outline,
    ax=None,
    color="red",
    linewidth=1.0,
    linestyle="-",
    alpha=1.0,
    label=None,
    **kwargs,
):
    """
    Plot the pixel-exact outline of a 2D boolean mask or precomputed outline.

    This function accepts either a boolean mask or an array of precomputed
    segments. For masks, the outline is computed via :func:`mask_to_outline`.
    The resulting segments are converted into connected polygonal paths and
    drawn as continuous lines. By closing each path explicitly, PDF outputs
    look smooth without visible gaps at the end of the path.

    Parameters
    ----------
    outline : array-like
        Either:
        - a 2D boolean mask of shape ``(ny, nx)`` with ``True`` values
          defining the region to outline, or
        - an array of precomputed line segments with shape
          ``(n_segments, 2, 2)``.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw. If None, the current axes is used.
    color : str, optional
        Color of the outline. Defaults to "red".
    linewidth : float, optional
        Width of the outline lines. Defaults to 1.0.
    linestyle : str, optional
        Matplotlib line style (e.g., '-', '--', etc.). Defaults to '-'.
    alpha : float, optional
        Transparency of the outline. Defaults to 1.0 (opaque).
    label : str, optional
        Label for the first outline. Subsequent outlines do not receive
        labels to avoid duplicate legend entries.
    **kwargs : dict, optional
        Additional keyword arguments passed to
        :class:`matplotlib.patches.PathPatch`.

    Returns
    -------
    patches : list of matplotlib.patches.PathPatch
        A list of PathPatch objects corresponding to each drawn path.

    Examples
    --------
    Pass a mask directly:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> mask = np.zeros((10, 10), dtype=bool)
    >>> mask[3:7, 2:8] = True
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(mask, origin="lower", cmap="gray")
    >>> plot_outline(mask, ax=ax, color="tab:blue", linewidth=2)

    Pass precomputed segments:

    >>> segments = mask_to_outline(mask)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(mask, origin="lower", cmap="gray")
    >>> plot_outline(segments, ax=ax, color="orange", linewidth=2)

    Minimal circular mask example:

    >>> ny, nx = 21, 21
    >>> y, x = np.indices((ny, nx))
    >>> cy, cx = 10, 10
    >>> r = 4.0
    >>> mask = (x - cx)**2 + (y - cy)**2 <= r**2
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(mask, origin="lower", cmap="gray")
    >>> plot_outline(mask, ax=ax, color="cyan", linewidth=2, label="circular region")
    >>> ax.scatter(cx, cy, c="red", marker="x")
    >>> ax.legend()
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if ax is None:
        ax = plt.gca()

    # Determine if input is a mask or precomputed segments
    arr = np.asarray(outline)
    if arr.ndim == 2:
        segments = mask_to_outline(arr)
    elif arr.ndim == 3 and arr.shape[1:] == (2, 2):
        segments = arr
    else:
        raise ValueError(
            "outline must be either a 2D mask or an array of segments "
            "with shape (n_segments, 2, 2)"
        )

    paths = _segments_to_paths(segments, close_paths=True)

    patches = []
    for idx, path in enumerate(paths):
        this_label = label if idx == 0 else None
        if path.shape[0] < 2:
            continue
        if not np.array_equal(path[0], path[-1]):
            raise ValueError("outline segments do not form a closed loop")
        codes = np.full(path.shape[0], Path.LINETO, dtype=np.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        patch = PathPatch(
            Path(path, codes),
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=this_label,
            joinstyle="miter",
            capstyle="butt",
            **kwargs,
        )
        ax.add_patch(patch)
        ax.update_datalim(path[:-1])
        patches.append(patch)
    if patches:
        ax.autoscale_view()
    return patches
