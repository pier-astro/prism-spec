"""Small statistical summaries and parameter-state association checks."""
import numpy as np


def _parameter_state(model):
    """Portable parameter state; no model, data or operator matrices are duplicated."""
    from ...parameters import _encode_links
    parameters = []
    for name in model.param_names:
        p = getattr(model, name)
        parameters.append(dict(name=name, value=np.asarray(p.value).tolist(),
                               bounds=[None if b is None else float(b) for b in p.bounds],
                               fixed=bool(p.fixed), tied=bool(p.tied),
                               unit=None if p.unit is None else str(p.unit)))
    return dict(parameters=parameters, links=_encode_links(model))


def _sample_interval(draws, bounds, confidence, tolerance):
    """Central percentiles with explicit, scale-relative boundary accumulation."""
    tail = (100-confidence)/200
    lower, upper = np.percentile(draws, [100*tail, 100*(1-tail)], axis=0)
    q25, q75 = np.percentile(draws, [25, 75], axis=0)
    width = q75-q25
    sides = []
    for side, bound in zip(('lower', 'upper'), bounds):
        if bound is None or not np.isfinite(bound):
            continue
        scale = np.maximum(np.max(np.abs(draws), axis=0), abs(bound))
        numerical = 32*np.finfo(float).eps*np.maximum(scale, np.finfo(float).tiny)
        band = np.maximum(tolerance*width, numerical)
        near = draws <= bound+band if side == 'lower' else draws >= bound-band
        dominated = np.mean(near, axis=0) >= tail
        if np.any(dominated):
            sides.append(side)
        if side == 'lower':
            lower = np.where(dominated, np.nan, lower)
        else:
            upper = np.where(dominated, np.nan, upper)
    return lower, upper, sides
