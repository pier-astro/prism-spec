"""Tabular physical measurements for one explicitly identified line component."""
import numpy as np
from astropy.table import QTable

from ...composition import Model
from ...models.components import get_components
from ..component import _LineGroup
from .selection import _line_context, _SelectionEntry, SelectedLineProfile


class _Lines:
    """Prepared transitions sharing one model/fit context, without implicit sums."""

    def __init__(self, model, component, bootstrap):
        available = get_components(model, additive=False)
        if component is None:
            if not (isinstance(model, Model) and model.op is None
                    and isinstance(model.component, _LineGroup)):
                raise ValueError('Omit component= only for a single LineSet or its single-component fit result.')
            key = available.indices[0]
        else:
            if not isinstance(component, (str, int, np.integer)) or isinstance(component, bool):
                raise TypeError('component must identify one component by name or index, not a list.')
            leaf = available[component]
            key = next(index for index in available.indices if available[index] is leaf)
        leaf = available[key]
        if not isinstance(leaf, _LineGroup):
            raise TypeError('The chosen component must be a LineSet.')
        self._name = available.names[key]
        self._selections = []
        sampled_model = None if bootstrap is None else bootstrap.model
        for template, (name, _) in enumerate(leaf._setup['parameters']):
            positions, weights, slots = leaf._groups[template]
            scale = leaf.param_names[slots[leaf._scale_slot]]
            for position, weight in zip(positions, weights):
                entry = _SelectionEntry(key, self._name, name, float(position), scale,
                                        float(weight), template)
                selected = SelectedLineProfile(model, name, [entry], _components=available)
                selected._bootstrap_result = bootstrap
                selected._bootstrap_model = sampled_model
                self._selections.append(selected)

    def _table(self, metrics, name):
        entries = [selection.entries[0] for selection in self._selections]
        table = QTable(dict(name=[entry.template_name for entry in entries],
                            position=[entry.template_position for entry in entries],
                            weight=[entry.weight for entry in entries]))
        table['position'] *= self._selections[0].position_unit
        table[name] = [metric.value for metric in metrics]
        for column in ('std', 'lolim', 'uplim'):
            table[column] = [np.nan if getattr(metric, column) is None
                             else getattr(metric, column) for metric in metrics]
        if metrics[0].unit is not None:
            for column in (name, 'std', 'lolim', 'uplim'):
                table[column] *= metrics[0].unit
        table.meta['component'] = self._name
        return table

    def flux(self):
        """Return total per-transition fluxes and available fit uncertainties.

        Analytic integrals precede external response operators. Unknown standard
        deviations/limits are NaN. Bootstrap inputs propagate their actual refits.
        Custom profiles without analytic integrals raise rather than guessing a
        finite integration range; use select_line(...).measure(x=...) instead.
        """
        return self._table([selection.flux() for selection in self._selections], 'flux')

    def eqw(self, continuum, x=None, window=None, num=2048):
        """Return per-transition finite-window EW and available uncertainties.

        A continuum component name includes its covariance and correlations in
        the result's parameter graph. A scalar or external callable is exact.
        The continuum must be explicit; no background is inferred. Each line
        uses its own inferred window unless x or window is supplied. No sum of
        transitions is implied. Emission EWs are positive for positive continua.
        """
        if continuum is None:
            raise ValueError('Supply an explicit continuum value, function or component name.')
        metrics = [selection.eqw(continuum, x=x, window=window, num=num)
                   for selection in self._selections]
        return self._table(metrics, 'eqw')


def lines(model, component=None):
    """Prepare every transition in one line component for physical measurement.

    Parameters
    ----------
    model : LineSet, FitResult, BootstrapResult or joint observation view
        Model or fitted context. Fits retain their complete parameter graph and
        covariance, including external links. A model alone gives nominal values.
    component : str or int, optional
        Exactly one LineSet identified by name or leaf index. Omit only when the
        source itself is a single LineSet. Compound models require this argument;
        no automatic search or component summation is performed.

    Returns
    -------
    object
        Prepared view with flux() and eqw(continuum=...) methods returning QTables.
        Rows identify template name, rest position and weight; each physical
        transition remains separate, including repeated template names. Values
        and uncertainties use the same scalar measurement engine as select_line.

    Notes
    -----
    Imported from prism.modeling.lines.measurements. This function is not exported
    from prism.modeling because that name identifies the spectral-lines module.
    Fit inputs are copied once when preparing the view, not once per transition.
    Standalone model inputs retain live parameters, as select_line does. Rebuild
    the view after structural changes to a model or its line table.
    """
    model, bootstrap = _line_context(model)
    return _Lines(model, component, bootstrap)
