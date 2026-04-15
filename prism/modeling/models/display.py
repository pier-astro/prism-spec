"""Display helpers for Astropy-compatible models."""

from __future__ import annotations

from html import escape
import inspect
import re
import warnings

import numpy as np
from astropy.modeling import CompoundModel

from .components import ModelComponents

__all__ = [
    'enable_model_display',
    'disable_model_display',
    'get_model_expression',
    'format_model_text',
    'format_model_html',
    'format_fit_text',
    'format_fit_html',
    'show_model',
]

_PATCHED = False
_ORIGINAL_METHODS = {}
_MISSING = object()
_SAFE_FLAT_OPS = {'+', '*', '&', '|'}


def _component_label(model):
    name = getattr(model, 'name', None)
    if name:
        return str(name)
    return type(model).__name__


def _format_child_expression(model, parent_op):
    expr = _model_expression(model, top=False)
    if isinstance(model, CompoundModel):
        if model.op == parent_op and parent_op in _SAFE_FLAT_OPS:
            return expr
        return f"({expr})"
    return expr


def _model_expression(model, top=True):
    if not isinstance(model, CompoundModel):
        return _component_label(model)

    left = _format_child_expression(model.left, model.op)
    right = _format_child_expression(model.right, model.op)
    return f"{left} {model.op} {right}"


def get_model_expression(model):
    """Return a clean human-readable expression string for a model."""
    return _model_expression(model)


def _component_index(param_name):
    if '_' not in param_name:
        return None

    _, suffix = param_name.rsplit('_', 1)
    if suffix.isdigit():
        return int(suffix)
    return None


def _format_value(value):
    try:
        val = np.asarray(value).item()
    except Exception:
        return str(value)

    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return 'nan'
        if np.isposinf(val):
            return '∞'
        if np.isneginf(val):
            return '-∞'
        return f"{val:.6g}"
    return str(val)


def _format_bounds(bounds):
    if bounds is None:
        return '—'

    try:
        low, high = bounds
    except Exception:
        return str(bounds)

    def _bound_str(val, minus=False):
        if val is None:
            return '-∞' if minus else '∞'
        try:
            if not np.isfinite(val):
                return '-∞' if val < 0 else '∞'
        except Exception:
            return str(val)
        return _format_value(val)

    low_str = _bound_str(low, minus=True)
    high_str = _bound_str(high, minus=False)
    if low_str == '-∞' and high_str == '∞':
        return '—'
    return f"[{low_str}, {high_str}]"


def _has_finite_attr(obj, attr):
    if not hasattr(obj, attr):
        return False
    value = getattr(obj, attr)
    if value is None:
        return False
    try:
        return bool(np.isfinite(value))
    except Exception:
        return False


def _resolve_component_index(model, submodel_name):
    if not isinstance(model, CompoundModel):
        return None

    components = ModelComponents(model, additive=False)
    for idx, name in zip(components.indices, components.names):
        if name == submodel_name:
            return idx
    return None


def _unwrap_prism_tie(tied):
    inner_func = tied
    submodel_name = None

    closure = getattr(tied, '__closure__', None) or ()
    if getattr(tied, '__name__', '') == '_tie' and closure:
        for cell in closure:
            value = cell.cell_contents
            if callable(value):
                inner_func = value
            elif isinstance(value, str):
                submodel_name = value

    return inner_func, submodel_name


def _map_submodel_params(expr, model, submodel_name):
    if not submodel_name:
        return expr

    try:
        submodel = model[submodel_name]
    except Exception:
        return expr

    component_index = _resolve_component_index(model, submodel_name)
    if component_index is None:
        return expr

    for pname in sorted(getattr(submodel, 'param_names', ()), key=len, reverse=True):
        expr = re.sub(rf'\b{re.escape(pname)}\b', f'{pname}_{component_index}', expr)
    return expr


def _extract_callable_expression(func):
    try:
        source = inspect.getsource(func).strip()
    except Exception:
        source = ''

    expr = ''
    if 'lambda' in source and ':' in source:
        expr = source.split(':', 1)[1].strip()
    elif source:
        match = re.search(r'return\s+(.+)', source)
        if match:
            expr = match.group(1).strip()

    if expr:
        expr = expr.splitlines()[0].strip()
        expr = expr.rstrip(',)').strip()
        expr = re.sub(r'\b(?:m|model|self|sub)\.', '', expr)
        expr = re.sub(r'\s*([+\-*/])\s*', r' \1 ', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
        return expr
    return ''


def _fallback_callable_expression(func, model=None, submodel_name=None):
    code = getattr(func, '__code__', None)
    if code is None:
        return 'tied'

    names = list(getattr(code, 'co_names', ()))
    consts = [c for c in getattr(code, 'co_consts', ()) if isinstance(c, (int, float)) and not isinstance(c, bool)]

    if submodel_name and model is not None:
        try:
            submodel = model[submodel_name]
            names = [name for name in names if name in getattr(submodel, 'param_names', ())]
        except Exception:
            pass

    if len(names) == 1:
        ref = names[0]
        ref = _map_submodel_params(ref, model, submodel_name) if model is not None else ref
        if consts:
            coeff = _format_value(consts[0])
            return f'{coeff} * {ref}'
        return ref

    return 'tied'


def _format_tied_expression(tied, model=None):
    if tied in (False, None):
        return '-'

    func, submodel_name = _unwrap_prism_tie(tied)
    expr = _extract_callable_expression(func)

    if expr and model is not None:
        expr = _map_submodel_params(expr, model, submodel_name)

    if not expr:
        expr = _fallback_callable_expression(func, model=model, submodel_name=submodel_name)

    return f'= {expr}'


def _detect_render_mode(option='auto'):
    if option in ('text', 'txt', 'cli'):
        return 'text'
    if option in ('html', 'notebook'):
        return 'html'

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and 'IPKernelApp' in getattr(ip, 'config', {}):
            return 'html'
    except Exception:
        pass
    return 'text'


def _iter_rows(model):
    if isinstance(model, CompoundModel):
        components = ModelComponents(model, additive=False)
        index_to_name = dict(zip(components.indices, components.names))
    else:
        index_to_name = {}

    for param_name in model.param_names:
        param = getattr(model, param_name)
        component_index = _component_index(param_name) if isinstance(model, CompoundModel) else None
        component = index_to_name.get(component_index, '')
        tied = getattr(param, 'tied', False)
        yield {
            'component': component,
            'parameter': param_name,
            'value': _format_value(getattr(param, 'value', param)),
            'fixed': bool(getattr(param, 'fixed', False)),
            'tied': tied not in (False, None),
            'bounds': _format_bounds(getattr(param, 'bounds', None)),
        }


def _column_names(model):
    cols = ['Parameter', 'Value', 'Fixed', 'Tied', 'Bounds']
    if isinstance(model, CompoundModel):
        cols.insert(0, 'Component')
    return cols


def _group_rows(rows):
    grouped = []
    current_component = None
    current_rows = []

    for row in rows:
        component = row['component']
        if component != current_component:
            if current_rows:
                grouped.append((current_component, current_rows))
            current_component = component
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        grouped.append((current_component, current_rows))

    return grouped


def _display_style():
    return """
<style>
.prism-model-display {
    background: #0f1115;
    color: #e6edf3;
    border-radius: 8px;
    padding: 0.65em 0.8em;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.prism-model-display .expr {
    font-weight: 600;
    margin-bottom: 0.45em;
    color: #f0f6fc;
}
.prism-model-display table {
    width: auto;
    max-width: 100%;
    margin: 0;
    border-collapse: collapse;
    background: #0f1115;
}
.prism-model-display thead th {
    text-align: center;
    vertical-align: middle;
    font-weight: 600;
    color: #9fb0c0;
    padding: 0.35em 0.55em;
    border: none;
    border-bottom: 1px solid #2d333b;
}
.prism-model-display tbody td {
    padding: 0.35em 0.55em;
    border: none;
}
.prism-model-display tbody tr.component-start td {
    border-top: 1px solid #2d333b;
}
.prism-model-display tbody tr:hover td:not(.component-cell) {
    background: #2C6DB8;
}
.prism-model-display td.component-cell {
    background: #e6edf3;
    color: #0f1115;
    font-weight: 600;
    text-align: center;
    vertical-align: middle;
    border-radius: 6px;
}
.prism-model-display td.checkbox-cell {
    text-align: center;
    width: 3.2em;
}
.prism-model-display input[type="checkbox"] {
    --size: 1.2em;
    --brand-color: #3fb950;
    appearance: none;
    -webkit-appearance: none;
    width: var(--size);
    height: var(--size);
    border: 1.5px solid #484f58;
    border-radius: 0.25em;
    background-color: transparent;
    cursor: default;
    vertical-align: middle;
    transition: background-color 0.2s, border-color 0.2s;
    outline: none;
}
.prism-model-display input[type="checkbox"]:checked {
    background-color: var(--brand-color);
    border-color: var(--brand-color);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E");
    background-size: 80%;
    background-repeat: no-repeat;
    background-position: center;
}
</style>
    """.strip()


def _iter_fit_rows(model):
    for row in _iter_rows(model):
        param = getattr(model, row['parameter'])
        if row['fixed']:
            extra = 'frozen'
        elif row['tied']:
            extra = _format_tied_expression(getattr(param, 'tied', None), model=model)
        elif _has_finite_attr(param, 'std'):
            extra = f"± {_format_value(param.std)}"
        else:
            extra = '-'

        limits = '-'
        if _has_finite_attr(param, 'lolim') or _has_finite_attr(param, 'uplim'):
            low = _format_value(getattr(param, 'lolim', '-')) if hasattr(param, 'lolim') else '-'
            high = _format_value(getattr(param, 'uplim', '-')) if hasattr(param, 'uplim') else '-'
            limits = f'[{low}, {high}]'

        yield {
            'component': row['component'],
            'parameter': row['parameter'],
            'value': row['value'],
            'extra': extra,
            'limits': limits,
        }


def _has_fit_info(model):
    for param_name in model.param_names:
        param = getattr(model, param_name)
        if _has_finite_attr(param, 'std') or _has_finite_attr(param, 'lolim') or _has_finite_attr(param, 'uplim'):
            return True
    return False


def format_model_text(model):
    rows = list(_iter_rows(model))
    headers = _column_names(model)
    keys = ['parameter', 'value', 'fixed', 'tied', 'bounds']
    if isinstance(model, CompoundModel):
        keys.insert(0, 'component')

    display_rows = []
    for row in rows:
        display_row = row.copy()
        display_row['fixed'] = 'yes' if row['fixed'] else 'no'
        display_row['tied'] = 'yes' if row['tied'] else 'no'
        display_rows.append(display_row)

    widths = []
    for header, key in zip(headers, keys):
        widths.append(max(len(header), *(len(str(row[key])) for row in display_rows)) if display_rows else len(header))

    def _fmt_row(values):
        return ' | '.join(str(value).ljust(width) for value, width in zip(values, widths))

    lines = [f"Expression: {get_model_expression(model)}", '', _fmt_row(headers)]
    lines.append('-+-'.join('-' * width for width in widths))

    for row in display_rows:
        lines.append(_fmt_row([row[key] for key in keys]))

    return '\n'.join(lines)


def _checkbox_html(checked):
    attrs = ' checked' if checked else ''
    return f'<input type="checkbox" disabled{attrs}>'


def format_model_html(model):
    rows = list(_iter_rows(model))
    headers = _column_names(model)
    compound = isinstance(model, CompoundModel)
    groups = _group_rows(rows) if compound else [(None, rows)]

    html_lines = [
        _display_style(),
        '<div class="prism-model-display">',
        f'<div class="expr">Expression: {escape(get_model_expression(model))}</div>',
        '<table>',
        '<thead><tr>',
    ]

    for header in headers:
        html_lines.append(f'<th>{escape(header)}</th>')
    html_lines.append('</tr></thead><tbody>')

    for group_index, (component, comp_rows) in enumerate(groups):
        for row_index, row in enumerate(comp_rows):
            row_class = ' class="component-start"' if compound and group_index > 0 and row_index == 0 else ''
            html_lines.append(f'<tr{row_class}>')

            if compound and row_index == 0:
                html_lines.append(
                    f'<td rowspan="{len(comp_rows)}" class="component-cell">{escape(str(component))}</td>'
                )

            html_lines.append(f'<td>{escape(str(row["parameter"]))}</td>')
            html_lines.append(f'<td>{escape(str(row["value"]))}</td>')
            html_lines.append(f'<td class="checkbox-cell">{_checkbox_html(row["fixed"])}</td>')
            html_lines.append(f'<td class="checkbox-cell">{_checkbox_html(row["tied"])}</td>')
            html_lines.append(f'<td>{escape(str(row["bounds"]))}</td>')
            html_lines.append('</tr>')

    html_lines.extend(['</tbody></table>', '</div>'])
    return ''.join(html_lines)


def format_fit_text(model):
    rows = list(_iter_fit_rows(model))
    show_limits = any(row['limits'] != '-' for row in rows)

    display_rows = []
    for row in rows:
        display_row = row.copy()
        display_row['value_block'] = f"{row['value']}  {row['extra']}"
        display_rows.append(display_row)

    headers = ['Parameter', 'Value']
    keys = ['parameter', 'value_block']
    if show_limits:
        headers.append('Limits')
        keys.append('limits')
    if isinstance(model, CompoundModel):
        headers.insert(0, 'Component')
        keys.insert(0, 'component')

    widths = []
    for header, key in zip(headers, keys):
        widths.append(max(len(header), *(len(str(row[key])) for row in display_rows)) if display_rows else len(header))

    def _fmt_row(values):
        return ' | '.join(str(value).ljust(width) for value, width in zip(values, widths))

    lines = [f"Expression: {get_model_expression(model)}", '', _fmt_row(headers)]
    lines.append('-+-'.join('-' * width for width in widths))
    for row in display_rows:
        lines.append(_fmt_row([row[key] for key in keys]))
    return '\n'.join(lines)


def format_fit_html(model):
    rows = list(_iter_fit_rows(model))
    show_limits = any(row['limits'] != '-' for row in rows)
    compound = isinstance(model, CompoundModel)
    groups = _group_rows(rows) if compound else [(None, rows)]

    html_lines = [
        _display_style(),
        '<div class="prism-model-display">',
        f'<div class="expr">Expression: {escape(get_model_expression(model))}</div>',
        '<table>',
        '<thead><tr>',
    ]

    if compound:
        html_lines.append('<th rowspan="2">Component</th>')
    html_lines.append('<th rowspan="2">Parameter</th>')
    html_lines.append('<th colspan="2" style="text-align:center;">Value</th>')
    if show_limits:
        html_lines.append('<th rowspan="2">Limits</th>')
    html_lines.append('</tr><tr><th></th><th></th></tr></thead><tbody>')

    for group_index, (component, comp_rows) in enumerate(groups):
        for row_index, row in enumerate(comp_rows):
            row_class = ' class="component-start"' if compound and group_index > 0 and row_index == 0 else ''
            html_lines.append(f'<tr{row_class}>')

            if compound and row_index == 0:
                html_lines.append(
                    f'<td rowspan="{len(comp_rows)}" class="component-cell">{escape(str(component))}</td>'
                )

            html_lines.append(f'<td>{escape(str(row["parameter"]))}</td>')
            html_lines.append(f'<td>{escape(str(row["value"]))}</td>')
            html_lines.append(f'<td>{escape(str(row["extra"]))}</td>')
            if show_limits:
                html_lines.append(f'<td>{escape(str(row["limits"]))}</td>')
            html_lines.append('</tr>')

    html_lines.extend(['</tbody></table>', '</div>'])
    return ''.join(html_lines)


def show_model(model, mode='auto', option=None):
    if option is not None:
        mode = option

    if not _has_fit_info(model):
        warnings.warn(
            'No parameter std or interval limits were found on this model. Compute or attach fit uncertainties first.',
            stacklevel=2,
        )

    render_mode = _detect_render_mode(mode)
    if render_mode == 'html':
        html = format_fit_html(model)
        if mode == 'auto':
            try:
                from IPython.display import HTML
                return HTML(html)
            except Exception:
                return html
        return html

    text = format_fit_text(model)
    if mode == 'auto':
        print(text)
    return text


def _patched_repr(self):
    values = []
    for name in self.param_names[:4]:
        param = getattr(self, name)
        values.append(f"{name}={_format_value(getattr(param, 'value', param))}")

    inner = ', '.join(values)
    if len(self.param_names) > 4:
        inner += ', ...'

    if isinstance(self, CompoundModel):
        if inner:
            return f"<CompoundModel({get_model_expression(self)}; {inner})>"
        return f"<CompoundModel({get_model_expression(self)})>"

    return f"<{type(self).__name__}({inner})>"


def _patched_str(self):
    return format_model_text(self)


def _patched_repr_html(self):
    return format_model_html(self)


def _patched_show(self, mode='auto', option=None):
    return show_model(self, mode=mode, option=option)


def enable_model_display():
    """Enable enhanced text and HTML display for Astropy-compatible models."""
    global _PATCHED

    if _PATCHED:
        return

    from astropy.modeling.core import Model

    classes = [Model, CompoundModel]
    for cls in classes:
        _ORIGINAL_METHODS[cls] = {
            'repr': cls.__repr__,
            'str': cls.__str__,
            'html': getattr(cls, '_repr_html_', None),
            'html_dunder': getattr(cls, '__repr_html__', None),
            'expression': getattr(cls, 'expression', _MISSING),
            'show': getattr(cls, 'show', _MISSING),
        }
        cls.__repr__ = _patched_repr
        cls.__str__ = _patched_str
        cls._repr_html_ = _patched_repr_html
        cls.__repr_html__ = _patched_repr_html
        cls.expression = property(get_model_expression)
        cls.show = _patched_show

    _PATCHED = True


def disable_model_display():
    """Restore the original Astropy model display methods."""
    global _PATCHED

    if not _PATCHED:
        return

    for cls, methods in _ORIGINAL_METHODS.items():
        cls.__repr__ = methods['repr']
        cls.__str__ = methods['str']
        if methods['html'] is None:
            try:
                delattr(cls, '_repr_html_')
            except AttributeError:
                pass
        else:
            cls._repr_html_ = methods['html']

        if methods['html_dunder'] is None:
            try:
                delattr(cls, '__repr_html__')
            except AttributeError:
                pass
        else:
            cls.__repr_html__ = methods['html_dunder']

        if methods['expression'] is _MISSING:
            try:
                delattr(cls, 'expression')
            except AttributeError:
                pass
        else:
            cls.expression = methods['expression']

        if methods['show'] is _MISSING:
            try:
                delattr(cls, 'show')
            except AttributeError:
                pass
        else:
            cls.show = methods['show']

    _PATCHED = False
