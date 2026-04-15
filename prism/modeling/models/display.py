"""Display helpers for Astropy-compatible models."""

from __future__ import annotations

from html import escape

import numpy as np
from astropy.modeling import CompoundModel

from .components import ModelComponents

__all__ = [
    'enable_model_display',
    'disable_model_display',
    'get_model_expression',
    'format_model_text',
    'format_model_html',
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

    style = """
<style>
/* Main container styling */
.prism-model-display {
    background: #0f1115;
    color: #e6edf3;
    border-radius: 8px;
    padding: 0.65em 0.8em;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Expression text styling */
.prism-model-display .expr {
    font-weight: 600;
    margin-bottom: 0.45em;
    color: #f0f6fc;
}

/* Table styling - Left aligned & Dynamic width */
.prism-model-display table {
    width: auto;           /* Shrinks to fit content */
    max-width: 100%;       /* Prevents overflow on small screens */
    margin: 0;             /* Forces alignment to the left */
    border-collapse: collapse;
    background: #0f1115;
}

/* Header cell styling */
.prism-model-display thead th {
    text-align: left;
    font-weight: 600;
    color: #9fb0c0;
    padding: 0.35em 0.55em;
    border: none;
    border-bottom: 1px solid #2d333b;
}

/* Body cell styling */
.prism-model-display tbody td {
    padding: 0.35em 0.55em;
    border: none;
}

/* Component row separator styling */
.prism-model-display tbody tr.component-start td {
    border-top: 1px solid #2d333b;
}

/* Row hover effect - Excludes the component cell to prevent highlight bleeding */
.prism-model-display tbody tr:hover td:not(.component-cell) {
    background: #2C6DB8;
}

/* Component name cell styling */
.prism-model-display td.component-cell {
    background: #e6edf3;
    color: #0f1115;
    font-weight: 600;
    text-align: center;
    vertical-align: middle;
    border-radius: 6px;
}

/* Checkbox cell styling */
.prism-model-display td.checkbox-cell {
    text-align: center;
    width: 3.2em;
}

/* Scalable Custom Checkbox styling */
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
    /* Inlined SVG checkmark for crispness and compactness */
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3E%3C/svg%3E");
    background-size: 80%;
    background-repeat: no-repeat;
    background-position: center;
}
</style>
    """.strip()

    html_lines = [
        style,
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
        }
        cls.__repr__ = _patched_repr
        cls.__str__ = _patched_str
        cls._repr_html_ = _patched_repr_html
        cls.__repr_html__ = _patched_repr_html
        cls.expression = property(get_model_expression)

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

    _PATCHED = False
