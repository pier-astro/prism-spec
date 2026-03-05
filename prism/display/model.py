"""
Model inspection and display utilities for Astropy models.

Provides formatted views of model parameters, constraints, and structure
optimized for both terminal and Jupyter notebooks.

ConvolvedModel support: Display functions automatically unwrap ConvolvedModel
to show the underlying source model structure with the wrapper name, e.g.,
"rsp(continuum (0) + line1 (1))" for a ResponseModel named 'rsp'.
"""

import numpy as np
import inspect
import uuid
from astropy.modeling import CompoundModel

from .sciformat import format_value
from .styles import MODEL_TABLE_CSS, COLLAPSIBLE_CSS, FLUX_TABLE_CSS, get_toggle_script


# --- Environment Detection ---

def _detect_environment():
    """Detect if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None and 'IPKernelApp' in ipy.config:
            return 'jupyter'
    except:
        pass
    return 'terminal'


# --- ConvolvedModel Helper ---

def _unwrap_convolved(model):
    """
    Unwrap ConvolvedModel to get source model and wrapper info.
    
    Returns (source_model, wrapper_name) where wrapper_name is None if not wrapped.
    """
    from fantasylab.models.convolved import ConvolvedModel
    
    if isinstance(model, ConvolvedModel):
        return model._source, model.name
    return model, None


# --- Component Helpers ---

def _get_components(model):
    """Get component names and indices from model."""
    # Unwrap ConvolvedModel first
    model, _ = _unwrap_convolved(model)
    
    if not isinstance(model, CompoundModel):
        name = model.name if hasattr(model, 'name') and model.name else type(model).__name__
        return [(0, name)]
    
    # Import here to avoid circular imports
    from fantasylab.models.components import get_components
    comps = get_components(model, additive=False)
    return list(zip(comps.indices, comps.names))


def _build_model_expression(model):
    """Build model expression string showing structure: "rsp(continuum (0) + line1 (1))" """
    # Unwrap ConvolvedModel and get wrapper name
    source_model, wrapper_name = _unwrap_convolved(model)
    
    if not isinstance(source_model, CompoundModel):
        name = source_model.name if hasattr(source_model, 'name') and source_model.name else type(source_model).__name__
        expr = f"{name} (0)"
        return f"{wrapper_name}({expr})" if wrapper_name else expr
    
    # Build expression tree to preserve operators
    idx = [0]  # Mutable counter for leaf indexing
    
    def build_expr(submodel):
        if not isinstance(submodel, CompoundModel):
            # Leaf node
            name = submodel.name if hasattr(submodel, 'name') and submodel.name else type(submodel).__name__
            result = f"{name} ({idx[0]})"
            idx[0] += 1
            return result
        
        # Compound: recurse left and right
        left_expr = build_expr(submodel.left)
        right_expr = build_expr(submodel.right)
        op = submodel.op  # '+', '-', '*', '/', '**', '&', '|'
        
        # Format: left op right
        # Add parentheses for clarity if mixing different operators
        return f"{left_expr} {op} {right_expr}"
    
    expr = build_expr(source_model)
    return f"{wrapper_name}({expr})" if wrapper_name else expr


def _param_to_component(model):
    """Map parameter names to (component_idx, component_name)."""
    # Unwrap ConvolvedModel first
    model, _ = _unwrap_convolved(model)
    
    if not isinstance(model, CompoundModel):
        name = model.name if hasattr(model, 'name') and model.name else type(model).__name__
        return {pname: (0, name) for pname in model.param_names}
    
    comps = _get_components(model)
    comp_dict = {idx: name for idx, name in comps}
    
    param_map = {}
    for pname in model.param_names:
        # Extract suffix _N
        if '_' in pname:
            parts = pname.rsplit('_', 1)
            if parts[-1].isdigit():
                idx = int(parts[-1])
                if idx in comp_dict:
                    param_map[pname] = (idx, comp_dict[idx])
                    continue
        param_map[pname] = (None, 'Unknown')
    
    return param_map


# --- Tie Expression Extraction ---

def _format_tie_expression(tie_func, model=None):
    """Extract human-readable expression from tie function."""
    # Try source code inspection (works for simple lambdas)
    try:
        source = inspect.getsource(tie_func)
        if 'lambda' in source:
            expr = source.split(':', 1)[1].strip()
            expr = expr.replace('m.', '').replace('model.', '')
            expr = expr.rstrip(')').split('\n')[0].strip()
            return f"→ {expr}"
    except:
        pass
    
    # Try closure inspection for wrapped functions
    if hasattr(tie_func, '__closure__') and tie_func.__closure__:
        for cell in tie_func.__closure__:
            try:
                cell_value = cell.cell_contents
                if callable(cell_value) and hasattr(cell_value, '__name__'):
                    if '<lambda>' in cell_value.__name__:
                        try:
                            inner_source = inspect.getsource(cell_value)
                            if 'lambda' in inner_source:
                                expr = inner_source.split(':', 1)[1].strip()
                                expr = expr.replace('m.', '').replace('model.', '').replace('sub.', '')
                                expr = expr.rstrip(')').rstrip(',').split('\n')[0].strip()
                                return f"→ {expr}"
                        except:
                            pass
            except:
                pass
    
    return "[tied]"


# --- Parameter Classification ---

def _classify_parameter(model, pname, output_format='html', sig_digits=2):
    """
    Classify parameter and extract display information.
    
    Returns dict with: value, unit, free, constraint
    """
    param = getattr(model, pname)
    value = param.value if hasattr(param, 'value') else param
    
    # Check for array values (not supported)
    if np.ndim(value) > 0:
        raise ValueError(f"Parameter '{pname}' has array value. Only scalars supported.")
    
    info = {'value': None, 'unit': None, 'free': True, 'constraint': ''}
    
    # Unit
    if hasattr(param, 'unit') and param.unit is not None:
        info['unit'] = str(param.unit)
    
    # Format value with uncertainty if available
    if hasattr(param, 'median') and hasattr(param, 'lolim') and hasattr(param, 'uplim'):
        delta_lo = param.median - param.lolim
        delta_hi = param.uplim - param.median
        info['value'] = format_value(param.median, (delta_lo, delta_hi), 
                                      output=output_format, sig_digits=sig_digits, use_dollars=False)
    elif hasattr(param, 'std') and param.std is not None:
        info['value'] = format_value(value, param.std, 
                                      output=output_format, sig_digits=sig_digits, use_dollars=False)
    else:
        info['value'] = format_value(value, None, 
                                      output=output_format, sig_digits=sig_digits, use_dollars=False)
    
    # Constraint status
    if hasattr(param, 'tied') and param.tied is not False:
        info['free'] = False
        info['constraint'] = _format_tie_expression(param.tied, model)
    elif hasattr(param, 'fixed') and param.fixed:
        info['free'] = False
        info['constraint'] = 'Fixed'
    else:
        # Check bounds
        if hasattr(param, 'bounds'):
            lb, ub = param.bounds
            if lb is not None or ub is not None:
                lb_str = f"{float(lb):.3g}" if isinstance(lb, (float, int)) else ('-∞' if lb is None else str(lb))
                ub_str = f"{float(ub):.3g}" if isinstance(ub, (float, int)) else ('+∞' if ub is None else str(ub))
                info['constraint'] = f"[{lb_str}, {ub_str}]"
    
    return info


# --- Row Generators ---

def _iter_param_rows(model, output_format='html', sig_digits=2, filter_fn=None):
    """
    Generate parameter rows for display.
    
    Yields dicts with: component, component_idx, parameter, value, unit, free, constraint
    """
    param_map = _param_to_component(model)
    
    for pname in model.param_names:
        comp_idx, comp_name = param_map.get(pname, (None, 'Unknown'))
        info = _classify_parameter(model, pname, output_format, sig_digits)
        
        row = {
            'component': comp_name,
            'component_idx': comp_idx,
            'parameter': pname,
            'value': info['value'],
            'unit': info['unit'],
            'free': info['free'],
            'constraint': info['constraint'],
            'flux': None  # Will be filled by flux integration
        }
        
        if filter_fn is None or filter_fn(row):
            yield row


def _format_flux_value(flux, output_format='html', sig_digits=2):
    """Format a Flux object for display."""
    if flux.method == 'samples':
        delta_lo = flux.value - flux.lolim if flux.lolim is not None else None
        delta_hi = flux.uplim - flux.value if flux.uplim is not None else None
        # Show asymmetric if at least one side is valid
        if (delta_lo is not None and delta_lo != 0) or (delta_hi is not None and delta_hi != 0):
            return format_value(flux.value, (delta_lo, delta_hi), 
                               output=output_format, sig_digits=sig_digits, use_dollars=False)
    elif flux.std is not None and flux.std != 0 and np.isfinite(flux.std):
        return format_value(flux.value, flux.std, 
                           output=output_format, sig_digits=sig_digits, use_dollars=False)
    
    return format_value(flux.value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)


def _iter_model_rows_with_fluxes(model, fluxes, output_format='html', sig_digits=2):
    """
    Generate rows interleaving parameters with flux rows after each component.
    
    Yields rows in order: params for comp 0, fluxes for comp 0, params for comp 1, ...
    """
    # Unwrap ConvolvedModel to get the source for component/flux matching
    source_model, _ = _unwrap_convolved(model)
    
    # Group parameters by component
    param_map = _param_to_component(model)
    
    # Determine component order
    comp_order = []
    seen = set()
    for pname in model.param_names:
        comp_idx, comp_name = param_map.get(pname, (None, 'Unknown'))
        if comp_idx not in seen:
            comp_order.append((comp_idx, comp_name))
            seen.add(comp_idx)
    
    for comp_idx, comp_name in comp_order:
        # Yield parameters for this component
        for pname in model.param_names:
            p_idx, p_name = param_map.get(pname, (None, 'Unknown'))
            if p_idx != comp_idx:
                continue
            
            info = _classify_parameter(model, pname, output_format, sig_digits)
            yield {
                'component': comp_name,
                'component_idx': comp_idx,
                'parameter': pname,
                'value': info['value'],
                'unit': info['unit'],
                'free': info['free'],
                'constraint': info['constraint'],
                'flux': None
            }
        
        # Yield flux rows for this component
        if not fluxes or comp_idx is None:
            continue
            
        if isinstance(source_model, CompoundModel):
            # Match fluxes by extracting the component index from the last underscore
            # Flux keys are like "Ha_6563_1" where _1 is the component suffix
            matching = []
            for k, v in fluxes.items():
                # Split on last underscore to get potential component index
                parts = k.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        flux_comp_idx = int(parts[1])
                        if flux_comp_idx == comp_idx:
                            matching.append((k, v))
                    except ValueError:
                        # Last part is not a number, skip
                        pass
        else:
            matching = list(fluxes.items())
        
        for name, flux in sorted(matching, key=lambda x: x[0]):
            # Strip component suffix for display
            if isinstance(source_model, CompoundModel):
                parts = name.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        int(parts[1])  # Verify it's a number
                        disp_name = parts[0]
                    except ValueError:
                        disp_name = name
                else:
                    disp_name = name
            else:
                disp_name = name
            
            if not disp_name.startswith('flux'):
                disp_name = f"flux_{disp_name}"
            
            yield {
                'component': comp_name,
                'component_idx': comp_idx,
                'parameter': disp_name,
                'value': _format_flux_value(flux, output_format, sig_digits),
                'unit': None,
                'free': None,  # Marker for flux row
                'constraint': '-',
                'flux': ''
            }


# --- Rendering ---

def _collect_flags(rows):
    """Check data for units and fluxes presence."""
    has_units = any(row['unit'] for row in rows)
    has_fluxes = any(row['free'] is None for row in rows)  # flux rows have free=None
    return has_units, has_fluxes


def _render_text(rows, has_units, has_fluxes, model_expr=None):
    """Render as plain text table."""
    rows = list(rows)  # Materialize if generator
    
    headers = ['Component', 'Parameter', 'Value']
    if has_units:
        headers.append('Unit')
    headers.extend(['Free', 'Constraint'])
    
    # Calculate column widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        widths['Component'] = max(widths['Component'], len(row['component']))
        widths['Parameter'] = max(widths['Parameter'], len(row['parameter']))
        widths['Value'] = max(widths['Value'], len(str(row['value'])))
        if has_units and row['unit']:
            widths['Unit'] = max(widths['Unit'], len(row['unit']))
        widths['Free'] = max(widths['Free'], 3)
        if row['constraint']:
            widths['Constraint'] = max(widths['Constraint'], len(row['constraint']))
    
    fmt = "  ".join(f"{{:<{widths[h]}}}" for h in headers)
    sep = "=" * (sum(widths.values()) + 2 * (len(headers) - 1))
    
    lines = []
    if model_expr:
        lines.extend([f"Model: {model_expr}", ""])
    
    lines.append(fmt.format(*headers))
    lines.append(sep)
    
    for row in rows:
        values = [row['component'], row['parameter'], str(row['value'])]
        if has_units:
            values.append(row['unit'] or '')
        values.append('-' if row['free'] is None else ('Yes' if row['free'] else 'No'))
        values.append(row['constraint'])
        lines.append(fmt.format(*values))
    
    lines.append(sep)
    return '\n'.join(lines)


def _render_html(rows, has_units, has_fluxes, model_expr=None):
    """Render as HTML table with optional collapsible feature."""
    rows = list(rows)  # Materialize if generator
    num_params = len(rows)
    is_collapsible = num_params > 10
    
    # Unique ID for this table (allows multiple tables on page)
    table_id = f"paramTable_{uuid.uuid4().hex[:8]}"
    
    html = [f'<style>{MODEL_TABLE_CSS}']
    if is_collapsible:
        html.append(COLLAPSIBLE_CSS)
    html.append('</style>')
    
    if model_expr:
        html.append(f'<div class="fantasylab-model-expr">Model: {model_expr}</div>')
    
    if is_collapsible:
        html.append(f'<button class="fantasylab-toggle-btn" id="btn_{table_id}" '
                   f'onclick="toggleTable_{table_id}()">▼ Show All {num_params} Parameters</button>')
        html.append(f'<table class="fantasylab-model-summary fantasylab-collapsible-table" id="{table_id}">')
    else:
        html.append('<table class="fantasylab-model-summary">')
    
    # Header
    html.append('  <thead><tr>')
    html.append('    <th>Component</th><th>Parameter</th><th>Value</th>')
    if has_units:
        html.append('    <th>Unit</th>')
    html.append('    <th>Free</th><th>Constraint</th>')
    html.append('  </tr></thead>')
    
    # Body
    html.append('  <tbody>')
    for row in rows:
        html.append('    <tr>')
        html.append(f'      <td class="component-name">{row["component"]}</td>')
        html.append(f'      <td class="param-name">{row["parameter"]}</td>')
        html.append(f'      <td style="text-align: right">{row["value"]}</td>')
        
        if has_units:
            html.append(f'      <td>{row["unit"] or ""}</td>')
        
        if row['free'] is None:
            html.append('      <td style="text-align: center">-</td>')
        else:
            cls = 'free-yes' if row['free'] else 'free-no'
            sym = '☑' if row['free'] else '☐'
            html.append(f'      <td class="{cls}">{sym}</td>')
        
        html.append(f'      <td class="constraint">{row["constraint"] or ""}</td>')
        html.append('    </tr>')
    
    html.append('  </tbody></table>')
    
    if is_collapsible:
        html.append(f'<script>{get_toggle_script(table_id, num_params)}</script>')
    
    return '\n'.join(html)


def _display_output(output, format_type):
    """Display output in appropriate format."""
    if format_type == 'html':
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        print(output)


# --- Public API ---

def show(model, format='auto', sig_digits=2, filter=None):
    """
    Display formatted summary of model parameters and structure.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect (single or compound).
    format : {'auto', 'text', 'html'}
        Output format. 'auto' detects environment.
    sig_digits : int
        Significant figures for display (default: 2).
    filter : {'free', 'tied', 'bounds'}, optional
        Show only specific parameter types.
        
    Examples
    --------
    >>> show(compound_model)
    >>> show(model, filter='free')  # Only free parameters
    >>> show(model, filter='tied')  # Only tied parameters
    """
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    # Define filter functions
    filter_fns = {
        'free': lambda r: r['free'] is True,
        'tied': lambda r: not r['free'] and ('→' in r['constraint'] or '[tied]' in r['constraint']),
        'bounds': lambda r: r['free'] and r['constraint'] and '[' in r['constraint'],
    }
    filter_fn = filter_fns.get(filter)
    
    rows = list(_iter_param_rows(model, format, sig_digits, filter_fn))
    
    if not rows:
        print(f"No {filter or 'matching'} parameters.")
        return
    
    has_units, has_fluxes = _collect_flags(rows)
    model_expr = _build_model_expression(model)
    
    if format == 'html':
        output = _render_html(rows, has_units, has_fluxes, model_expr)
    else:
        output = _render_text(rows, has_units, has_fluxes, model_expr)
    
    _display_output(output, format)


def show_model_lineflux(model, fluxes=None, format='auto', sig_digits=2):
    """
    Display model parameters with derived line fluxes.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    fluxes : dict, optional
        Pre-computed {name: Flux} dict. If None, extracts from model.
    format : {'auto', 'text', 'html'}
        Output format.
    sig_digits : int
        Significant figures for display (default: 2).
    """
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    # Extract fluxes if not provided
    if fluxes is None:
        from fantasylab.models.flux import extract_fluxes
        fluxes = extract_fluxes(model)
    
    rows = list(_iter_model_rows_with_fluxes(model, fluxes, format, sig_digits))
    has_units, has_fluxes = _collect_flags(rows)
    model_expr = _build_model_expression(model)
    
    if format == 'html':
        output = _render_html(rows, has_units, has_fluxes, model_expr)
    else:
        output = _render_text(rows, has_units, has_fluxes, model_expr)
    
    _display_output(output, format)


def show_fluxes(model, fluxes=None, format='auto', sig_digits=2):
    """
    Display a summary table of line fluxes only.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    fluxes : dict, optional
        Pre-computed fluxes. If None, extracts from model.
    format : {'auto', 'text', 'html'}
        Output format.
    sig_digits : int
        Significant figures for display (default: 2).
    """
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    if fluxes is None:
        from fantasylab.models.flux import extract_fluxes
        fluxes = extract_fluxes(model)
    
    if not fluxes:
        print("No line fluxes found in model.")
        return
    
    # Prepare data
    data = [(name, _format_flux_value(flux, format, sig_digits)) for name, flux in fluxes.items()]
    
    if format == 'html':
        html = [f'<style>{FLUX_TABLE_CSS}</style>']
        html.append('<table class="fantasylab-flux-table">')
        html.append('<thead><tr><th>Line Name</th><th>Flux</th></tr></thead>')
        html.append('<tbody>')
        for name, val in data:
            html.append(f'<tr><td>{name}</td><td>{val}</td></tr>')
        html.append('</tbody></table>')
        _display_output('\n'.join(html), 'html')
    else:
        # Text table
        w_name = max(len('Line Name'), max(len(n) for n, _ in data))
        w_flux = max(len('Flux'), max(len(v) for _, v in data))
        fmt = f"{{:<{w_name}}}  {{:<{w_flux}}}"
        
        print(fmt.format('Line Name', 'Flux'))
        print("=" * (w_name + w_flux + 2))
        for name, val in data:
            print(fmt.format(name, val))


def count_free(model):
    """
    Count number of free (fittable) parameters.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
        
    Returns
    -------
    int
        Number of free parameters.
        
    Examples
    --------
    >>> n_free = count_free(model)
    >>> dof = len(wavelength) - n_free
    """
    count = 0
    for pname in model.param_names:
        param = getattr(model, pname)
        is_fixed = getattr(param, 'fixed', False)
        is_tied = getattr(param, 'tied', False) is not False
        if not is_fixed and not is_tied:
            count += 1
    return count


# --- Convenience Wrappers (for backward compatibility) ---

def show_free(model, format='auto', sig_digits=2):
    """Display only free (fittable) parameters. Wrapper for show(filter='free')."""
    show(model, format=format, sig_digits=sig_digits, filter='free')


def show_tied(model, format='auto', sig_digits=2):
    """Display only tied parameters. Wrapper for show(filter='tied')."""
    show(model, format=format, sig_digits=sig_digits, filter='tied')


def show_bounds(model, format='auto', sig_digits=2):
    """Display only parameters with bounds. Wrapper for show(filter='bounds')."""
    show(model, format=format, sig_digits=sig_digits, filter='bounds')
