"""
Model inspection and display utilities for Astropy models.

Provides formatted views of model parameters, constraints, and structure
optimized for both terminal and Jupyter notebooks.
"""

import numpy as np
import inspect
from astropy.modeling import CompoundModel
from .components import get_components
# Import scientific formatter
from .sciformat import format_value
from .flux import extract_fluxes


def _detect_environment():
    """Detect if running in Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
            return 'jupyter'
    except:
        pass
    return 'terminal'


def _build_model_expression(model):
    """
    Build model expression string showing structure.
    
    Returns
    -------
    str
        Expression like "continuum (0) + line1 (1) + line2 (2)"
    """
    if not isinstance(model, CompoundModel):
        name = model.name if hasattr(model, 'name') and model.name else type(model).__name__
        return f"{name} (0)"
    
    # Get components
    comps = get_components(model, additive=False)
    
    # Build expression
    parts = []
    for idx, name in zip(comps.indices, comps.names):
        parts.append(f"{name} ({idx})")
    
    return " + ".join(parts)


def _get_component_info(model):
    """
    Map parameters to their components.
    
    Returns
    -------
    dict
        Mapping param_name → (component_idx, component_name)
    """
    if not isinstance(model, CompoundModel):
        # Single model
        name = model.name if hasattr(model, 'name') and model.name else type(model).__name__
        return {pname: (0, name) for pname in model.param_names}
    
    # Compound model - use component decomposition
    comps = get_components(model, additive=False)
    
    # Map parameters to components by suffix _N
    param_map = {}
    for pname in model.param_names:
        # Extract suffix
        if '_' in pname:
            parts = pname.rsplit('_', 1)
            if parts[-1].isdigit():
                idx = int(parts[-1])
                if idx < len(comps.indices):
                    param_map[pname] = (comps.indices[idx], comps.names[idx])
                    continue
        
        # Fallback: couldn't determine component
        param_map[pname] = (None, 'Unknown')
    
    return param_map


def _format_tie_expression(tie_func, model=None):
    """
    Extract human-readable expression from tie function.
    
    Parameters
    ----------
    tie_func : callable
        The tied parameter function
    model : Model, optional
        The model instance (used to extract parameter relationships)
    
    Returns
    -------
    str
        Expression like "→ amplitude_0 / 3" or "[tied]"
    """
    # Method 1: Try to get source code (works for lambdas defined in scripts)
    try:
        source = inspect.getsource(tie_func)
        # Look for lambda expression
        if 'lambda' in source:
            # Extract part after ':'
            expr = source.split(':', 1)[1].strip()
            # Clean up
            expr = expr.replace('m.', '').replace('model.', '')
            expr = expr.rstrip(')')  # Remove trailing )
            expr = expr.split('\n')[0]  # Take first line only
            expr = expr.strip()
            return f"→ {expr}"
    except:
        pass
    
    # Method 2: Check if this is a wrapper from fantasylab.fitting.tie()
    # Look for 'func' in closure variables
    if hasattr(tie_func, '__closure__') and tie_func.__closure__ is not None:
        for i, cell in enumerate(tie_func.__closure__):
            try:
                cell_value = cell.cell_contents
                # Check if it's a lambda or function
                if callable(cell_value) and hasattr(cell_value, '__name__'):
                    if '<lambda>' in cell_value.__name__ or 'lambda' in str(cell_value):
                        # Try to get source of the inner function
                        try:
                            inner_source = inspect.getsource(cell_value)
                            if 'lambda' in inner_source:
                                expr = inner_source.split(':', 1)[1].strip()
                                expr = expr.replace('m.', '').replace('model.', '').replace('sub.', '')
                                expr = expr.rstrip(')').rstrip(',')
                                expr = expr.split('\n')[0]
                                expr = expr.strip()
                                return f"→ {expr}"
                        except:
                            pass
                        
                        # Try bytecode analysis on inner function
                        try:
                            import sys
                            import dis
                            from io import StringIO
                            
                            old_stdout = sys.stdout
                            sys.stdout = StringIO()
                            dis.dis(cell_value)
                            bytecode = sys.stdout.getvalue()
                            sys.stdout = old_stdout
                            
                            # Look for parameter names in bytecode
                            lines = bytecode.split('\n')
                            attrs = []
                            for line in lines:
                                if 'LOAD_ATTR' in line:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        attr = parts[-1].strip('()')
                                        attrs.append(attr)
                            
                            # Filter and show parameter names
                            if attrs:
                                param_attrs = [a for a in attrs if a not in ('value', '__call__', 'submodel_names', 'param_names')]
                                if param_attrs:
                                    # Show the operation hint if we can find it
                                    if 'BINARY_TRUE_DIVIDE' in bytecode or 'BINARY_DIVIDE' in bytecode:
                                        return f"→ {param_attrs[0]} / ..."
                                    elif 'BINARY_MULTIPLY' in bytecode:
                                        return f"→ {param_attrs[0]} * ..."
                                    else:
                                        return f"→ {param_attrs[0]}"
                        except:
                            pass
            except:
                pass
    
    # Method 3: Try to analyze bytecode of outer function for simple cases
    if model is not None:
        try:
            import sys
            import dis
            from io import StringIO
            
            # Capture disassembly
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            dis.dis(tie_func)
            bytecode = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Look for LOAD_ATTR pattern (accessing model attributes)
            lines = bytecode.split('\n')
            attrs = []
            for line in lines:
                if 'LOAD_ATTR' in line:
                    # Extract attribute name from bytecode line
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        attr = parts[-1].strip('()')
                        attrs.append(attr)
            
            # Simple pattern: if we see parameter names, show them
            if attrs:
                # Filter out common non-parameter attributes
                param_attrs = [a for a in attrs if a not in ('value', '__call__', 'submodel_names')]
                if param_attrs:
                    return f"→ {param_attrs[0]}"  # Show first parameter reference
        except:
            pass
    
    return "[tied]"


def _classify_parameter(model, pname, output_format='html', sig_digits=2):
    """
    Classify parameter and extract all relevant information.
    
    Parameters
    ----------
    model : Model
        The model instance
    pname : str
        Parameter name
    output_format : {'html', 'text'}
        Output format for value strings
    sig_digits : int, optional
        Number of significant figures for display (default: 2)
    
    Returns
    -------
    dict
        Information about the parameter: value, uncertainty, unit, free status, constraint
    """
    param = getattr(model, pname)
    
    # Get value
    value = param.value if hasattr(param, 'value') else param
    
    # Check for multi-dimensional values
    if np.ndim(value) > 0:
        raise ValueError(
            f"Parameter '{pname}' has array value with shape {np.shape(value)}. "
            f"Model display only supports scalar parameters."
        )
    
    # Use sciformat for value and uncertainty
    info = {
        'value': None,
        'uncertainty': None,
        'unit': None,
        'free': True,
        'constraint': ''
    }
    
    # Check for unit
    if hasattr(param, 'unit') and param.unit is not None:
        info['unit'] = str(param.unit)
    
    # Format value and uncertainty using sciformat
    if hasattr(param, 'median') and hasattr(param, 'lolim') and hasattr(param, 'uplim'):
        delta_lo = param.median - param.lolim
        delta_hi = param.uplim - param.median
        info['value'] = format_value(param.median, (delta_lo, delta_hi), output=output_format, sig_digits=sig_digits, use_dollars=False)
        info['uncertainty'] = None  # Will be shown in value
    elif hasattr(param, 'std') and param.std is not None:
        info['value'] = format_value(value, param.std, output=output_format, sig_digits=sig_digits, use_dollars=False)
        info['uncertainty'] = None  # Will be shown in value
    else:
        info['value'] = format_value(value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)
        info['uncertainty'] = None
    
    # Check if tied
    if hasattr(param, 'tied') and param.tied is not False:
        info['free'] = False
        info['constraint'] = _format_tie_expression(param.tied, model)
    # Check if fixed
    elif hasattr(param, 'fixed') and param.fixed:
        info['free'] = False
        info['constraint'] = 'Fixed'
    else:
        # Free parameter - check for bounds
        if hasattr(param, 'bounds'):
            bounds = param.bounds
            if bounds != (None, None) and (bounds[0] is not None or bounds[1] is not None):
                lb = bounds[0] if bounds[0] is not None else '-∞'
                ub = bounds[1] if bounds[1] is not None else '+∞'
                # Format bounds with .3g precision if numeric
                try:
                    lb_fmt = f"{float(lb):.3g}" if isinstance(lb, (float, int)) and lb != '-∞' else lb
                except:
                    lb_fmt = lb
                try:
                    ub_fmt = f"{float(ub):.3g}" if isinstance(ub, (float, int)) and ub != '+∞' else ub
                except:
                    ub_fmt = ub
                info['constraint'] = f"[{lb_fmt}, {ub_fmt}]"
    
    return info


def _collect_model_data(model, output_format='html', fluxes=None, show_flux_column=True, sig_digits=2):
    """
    Collect all parameter information from model.
    
    Parameters
    ----------
    model : Model
        The model instance
    output_format : {'html', 'text'}
        Output format for value strings
    fluxes : dict, optional
        Dictionary of {name: Flux} to map to parameters
    show_flux_column : bool, optional
        Whether to show the "Derived Flux" column for amplitude parameters (default: True)
    sig_digits : int, optional
        Number of significant figures for display (default: 2)
    
    Returns
    -------
    list of dict
        Each dict contains: component, parameter, value, uncertainty, unit, free, constraint, flux
    bool, bool, bool
        has_units, has_uncertainties, has_fluxes flags
    """
    param_map = _get_component_info(model)
    
    data = []
    has_units = False
    has_uncertainties = False
    has_fluxes = False
    
    for pname in model.param_names:
        comp_idx, comp_name = param_map.get(pname, (None, 'Unknown'))
        
        # Get parameter info
        info = _classify_parameter(model, pname, output_format=output_format, sig_digits=sig_digits)
        
        # Track if we have units or uncertainties
        if info['unit'] is not None:
            has_units = True
        if info['uncertainty'] is not None:
            has_uncertainties = True
            
        # Map to flux
        flux_str = None
        if fluxes and show_flux_column:
            flux_obj = None
            # Case 1: Single model
            if not isinstance(model, CompoundModel):
                if pname == 'amplitude':
                    flux_obj = fluxes.get('flux')
                elif pname.startswith('amp_'):
                    flux_obj = fluxes.get(pname[4:])
            # Case 2: Compound model
            elif comp_idx is not None:
                suffix = f"_{comp_idx}"
                if pname == f"amplitude{suffix}":
                    # Find unique flux ending in suffix
                    candidates = [k for k in fluxes.keys() if k.endswith(suffix)]
                    if len(candidates) == 1:
                        flux_obj = fluxes[candidates[0]]
                elif pname.startswith('amp_') and pname.endswith(suffix):
                    # amp_Name_N -> Name_N
                    # But extract_fluxes returns Name_N
                    # And pname is amp_Name_N
                    # So pname[4:] should match
                    flux_obj = fluxes.get(pname[4:])
            
            if flux_obj is not None:
                has_fluxes = True
                # Format flux
                if flux_obj.method == 'samples':
                    delta_lo = flux_obj.value - flux_obj.lolim if flux_obj.lolim is not None else None
                    delta_hi = flux_obj.uplim - flux_obj.value if flux_obj.uplim is not None else None
                    # Check if errors are valid (not None or 0; NaN and inf are acceptable for boundaries)
                    delta_lo_valid = delta_lo is not None and delta_lo != 0
                    delta_hi_valid = delta_hi is not None and delta_hi != 0
                    # Show asymmetric errors if at least one side is valid (even if other is NaN)
                    if delta_lo_valid or delta_hi_valid:
                        flux_str = format_value(flux_obj.value, (delta_lo, delta_hi), output=output_format, sig_digits=sig_digits, use_dollars=False)
                    else:
                        flux_str = format_value(flux_obj.value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)
                elif flux_obj.std is not None and flux_obj.std != 0 and np.isfinite(flux_obj.std):
                    flux_str = format_value(flux_obj.value, flux_obj.std, output=output_format, sig_digits=sig_digits, use_dollars=False)
                else:
                    flux_str = format_value(flux_obj.value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)
        
        data.append({
            'component': comp_name,
            'component_idx': comp_idx,
            'parameter': pname,
            'value': info['value'],
            'uncertainty': info['uncertainty'],
            'unit': info['unit'],
            'free': info['free'],
            'constraint': info['constraint'],
            'flux': flux_str
        })
    
    # Insert flux rows if fluxes provided
    if fluxes:
        new_data = []
        current_comp_idx = None
        current_comp_name = None
        
        # Helper to add flux rows
        def add_fluxes_for_comp(c_idx, c_name):
            # Find matching fluxes
            matching_fluxes = []
            if not isinstance(model, CompoundModel):
                # Single model - all fluxes belong to it
                for name, flux in fluxes.items():
                    matching_fluxes.append((name, flux))
            elif c_idx is not None:
                # Compound model - look for suffix _{c_idx}
                suffix = f"_{c_idx}"
                for name, flux in fluxes.items():
                    if name.endswith(suffix):
                        matching_fluxes.append((name, flux))
            
            # Sort by name to be deterministic
            matching_fluxes.sort(key=lambda x: x[0])
            
            for name, flux in matching_fluxes:
                # Determine display name
                if isinstance(model, CompoundModel):
                    # Strip suffix
                    suffix = f"_{c_idx}"
                    if name.endswith(suffix):
                        disp_name = name[:-len(suffix)]
                    else:
                        disp_name = name
                else:
                    disp_name = name
                
                # Append flux_ prefix if not present
                if not disp_name.startswith('flux'):
                    disp_name = f"flux_{disp_name}"
                
                # Format value
                if flux.method == 'samples':
                    delta_lo = flux.value - flux.lolim if flux.lolim is not None else None
                    delta_hi = flux.uplim - flux.value if flux.uplim is not None else None
                    # Check if errors are valid (not None or 0; NaN and inf are acceptable for boundaries)
                    delta_lo_valid = delta_lo is not None and delta_lo != 0
                    delta_hi_valid = delta_hi is not None and delta_hi != 0
                    # Show asymmetric errors if at least one side is valid (even if other is NaN)
                    if delta_lo_valid or delta_hi_valid:
                        val_str = format_value(flux.value, (delta_lo, delta_hi), output=output_format, sig_digits=sig_digits, use_dollars=False)
                    else:
                        val_str = format_value(flux.value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)
                elif flux.std is not None and flux.std != 0 and np.isfinite(flux.std):
                    val_str = format_value(flux.value, flux.std, output=output_format, sig_digits=sig_digits, use_dollars=False)
                else:
                    val_str = format_value(flux.value, None, output=output_format, sig_digits=sig_digits, use_dollars=False)
                
                new_data.append({
                    'component': c_name,
                    'component_idx': c_idx,
                    'parameter': disp_name,
                    'value': val_str,
                    'uncertainty': None,
                    'unit': None,
                    'free': None, # Special flag for flux row
                    'constraint': '-',
                    'flux': '' # Derived Flux column empty
                })

        for row in data:
            c_idx = row['component_idx']
            c_name = row['component']
            
            if c_idx != current_comp_idx and current_comp_idx is not None:
                # Component changed
                add_fluxes_for_comp(current_comp_idx, current_comp_name)
            
            new_data.append(row)
            current_comp_idx = c_idx
            current_comp_name = c_name
            
        # Last component
        if current_comp_idx is not None:
            add_fluxes_for_comp(current_comp_idx, current_comp_name)
            
        data = new_data
    
    return data, has_units, has_uncertainties, has_fluxes


def _render_text(data, has_units, has_uncertainties, has_fluxes=False, model_expr=None):
    """Render parameter table as plain text."""
    # Build column headers
    headers = ['Component', 'Parameter', 'Value']
    if has_uncertainties:
        headers.append('Uncertainty')
    if has_units:
        headers.append('Unit')
    if has_fluxes:
        headers.append('Derived Flux')
    headers.extend(['Free', 'Constraint'])
    
    # Calculate column widths
    widths = {h: len(h) for h in headers}
    
    for row in data:
        widths['Component'] = max(widths['Component'], len(row['component']))
        widths['Parameter'] = max(widths['Parameter'], len(row['parameter']))
        # Value is already a formatted string
        widths['Value'] = max(widths['Value'], len(str(row['value'])))
        if has_uncertainties and row['uncertainty']:
            widths['Uncertainty'] = max(widths['Uncertainty'], len(row['uncertainty']))
        if has_units and row['unit']:
            widths['Unit'] = max(widths['Unit'], len(row['unit']))
        if has_fluxes and row['flux']:
            widths['Derived Flux'] = max(widths['Derived Flux'], len(row['flux']))
        widths['Free'] = max(widths['Free'], 3)  # "Yes" or "No"
        if row['constraint']:
            widths['Constraint'] = max(widths['Constraint'], len(row['constraint']))
    
    # Build format string
    fmt_parts = []
    for h in headers:
        fmt_parts.append(f"{{:<{widths[h]}}}")
    fmt = "  ".join(fmt_parts)
    
    # Build table
    lines = []
    
    # Model expression
    if model_expr:
        lines.append(f"Model: {model_expr}")
        lines.append("")
    
    # Header
    lines.append(fmt.format(*headers))
    lines.append("=" * (sum(widths.values()) + 2 * (len(headers) - 1)))
    
    # Data rows
    for row in data:
        values = [
            row['component'],
            row['parameter'],
            str(row['value'])  # Already formatted string
        ]
        if has_uncertainties:
            values.append(row['uncertainty'] or '')
        if has_units:
            values.append(row['unit'] or '')
        if has_fluxes:
            values.append(row['flux'] or '')
        
        if row['free'] is None:
            values.append('-')
        else:
            values.append('Yes' if row['free'] else 'No')
            
        values.append(row['constraint'])
        
        lines.append(fmt.format(*values))
    
    lines.append("=" * (sum(widths.values()) + 2 * (len(headers) - 1)))
    
    return '\n'.join(lines)


def _render_html(data, has_units, has_uncertainties, has_fluxes=False, model_expr=None):
    """Render parameter table as HTML."""
    # Check if we need collapsible table
    num_params = len(data)
    is_collapsible = num_params > 10
    
    # Build HTML table
    html = ['<style>']
    html.append('.fantasylab-model-summary {')
    html.append('  border-collapse: collapse;')
    html.append('  font-family: Arial, sans-serif;')
    html.append('  font-size: 13px;')
    html.append('  margin-top: 10px;')
    html.append('}')
    html.append('.fantasylab-model-expr {')
    html.append('  font-family: "Courier New", Consolas, monospace;')
    html.append('  font-size: 14px;')
    html.append('  font-weight: bold;')
    html.append('  margin-bottom: 8px;')
    html.append('  color: #333;')
    html.append('}')
    html.append('.fantasylab-model-summary th {')
    html.append('  background-color: #e8e8e8;')
    html.append('  border: 1px solid #ccc;')
    html.append('  padding: 8px 10px;')
    html.append('  text-align: left;')
    html.append('  font-weight: bold;')
    html.append('  color: #333;')
    html.append('}')
    html.append('.fantasylab-model-summary td {')
    html.append('  border: 1px solid #ddd;')
    html.append('  padding: 6px 10px;')
    html.append('  color: #333;')
    html.append('}')
    html.append('.fantasylab-model-summary tr:nth-child(even) {')
    html.append('  background-color: #f5f5f5;')
    html.append('}')
    html.append('.fantasylab-model-summary tr:nth-child(odd) {')
    html.append('  background-color: #ffffff;')
    html.append('}')
    html.append('.fantasylab-model-summary .component-name {')
    html.append('  font-family: "Courier New", Consolas, monospace;')
    html.append('  color: #333;')
    html.append('}')
    html.append('.fantasylab-model-summary .param-name {')
    html.append('  font-family: "Courier New", Consolas, monospace;')
    html.append('  color: #0066cc;')
    html.append('}')
    html.append('.fantasylab-model-summary .constraint {')
    html.append('  color: #666;')
    html.append('  font-style: italic;')
    html.append('  font-size: 12px;')
    html.append('}')
    html.append('.fantasylab-model-summary .uncertainty {')
    html.append('  font-size: 12px;')
    html.append('}')
    html.append('.fantasylab-model-summary .free-yes {')
    html.append('  color: #00aa00;')
    html.append('  text-align: center;')
    html.append('  font-size: 18px;')
    html.append('}')
    html.append('.fantasylab-model-summary .free-no {')
    html.append('  color: #cc0000;')
    html.append('  text-align: center;')
    html.append('  font-size: 18px;')
    html.append('}')
    html.append('.fantasylab-toggle-btn {')
    html.append('  background-color: #007acc;')
    html.append('  color: white;')
    html.append('  border: none;')
    html.append('  padding: 6px 12px;')
    html.append('  cursor: pointer;')
    html.append('  border-radius: 4px;')
    html.append('  font-size: 13px;')
    html.append('  margin-bottom: 8px;')
    html.append('}')
    html.append('.fantasylab-toggle-btn:hover {')
    html.append('  background-color: #005a9e;')
    html.append('}')
    html.append('.fantasylab-collapsible-table {')
    html.append('  display: none;')
    html.append('}')
    html.append('.fantasylab-collapsible-table.expanded {')
    html.append('  display: table;')
    html.append('}')
    html.append('</style>')
    
    # Model expression
    if model_expr:
        html.append(f'<div class="fantasylab-model-expr">Model: {model_expr}</div>')
    
    # Add toggle button if collapsible
    if is_collapsible:
        html.append(f'<button class="fantasylab-toggle-btn" onclick="toggleTable()">▼ Show All {num_params} Parameters</button>')
        html.append('<table class="fantasylab-model-summary fantasylab-collapsible-table" id="paramTable">')
    else:
        html.append('<table class="fantasylab-model-summary">')
    
    # Header
    html.append('  <thead>')
    html.append('    <tr>')
    html.append('      <th>Component</th>')
    html.append('      <th>Parameter</th>')
    html.append('      <th>Value</th>')
    if has_uncertainties:
        html.append('      <th>Uncertainty</th>')
    if has_units:
        html.append('      <th>Unit</th>')
    if has_fluxes:
        html.append('      <th>Derived Flux</th>')
    html.append('      <th>Free</th>')
    html.append('      <th>Constraint</th>')
    html.append('    </tr>')
    html.append('  </thead>')
    
    # Body
    html.append('  <tbody>')
    for row in data:
        html.append('    <tr>')
        html.append(f'      <td class="component-name">{row["component"]}</td>')
        html.append(f'      <td class="param-name">{row["parameter"]}</td>')
        # Value already formatted with error using sciformat
        html.append(f'      <td style="text-align: right">{row["value"]}</td>')
        # Uncertainty column is now always blank (all info in value)
        if has_uncertainties:
            html.append('      <td style="text-align: center"></td>')
        if has_units:
            unit = row['unit'] if row['unit'] else ''
            html.append(f'      <td>{unit}</td>')
        if has_fluxes:
            flux = row['flux'] if row['flux'] else ''
            html.append(f'      <td style="text-align: right">{flux}</td>')
        
        if row['free'] is None:
            html.append('      <td style="text-align: center">-</td>')
        else:
            free_class = 'free-yes' if row['free'] else 'free-no'
            free_text = '☑' if row['free'] else '☐'
            html.append(f'      <td class="{free_class}">{free_text}</td>')
            
        constraint = row['constraint'] if row['constraint'] else ''
        html.append(f'      <td class="constraint">{constraint}</td>')
        html.append('    </tr>')
    html.append('  </tbody>')
    html.append('</table>')
    
    # Add JavaScript for collapsible functionality
    if is_collapsible:
        html.append('<script>')
        html.append('function toggleTable() {')
        html.append('  var table = document.getElementById("paramTable");')
        html.append('  var btn = document.querySelector(".fantasylab-toggle-btn");')
        html.append('  if (table.classList.contains("expanded")) {')
        html.append('    table.classList.remove("expanded");')
        html.append(f'    btn.innerHTML = "▼ Show All {num_params} Parameters";')
        html.append('  } else {')
        html.append('    table.classList.add("expanded");')
        html.append(f'    btn.innerHTML = "▲ Hide Parameters";')
        html.append('  }')
        html.append('}')
        html.append('</script>')
    
    return '\n'.join(html)


def show_fluxes(model, fluxes=None, format='auto', sig_digits=2):
    """
    Display a summary table of line fluxes derived from the model.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    fluxes : dict, optional
        Pre-computed fluxes to display. If None, extracts from model.
    format : {'auto', 'text', 'html'}
        Output format.
    sig_digits : int, optional
        Number of significant figures for display (default: 2).
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    # Extract fluxes if not provided
    if fluxes is None:
        fluxes = extract_fluxes(model)
    
    if not fluxes:
        print("No line fluxes found in model.")
        return
    
    # Prepare data for display
    data = []
    for name, flux in fluxes.items():
        # Format value and uncertainty
        val_str = str(flux) # Flux.__repr__ handles formatting well, but let's use sciformat for consistency if needed
        # Actually Flux.__repr__ is "val ± err" or "val [lo, hi]"
        # But for table we might want separate columns or consistent formatting
        
        # Let's use sciformat explicitly to match style
        if flux.method == 'samples':
            # Asymmetric
            delta_lo = flux.value - flux.lolim if flux.lolim is not None else None
            delta_hi = flux.uplim - flux.value if flux.uplim is not None else None
            # Show errors if at least one side is valid (NaN indicates boundary)
            delta_lo_valid = delta_lo is not None and delta_lo != 0
            delta_hi_valid = delta_hi is not None and delta_hi != 0
            if delta_lo_valid or delta_hi_valid:
                val_fmt = format_value(flux.value, (delta_lo, delta_hi), output=format, sig_digits=sig_digits, use_dollars=False)
            else:
                val_fmt = format_value(flux.value, None, output=format, sig_digits=sig_digits, use_dollars=False)
        elif flux.std is not None:
            val_fmt = format_value(flux.value, flux.std, output=format, sig_digits=sig_digits, use_dollars=False)
        else:
            val_fmt = format_value(flux.value, None, output=format, sig_digits=sig_digits, use_dollars=False)
            
        data.append({
            'name': name,
            'flux': val_fmt,
            'raw_flux': flux
        })
    
    if format == 'html':
        # Build HTML table
        html = ['<style>']
        html.append('.fantasylab-flux-table { border-collapse: collapse; font-family: Arial, sans-serif; font-size: 13px; }')
        html.append('.fantasylab-flux-table th { background-color: #e8e8e8; border: 1px solid #ccc; padding: 8px 10px; text-align: left; }')
        html.append('.fantasylab-flux-table td { border: 1px solid #ddd; padding: 6px 10px; }')
        html.append('</style>')
        html.append('<table class="fantasylab-flux-table">')
        html.append('<thead><tr><th>Line Name</th><th>Flux</th></tr></thead>')
        html.append('<tbody>')
        for row in data:
            html.append(f'<tr><td>{row["name"]}</td><td>{row["flux"]}</td></tr>')
        html.append('</tbody></table>')
        
        try:
            from IPython.display import display, HTML
            display(HTML('\n'.join(html)))
        except:
            print('\n'.join(html))
            
    else:
        # Text table
        headers = ['Line Name', 'Flux']
        widths = {h: len(h) for h in headers}
        for row in data:
            widths['Line Name'] = max(widths['Line Name'], len(row['name']))
            widths['Flux'] = max(widths['Flux'], len(row['flux']))
            
        fmt = f"{{:<{widths['Line Name']}}}  {{:<{widths['Flux']}}}"
        
        print(fmt.format(*headers))
        print("=" * (sum(widths.values()) + 2))
        for row in data:
            print(fmt.format(row['name'], row['flux']))


def show(model, format='auto', sig_digits=2):
    """
    Display formatted summary of model parameters and structure.
    
    Shows component structure, parameter values, uncertainties (if available),
    units (if present), free/fixed/tied status, and constraints (bounds or ties).
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect (single or compound).
    format : {'auto', 'text', 'html'}
        Output format:
        - 'auto': Detect environment (HTML in notebook, text in terminal)
        - 'text': Plain text table
        - 'html': HTML table (for notebooks)
    sig_digits : int, optional
        Number of significant figures for display (default: 2).
        
    Examples
    --------
    >>> from astropy.modeling import models
    >>> m1 = models.PowerLaw1D(1, 5000, 1.5, name='continuum')
    >>> m2 = models.Gaussian1D(10, 5000, 50, name='line')
    >>> compound = m1 + m2
    >>> 
    >>> # Set constraints
    >>> compound.x_0_0.fixed = True
    >>> compound.amplitude_1.bounds = (0, 100)
    >>> 
    >>> # Display
    >>> show(compound)  # Auto-detects environment
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    # Collect data with appropriate output format
    data, has_units, has_uncertainties, has_fluxes = _collect_model_data(model, output_format=format, sig_digits=sig_digits)
    
    # Build model expression
    model_expr = _build_model_expression(model)
    
    # Render
    if format == 'html':
        output = _render_html(data, has_units, has_uncertainties, has_fluxes, model_expr)
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        output = _render_text(data, has_units, has_uncertainties, has_fluxes, model_expr)
        print(output)


def show_model_lineflux(model, fluxes=None, format='auto', sig_digits=2):
    """
    Display model parameters with derived line fluxes.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    fluxes : dict, optional
        Pre-computed fluxes to display. If None, extracts from model.
    format : {'auto', 'text', 'html'}
        Output format.
    sig_digits : int, optional
        Number of significant figures for display (default: 2).
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    # Extract fluxes if not provided
    if fluxes is None:
        fluxes = extract_fluxes(model)
    
    # Collect data with fluxes (but don't show flux column, only flux rows)
    data, has_units, has_uncertainties, has_fluxes = _collect_model_data(model, output_format=format, fluxes=fluxes, show_flux_column=False, sig_digits=sig_digits)
    
    # Build model expression
    model_expr = _build_model_expression(model)
    
    # Render
    if format == 'html':
        output = _render_html(data, has_units, has_uncertainties, has_fluxes, model_expr)
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        output = _render_text(data, has_units, has_uncertainties, has_fluxes, model_expr)
        print(output)


def show_free(model, format='auto'):
    """
    Display only free (fittable) parameters.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    format : {'auto', 'text', 'html'}
        Output format.
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    data, has_units, has_uncertainties, _ = _collect_model_data(model, output_format=format)
    
    # Filter to free parameters only
    data = [row for row in data if row['free']]
    
    if not data:
        print("No free parameters.")
        return
    
    # Build model expression
    model_expr = _build_model_expression(model)
    
    # Render
    if format == 'html':
        output = _render_html(data, has_units, has_uncertainties, model_expr)
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        output = _render_text(data, has_units, has_uncertainties, model_expr)
        print(output)


def show_tied(model, format='auto'):
    """
    Display only tied parameters with their constraints.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    format : {'auto', 'text', 'html'}
        Output format.
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    data, has_units, has_uncertainties, _ = _collect_model_data(model, output_format=format)
    
    # Filter to tied parameters only
    data = [row for row in data if not row['free'] and '[tied]' in row['constraint'] or '→' in row['constraint']]
    
    if not data:
        print("No tied parameters.")
        return
    
    # Build model expression
    model_expr = _build_model_expression(model)
    
    # Render
    if format == 'html':
        output = _render_html(data, has_units, has_uncertainties, model_expr)
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        output = _render_text(data, has_units, has_uncertainties, model_expr)
        print(output)


def show_bounds(model, format='auto'):
    """
    Display only parameters with bounds constraints.
    
    Parameters
    ----------
    model : astropy.modeling.Model
        Model to inspect.
    format : {'auto', 'text', 'html'}
        Output format.
    """
    # Determine format first
    if format == 'auto':
        format = 'html' if _detect_environment() == 'jupyter' else 'text'
    
    data, has_units, has_uncertainties, _ = _collect_model_data(model, output_format=format)
    
    # Filter to parameters with bounds
    data = [row for row in data if row['constraint'] and '[' in row['constraint'] and row['free']]
    
    if not data:
        print("No bounded parameters.")
        return
    
    # Build model expression
    model_expr = _build_model_expression(model)
    
    # Render
    if format == 'html':
        output = _render_html(data, has_units, has_uncertainties, model_expr)
        try:
            from IPython.display import display, HTML
            display(HTML(output))
        except:
            print(output)
    else:
        output = _render_text(data, has_units, has_uncertainties, model_expr)
        print(output)


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
    >>> n_data = len(wavelength)
    >>> dof = n_data - n_free
    >>> print(f"Degrees of freedom: {dof}")
    """
    count = 0
    for pname in model.param_names:
        param = getattr(model, pname)
        is_fixed = getattr(param, 'fixed', False)
        is_tied = getattr(param, 'tied', False) is not False
        if not is_fixed and not is_tied:
            count += 1
    return count
