
# -----------------------
# Combined Line Analysis
# -----------------------

import numpy as np

def combined_flux(models, line_name=None, position=None, position_tol=10.0):
    """
    Calculate the total flux of a list of line models, optionally filtering by line name or position.
    
    Parameters
    ----------
    models : list of Fittable1DModel
        List of model instances.
    line_name : str, optional
        Filter components by name (substring match).
    position : float, optional
        Filter components by position (wavelength).
    position_tol : float, optional
        Tolerance for position matching.
        
    Returns
    -------
    float
        Total integrated flux.
    """
    total_flux = 0.0
    for m in models:
        # Tied Models (including Balmer)
        if hasattr(m, 'fluxes') and hasattr(m, '_param_names_list') and hasattr(m, '_positions'):
            fluxes = m.fluxes
            for i, pname in enumerate(m._param_names_list):
                pos = m._positions[i]
                match = True
                if position is not None and abs(pos - position) > position_tol: match = False
                if line_name is not None and line_name not in pname: match = False
                
                if match:
                    total_flux += fluxes[pname]
        
        # FeII Models (Templates)
        elif hasattr(m, 'fluxes') and hasattr(m, '_templates'):
            # Only filter by line_name (template name)
            if position is None:
                fluxes = m.fluxes
                for tmpl in m._templates:
                    match = True
                    if line_name is not None and line_name not in tmpl: match = False
                    if match:
                        total_flux += fluxes[tmpl]

        # Single Models
        elif hasattr(m, 'flux'):
            match = True
            if position is not None:
                m_pos = getattr(m, 'position', None)
                if m_pos is not None:
                    val = m_pos.value if hasattr(m_pos, 'value') else m_pos
                    if abs(val - position) > position_tol: match = False
            
            if line_name is not None:
                if hasattr(m, 'name') and m.name:
                    if line_name not in m.name: match = False
                else:
                    match = False
            
            if match:
                total_flux += m.flux
                
    return total_flux

def combined_profile(models, x, line_name=None, position=None, position_tol=10.0):
    """
    Evaluate the sum of profiles for a list of models, optionally filtering by line name or position.
    """
    total = np.zeros_like(x, dtype=float)
    
    for m in models:
        # Tied Models
        if hasattr(m, '_param_names_list') and hasattr(m, '_positions'):
            shared_params = m._shared_params.keys()
            shared_values = [getattr(m, p).value if hasattr(getattr(m, p), 'value') else getattr(m, p) for p in shared_params]
            
            for i, pname in enumerate(m._param_names_list):
                pos = m._positions[i]
                
                match = True
                if position is not None and abs(pos - position) > position_tol: match = False
                if line_name is not None and line_name not in pname: match = False
                
                if match:
                    amp = getattr(m, pname).value if hasattr(getattr(m, pname), 'value') else getattr(m, pname)
                    
                    if hasattr(m, '_ratios'): # Balmer
                        ratio = m._ratios[i]
                        amp = amp * ratio
                        args = m._profile_args(pos, amp, *shared_values)
                        prof = m._profile_func(x, *args)
                        # Apply mask for Balmer (lambda_BE = 3646.0)
                        prof = np.where(x <= 3646.0, 0.0, prof)
                        total += prof
                    else:
                        args = m._profile_args(pos, amp, *shared_values)
                        total += m._profile_func(x, *args)

        # FeII Models
        elif hasattr(m, '_templates'):
            if position is None:
                shared_params = m._shared_params.keys()
                shared_values = [getattr(m, p).value if hasattr(getattr(m, p), 'value') else getattr(m, p) for p in shared_params]
                
                for tmpl in m._templates:
                    match = True
                    if line_name is not None and line_name not in tmpl: match = False
                    
                    if match:
                        pname = f'amp_{tmpl}'
                        amp = getattr(m, pname).value if hasattr(getattr(m, pname), 'value') else getattr(m, pname)
                        df_tmpl = m._df[m._df['ime'] == tmpl]
                        
                        positions = df_tmpl['wav'].values
                        intensities = df_tmpl['Int'].values
                        
                        for p, inten in zip(positions, intensities):
                            args = m._profile_args(p, amp, inten, *shared_values)
                            total += m._profile_func(x, *args)

        # Single Models
        elif hasattr(m, 'evaluate'): # Fittable1DModel
            match = True
            if position is not None:
                m_pos = getattr(m, 'position', None)
                if m_pos is not None:
                    val = m_pos.value if hasattr(m_pos, 'value') else m_pos
                    if abs(val - position) > position_tol: match = False
            
            if line_name is not None:
                if hasattr(m, 'name') and m.name:
                    if line_name not in m.name: match = False
                else:
                    match = False
            
            if match:
                total += m(x)
                
    return total

def _estimate_grid(models, n_points=10000, padding_factor=5.0, line_name=None, position=None, position_tol=10.0):
    """
    Helper to estimate a wavelength grid covering the selected components.
    """
    if not models:
        return None

    min_x = np.inf
    max_x = -np.inf
    max_width = 0.0
    
    found_any = False
    
    for m in models:
        # Check for matching components to estimate range
        
        # Tied Models
        if hasattr(m, '_param_names_list') and hasattr(m, '_positions'):
            for i, pname in enumerate(m._param_names_list):
                pos = m._positions[i]
                match = True
                if position is not None and abs(pos - position) > position_tol: match = False
                if line_name is not None and line_name not in pname: match = False
                
                if match:
                    found_any = True
                    min_x = min(min_x, pos)
                    max_x = max(max_x, pos)
                    
                    # Width
                    width = 100.0
                    if 'fwhm' in m._shared_params:
                        width = getattr(m, 'fwhm').value
                    elif 'fwhm_G' in m._shared_params:
                        width = max(getattr(m, 'fwhm_G').value, getattr(m, 'fwhm_L').value)
                    
                    width_A = width / 299792.458 * pos
                    max_width = max(max_width, width_A)

        # FeII Models
        elif hasattr(m, '_templates'):
            if position is None:
                for tmpl in m._templates:
                    match = True
                    if line_name is not None and line_name not in tmpl: match = False
                    if match:
                        found_any = True
                        df_tmpl = m._df[m._df['ime'] == tmpl]
                        min_x = min(min_x, df_tmpl['wav'].min())
                        max_x = max(max_x, df_tmpl['wav'].max())
                        max_width = max(max_width, 50.0) # Arbitrary for FeII

        # Single Models
        elif hasattr(m, 'position'):
            match = True
            pos = getattr(m, 'position', None)
            if pos is not None:
                val = pos.value if hasattr(pos, 'value') else pos
                if position is not None and abs(val - position) > position_tol: match = False
            
            if line_name is not None:
                if hasattr(m, 'name') and m.name:
                    if line_name not in m.name: match = False
                else:
                    match = False
            
            if match:
                found_any = True
                if pos is not None:
                    val = pos.value if hasattr(pos, 'value') else pos
                    min_x = min(min_x, val)
                    max_x = max(max_x, val)
                    
                width = 100.0
                if hasattr(m, 'fwhm'): width = m.fwhm.value
                elif hasattr(m, 'fwhm_G'): width = max(m.fwhm_G.value, getattr(m, 'fwhm_L', 0).value)
                
                width_A = width / 299792.458 * val
                max_width = max(max_width, width_A)
    
    if not found_any:
        return None

    if min_x == np.inf: min_x = 4000.0
    if max_x == -np.inf: max_x = 7000.0
    if max_width == 0: max_width = 50.0
    
    start = min_x - padding_factor * max_width
    end = max_x + padding_factor * max_width
    return np.linspace(start, end, n_points)

def combined_fwhm(models, x_grid=None, n_points=10000, padding_factor=5.0, line_name=None, position=None, position_tol=10.0):
    """
    Calculate the FWHM of the combined profile of a list of models numerically.
    """
    if not models:
        return 0.0
        
    if x_grid is None:
        x_grid = _estimate_grid(models, n_points, padding_factor, line_name, position, position_tol)
        if x_grid is None: return 0.0
        
    y = combined_profile(models, x_grid, line_name=line_name, position=position, position_tol=position_tol)
    
    peak_val = np.max(y)
    if peak_val == 0: return 0.0
    
    half_max = peak_val / 2.0
    
    above = y > half_max
    crossings = np.where(np.diff(above))[0]
    
    if len(crossings) < 2:
        return 0.0
        
    x_left_idx = crossings[0]
    x_right_idx = crossings[-1]
    
    def interpolate_x(idx):
        x1, x2 = x_grid[idx], x_grid[idx+1]
        y1, y2 = y[idx], y[idx+1]
        if y2 == y1: return x1
        return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        
    x_left = interpolate_x(x_left_idx)
    x_right = interpolate_x(x_right_idx)
    
    return x_right - x_left

def combined_sigma(models, x_grid=None, n_points=10000, padding_factor=5.0, line_name=None, position=None, position_tol=10.0):
    """
    Calculate the second moment (sigma) of the combined profile numerically.
    
    Parameters
    ----------
    models : list of Fittable1DModel
        List of model instances.
    x_grid : array-like, optional
        Grid to evaluate the profile.
    n_points : int
        Number of points for the generated grid if x_grid is None.
    padding_factor : float
        Factor to extend the grid range beyond the min/max position +/- width.
    line_name : str, optional
        Filter components by name.
    position : float, optional
        Filter components by position.
    position_tol : float, optional
        Tolerance for position matching.
        
    Returns
    -------
    float
        Sigma (standard deviation) of the combined profile in the same units as x (usually Angstroms).
    """
    if not models:
        return 0.0

    if x_grid is None:
        x_grid = _estimate_grid(models, n_points, padding_factor, line_name, position, position_tol)
        if x_grid is None: return 0.0
        
    y = combined_profile(models, x_grid, line_name=line_name, position=position, position_tol=position_tol)
    
    # Numerical integration
    norm = np.trapz(y, x_grid)
    if norm == 0: return 0.0
    
    mean = np.trapz(x_grid * y, x_grid) / norm
    var = np.trapz((x_grid - mean)**2 * y, x_grid) / norm
    
    return np.sqrt(var)

