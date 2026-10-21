"""Optional extensions for Astropy models; importing does not install them.

Use ``enable_model_display()`` to opt into enhanced Astropy display, or call
``show_model(model)`` directly without modifying Astropy classes. ``tie`` adapts
component-local callables to Astropy compound-model constraints.
"""
import inspect
import re
from astropy.modeling import CompoundModel
from ..display import *
from ..display import _format_value, _format_unit
from ..composition import _is_compound
from ..models.components import ModelComponents
from ..display import __all__ as _display_exports

__all__ = ['install_model_io_methods', 'install_pipe_patch', 'tie', 'enable_model_display', 'disable_model_display', *_display_exports]
_PATCHED = False
_ORIGINAL_METHODS = {}
_MISSING = object()

def tie(submodel_name, func):
    """
    Create a tie function for parameter constraints in compound models.

    Use this adapter for arbitrary callable constraints and native Astropy models.
    Prism line parameters also support declarative ``parameter.link`` expressions.

    Parameters
    ----------
    submodel_name : str
        Name of the submodel containing the parameter to reference.
    func : callable
        Function taking submodel, returns tied value.

    Returns
    -------
    callable
        Tie function for use with model.param.tied = ...
    """
    def _tie(model):
        if isinstance(model, CompoundModel) or hasattr(model, 'submodel_names'):
            if submodel_name not in model.submodel_names:
                raise KeyError(
                    f"Submodel '{submodel_name}' not found. "
                    f"Available: {model.submodel_names}"
                )
            sub = model[submodel_name]
        else:
            sub = model
        try:
            return func(sub)
        except AttributeError as e:
            params = list(getattr(sub, 'param_names', []))
            raise AttributeError(
                f"Tie error: {e}. Available params in '{submodel_name}': {params}"
            ) from e
    return _tie


def _resolve_component_index(model, submodel_name):
    if not _is_compound(model):
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


def _patched_repr(self):
    values = []
    for name in self.param_names[:4]:
        param = getattr(self, name)
        value = _format_value(getattr(param, 'value', param))
        unit = _format_unit(getattr(param, 'unit', None))
        if unit != '—':
            values.append(f"{name}={value} {unit}")
        else:
            values.append(f"{name}={value}")

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
    # Snapshot every class before changing any inherited method.
    for cls in classes:
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


def _from_astropy(model, cls):
    """Convert supported Astropy expression trees into native Prism nodes."""
    from astropy.modeling import Model as AstropyModel
    from ..composition import _OPERATIONS
    from ..operators.matop import LinearOperatorModel
    if isinstance(model, cls):
        return model
    if isinstance(model, LinearOperatorModel):
        raise TypeError('Apply a linear operator with source | operator.')
    if isinstance(model, CompoundModel):
        if model.op in _OPERATIONS:
            return cls(op=model.op, left=cls.from_astropy(model.left),
                       right=cls.from_astropy(model.right), name=model.name)
        if model.op == '|' and isinstance(model.right, LinearOperatorModel):
            return cls(op='|', left=cls.from_astropy(model.left), right=model.right, name=model.name)
        raise ValueError(f'Unsupported Astropy composition: {model.op!r}.')
    if not isinstance(model, AstropyModel) or model.n_outputs != 1 or model.n_inputs not in (1, 2):
        raise TypeError('Adapt a one-output Astropy model with one or two inputs.')
    return cls(component=model)


def install_model_io_methods():
    """Install class-level ``to_fits`` / ``save`` on Astropy models (idempotent)."""
    import numpy as np
    from astropy.modeling.core import Model as AstropyModel
    from ..io import model_to_fits

    if getattr(AstropyModel, '_prism_io_installed', False):
        return

    def _to_fits(self, filename, overwrite=False,
                 parfilter=None, std=True, model=True, operator_file=None, **custom_exts):
        return model_to_fits(
            self,
            filename,
            overwrite=overwrite,
            parfilter=parfilter,
            std=std,
            model_def=model,
            operator_file=operator_file,
            **custom_exts,
        )

    AstropyModel.to_fits = _to_fits
    AstropyModel.save = _to_fits

    def _model_covariance_getter(self):
        return getattr(self, '_param_cov', None)

    def _model_covariance_setter(self, value):
        if value is None:
            if hasattr(self, '_param_cov'):
                del self._param_cov
        else:
            self._param_cov = np.asarray(value)

    AstropyModel.covariance = property(_model_covariance_getter, _model_covariance_setter)
    AstropyModel._prism_io_installed = True



_ORIGINAL_EVALUATE = None
_ORIGINAL_STATIC_EVALUATE = None
_ORIGINAL_FIT_DERIV = None


def install_pipe_patch():
    """Install Prism's opt-in ``CompoundModel`` pipe patch exactly once."""
    from ..operators.matop import LinearOperatorModel, is_linear_operator_pipe

    global _ORIGINAL_EVALUATE, _ORIGINAL_STATIC_EVALUATE, _ORIGINAL_FIT_DERIV

    if getattr(CompoundModel, '_prism_linear_operator_patch', False):
        return

    _ORIGINAL_EVALUATE = CompoundModel._evaluate
    _ORIGINAL_STATIC_EVALUATE = CompoundModel.evaluate
    _ORIGINAL_FIT_DERIV = CompoundModel.fit_deriv

    def _patched_evaluate(self, *args, **kw):
        if is_linear_operator_pipe(self):
            left_inputs = args[: self.left.n_inputs]
            leftval = self.left(*left_inputs, **kw) if not isinstance(self.right, LinearOperatorModel) or self.right._input_matches(left_inputs[0]) else None
            right_params = []
            for name in self.right.param_names:
                parameter = getattr(self.right, name)
                right_params.append(parameter.quantity if parameter.unit is not None else parameter.value)
            return self.right._prism_pipe_evaluate(
                leftval,
                left_inputs,
                right_params,
                left_model=self.left,
                **kw,
            )
        return _ORIGINAL_EVALUATE(self, *args, **kw)

    def _patched_static_evaluate(self, *args, **kw):
        if is_linear_operator_pipe(self):
            args, kw = self._get_kwarg_model_parameters_as_positional(args, kw)
            left_inputs = self._get_left_inputs_from_args(args)
            left_params = self._get_left_params_from_args(args)
            leftval = self.left.evaluate(*left_inputs, *left_params) if not isinstance(self.right, LinearOperatorModel) or self.right._input_matches(left_inputs[0]) else None
            right_params = self._get_right_params_from_args(args)
            return self.right._prism_pipe_evaluate(
                leftval,
                left_inputs,
                right_params,
                left_model=self.left,
                left_params=left_params,
                **kw,
            )
        return _ORIGINAL_STATIC_EVALUATE(self, *args, **kw)

    @property
    def _patched_fit_deriv(self):
        if is_linear_operator_pipe(self):
            if self.left.fit_deriv is None:
                return None

            def _calc(*args, **kwargs):
                args, kwargs = self._get_kwarg_model_parameters_as_positional(args, kwargs)
                left_inputs = self._get_left_inputs_from_args(args)
                left_params = self._get_left_params_from_args(args)
                right_params = self._get_right_params_from_args(args)
                left_deriv = self.left.fit_deriv(*left_inputs, *left_params) if not isinstance(self.right, LinearOperatorModel) or self.right._input_matches(left_inputs[0]) else None
                return self.right._prism_pipe_fit_deriv(
                    left_deriv,
                    left_inputs,
                    left_params,
                    right_params,
                    self.left,
                )

            return _calc

        return _ORIGINAL_FIT_DERIV.fget(self)

    CompoundModel._evaluate = _patched_evaluate
    CompoundModel.evaluate = _patched_static_evaluate
    CompoundModel.fit_deriv = _patched_fit_deriv
    CompoundModel._prism_linear_operator_patch = True




# Fitters adapt Astropy model/parameter conventions, including 2D fits.
from .astropyfitters import *
from .astropyfitters import __all__ as _fitter_exports
__all__ += _fitter_exports
