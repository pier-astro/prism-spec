# Spectral lines

`GaussianLine`, `LorentzianLine` and `VoigtLine` describe individual lines.
`GaussianLines`, `LorentzianLines` and `VoigtLines` describe templates whose
members share kinematics and an amplitude scaled by tabulated weights.
The six models remain Astropy models; their numerical profile functions are
independent of Astropy parameter objects.

## Coordinates, amplitudes and widths

Positions and evaluation retain the native spectral unit. Length uses
`position * (1 + redshift) * exp(offset / c)` for the center; energy and
frequency use the inverse factor. The domain is inferred from the position
unit, not selected with a `domain` constructor argument.

Amplitudes are peak parameters: a template member has effective peak
`amplitude * weight / (1 + redshift)`. Weights therefore specify relative
peaks, not necessarily equal integrated-flux ratios. Coordinate-unit conversion
on evaluation does not redefine the model's spectral-density convention.
Explicit table-domain conversion utilities handle the associated weight changes.

Intrinsic widths are FWHM in velocity units. `VoigtLine(s)` keeps separate
Gaussian and Lorentzian FWHM parameters. The numerical Gaussian sigma and
Lorentzian half-width are derived from those parameters and each shifted center.
`instfwhm` adds a Gaussian instrumental width; a Lorentzian with nonzero
instrumental width is evaluated as a Voigt profile. General matrix responses
remain in `prism.modeling.operators`, separate from this Gaussian prescription.

## Units

`velocity_unit` selects the unit of `offset`, `fwhm`, `fwhm_G`, and `fwhm_L`.
It defaults to km/s. Numbers use the selected unit; Quantity inputs and
Quantity bounds are converted at construction. Omitted parameters retain
their physical defaults. Values, numeric bounds and attached errors use the
selected unit; covariance entries use the corresponding products of units.

```python
import astropy.units as u
from prism.modeling.models.lines import GaussianLine, GaussianLines

line = GaussianLine(position=5007*u.AA, fwhm=300*u.km/u.s,
                    velocity_unit=u.m/u.s)
assert line.fwhm.quantity == 300_000*u.m/u.s
```

The profile calculation uses km/s with a prepared conversion factor.
Construct a new model to change `velocity_unit`: replacing unit attributes
does not convert attached bounds, errors or covariance. Links between components
using different numeric units must express the conversion explicitly.

Numeric `instfwhm` is always in km/s, independently of `velocity_unit`.
A scalar velocity Quantity is accepted. A numeric two-column table contains
native spectral positions and FWHM in km/s. Instrumental inputs must be finite
and widths non-negative; an instrumental table must have distinct positions.

## Constructing templates

`from_arrays`, `from_csv`, and `from_templates` converge on the same table
preparation and parameter factory. A table contains `name`, `position`, and
optional `weight` columns. Repeated names with explicit weights form one
template; without weights, names are made unique and amplitudes independent.
The first appearance of each name determines parameter order.

```python
group = GaussianLines.from_arrays(
    names=['doublet', 'doublet'], pos=[4959., 5007.]*u.AA,
    weights=[0.33, 1.0], amplitude=2., fwhm=300.,
    bounds={'amplitude': (0, None), 'fwhm': (30, 1000)},
)
```

`from_templates` accepts Astropy tables, DataFrames and mappings. Plain
positions require `position_unit`; Quantity columns retain their native unit.
ECSV files carry units; plain CSV requires an explicit `position_unit`.
Positions and weights must be finite. Supplied air/vacuum conventions are not
silently inferred or changed; explicit conversion utilities remain in `tables`.
The current global wavelength selection is applied during group construction.

The class names and parameter names remain available to the existing fitters,
link/freeze/bounds operations and serialization. `velocity_unit` is saved with
the model; old definitions lacking this field retain km/s behavior.

## Analytic measurements

`line.flux` returns a `Metric`; `group.flux` returns a DataFrame indexed by
template with `value`, `std`, `lolim`, `uplim`, and `unit`. Flux is integrated
analytically over the whole profile. Quantity amplitudes produce the amplitude
unit times the native spectral unit. Plain amplitudes retain numeric results.

Uncertainty propagation uses the local free-parameter covariance in Astropy
order. Without it, attached parameter errors are treated as independent.
Missing uncertainty is NaN; a known zero error remains zero. Temporary parameter
changes are restored even if a measurement fails. Stored amplitude/width limits
produce parameter envelopes, not joint confidence intervals. Cross-component
uncertainties and measurements of selected sums belong to `line_analysis`.

`eqw(continuum=..., method='constant-continuum')` divides analytic flux by a
fixed continuum. Continuum uncertainty is not propagated. With
`method='integration'`, it integrates line/continuum on an explicit grid or
an automatically inferred finite window and does not estimate errors. Automatic
group windows contain all members and include instrumental broadening. Voigt
width estimates prepare the grid; they do not replace measured FWHM crossings.
Zero-width profiles require an explicit integration grid or window.

An omitted continuum preserves the legacy self-profile normalization, using
the profile at the center (mean transition position for a template). This is
an effective width of that profile, not an independently measured continuum EW.
For physical EWs, supply the continuum explicitly. Finite-window profile
moments and grouped kinematics remain separate from analytic infinite-profile flux.

## Implementation boundaries

- `profiles.py`: numerical profiles, parameter conversion, analytic Jacobians,
  flux formulas and width estimates. Lorentzian derivatives reuse the Voigt
  Jacobian with zero intrinsic Gaussian width, including the Gaussian boundary.
- `base.py`: common Astropy unit/instrument adapter and shared profile calls.
- `single.py`: individual-line descriptors and a common single-line adapter.
- `groups.py`: table constructors, parameter factory and explicit template sums.
- `_measurements.py`: one analytic flux/error/EW implementation used by both
  adapters; no dataset-specific pipeline or inference algorithm.
- `tables.py`: table validation, conversions and resource utilities.

A profile definition supplies evaluation, native-parameter conversion, analytic
derivatives, an integral and a width estimate for automatic grids. The built-in
width estimate covers Gaussian/Voigt parameters; other profiles override it.
Single and grouped models bind the same functions;
adding one does not require another uncertainty or EW implementation. Astropy
parameter descriptor generation remains confined to the group constructor.

## Reading the implementation

The built-in classes in `single.py` and `groups.py` show the evaluation sequence
explicitly: velocity/unit preparation, observed center, effective profile
parameters, named Gaussian or Voigt evaluation, and component accumulation.
`profiles.py` holds the shared physical expressions and analytic derivatives;
`base.py` handles common units, instrumental setup, and measurement adapters.
A Lorentzian with Gaussian instrumental broadening is evaluated as a Voigt.

## Parameter links

Use `component_b.fwhm.link = component_a.fwhm` or arithmetic expressions such as
`component_b.amplitude.link = component_a.amplitude / 2.98`. Assemble all referenced
components before fitting or saving. Copies of the complete model preserve links;
renaming components does not redirect them. Declarative links are restored by
`load_model()`. See the fitting README for units, arithmetic and callable limits.
