# Spectral lines

`lines` contains numerical profiles, transition tables and native weighted line
components. `modeling.Component`, parameter graphs and fitting remain generic.
The historical Astropy line subclasses have been removed. Existing external
Astropy models enter the generic core through `Model.from_astropy`.

```python
from prism.modeling.lines import LineSet, GaussianProfile
from prism.modeling.lines.measurements import select_line

line = select_line(best, "OIII", components="narrow")
result = line.measure()
uncertainty = line.sample(n_samples=1000, random_state=41)
```

## Selected transition contract

A selection retains the complete parent model for parameter links and covariance.
Each contributing member supplies its live profile, rest position, coordinate
and output units, and participating parameters. Selection resolves component
identities and template membership once; parameter values are read at evaluation.
Rebuild the selection after replacing components or modifying table membership.

Measurements integrate the sampled profile; they do not dispatch on Gaussian,
Voigt or Astropy classes. Custom profiles use an explicit `x` grid or `window`.
Known profiles may infer a window from the observed center and effective width;
that width sets the integration extent, not the reported numerical FWHM.
Weights multiply the chosen scale argument exactly once. Instrumental broadening
inside a profile is retained; selecting a source component does not deconvolve it.

The selected member reads the same profile and live parameters as fitting.
Sampling uses those parameter identities and the parent graph, preserving
links and correlated uncertainties. See `measurements/README.md` for units,
equivalent widths, missing cube fits and uncertainty assumptions.

## Import migration

- `modeling.line_analysis` → `modeling.lines.measurements`.
- `modeling.line_component` → `modeling.lines.component`.
- Numerical `modeling.models.lines.profiles` and transition `tables` →
  `modeling.lines.profiles` and `modeling.lines.tables`.
- `from prism.modeling import LineSet, select_line` remains available.
- Historical `GaussianLine`/`GaussianLines`, Lorentzian and Voigt Astropy classes
  and their YAML tags are removed. Use `Component` or `LineSet` with a
  configured profile. Old subclass sessions require reconstruction with the
  previous version; new native sessions retain their profile recipes.
- Native line inputs and parameter values are numeric in explicitly chosen
  units. Convert quantities before construction/evaluation. The table records
  spectral units; profile velocities are km/s. Automatic Astropy quantity
  stripping and its per-model `velocity_unit` option are not native APIs.
- Ready-made native sets expose `flux()` for total analytic integrals.
  Selected profiles use `.measure()` for finite numerical windows and equivalent
  widths; custom profiles require an explicit grid/window.
- `infer_window` no longer has the legacy zero-width `min_width` fallback;
  supply an explicit grid/window for a zero-width or unknown profile.

## Ready-made sets and custom profiles

```python
from prism.modeling.lines import GaussianLines, LorentzianLines, VoigtLines

narrow = GaussianLines("oiii", fwhm=300., instfwhm=120., name="narrow")
broad = LorentzianLines(linetable, fwhm=1500., instfwhm=120., name="broad")
voigt = VoigtLines(linetable, fwhm_G=300., fwhm_L=100., name="voigt")
```

These are thin native `LineSet` subclasses, with shared kinematics and
one peak amplitude per template. They prepare the same profiles, derivatives
and parameter maps as an explicit custom set; no evaluation layer is added.
Tables/mappings carry spectral units. Bare preset names search the configured
local directory then packaged defaults; paths are read exactly as given. `redshift` starts fixed. `bounds={"amplitude": (0, 100)}`
applies to every template amplitude; individual parameters remain editable.
Instrumental widths may be scalar or tabulated at observed positions. Gaussian
broadening combines in quadrature only with Gaussian widths; a broadened
Lorentzian is a Voigt. The peak convention does not imply flux preservation.

Use `LineSet(profile, lines, shared, parameters, ...)` when specifying a custom
profile or parameter sharing. This replaces the former native `LineComponent`
name. New sessions emit `!prism.LineSet`; the previous native tag remains readable.
The ready-made constructors are unrelated to the removed Astropy subclasses.

## Total fluxes and uncertainties

```python
narrow = GaussianLines(["hydrogen", "oiii"],
                       window=(4800., 6800.), fwhm=300., name="narrow")
narrow.flux()  # table: name, rest position, weight, total flux
select_line(result, "OIII", components="narrow").flux()  # Metric(value, std)
```

`window` filters rest transition positions in the table's unit, before velocity
shifts/redshift. It is unrelated to a numerical integration window. The table
reports every transition; weights multiply its peak exactly once. Analytic
integrals include profile-local broadening and precede any response operator.
No flux normalization is imposed on generic `LineSet`: custom profiles may use
peaks, integrated fluxes or another scale argument, shared or per template.

For a fitted result, selected analytic fluxes propagate the complete covariance
by the delta method, including links and cross-observation terms. Without a
covariance, `std` is `None`. Selecting a `BootstrapResult` instead propagates the
actual successful refits and reports their spread and central interval. The
nominal value remains the integral at the supplied model's nominal parameters.
These intervals are not posterior probabilities. `.measure(window=...)` always
integrates the specified finite window, including for built-in profiles.

## Inspect, export and customize presets

```python
from prism.modeling import lines

lines.set_directory("lines")
lines.export(["hydrogen", "helium", "oiii"], directory="lines",
             window=(4800., 7000.), overwrite=False)
table = lines.load("oiii")
# Edit the table in Python or open the exported ECSV in a text editor.
table.write("lines/custom.ecsv", overwrite=True)
narrow = lines.GaussianLines("custom", fwhm=300.)
```

`load` returns an independent, editable QTable. Its standard `write` method can
write ECSV or another Astropy-supported format. ECSV preserves spectral units.
`export(all=True, directory="lines", window=...)` writes every packaged preset;
export always reads packaged defaults, even if a local override is configured.
Existing destinations raise before any files are written unless overwrite=True.
Empty filtered presets are exported as empty tables. The window uses rest
positions in the table's unit and never changes the source files.

`set_directory` is explicit session configuration for future name lookups. The
directory may not yet exist and its path is resolved when configured. A bare
name first uses a local ECSV if present, otherwise the packaged preset; malformed
local files raise rather than silently falling back. Strings with extensions
or path separators, and all Path objects, are explicit paths without fallback.
`set_directory(None)` restores packaged-only lookup. Export does not configure
the directory; existing models retain their prepared tables after any change.

This replaces `setup_local_lines` and `set_wavelength_range`: no global wavelength
filter is applied. Use window explicitly for model construction or table loading.
