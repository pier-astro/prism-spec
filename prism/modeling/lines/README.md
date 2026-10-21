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
- Analytic `.flux`/`.ew` properties are replaced by explicit selected-profile
  measurements and sampling. Custom profiles require an explicit grid/window.
- `infer_window` no longer has the legacy zero-width `min_width` fallback;
  supply an explicit grid/window for a zero-width or unknown profile.

## Ready-made sets and custom profiles

```python
from prism.modeling.lines import GaussianLines, LorentzianLines, VoigtLines

narrow = GaussianLines("oiii.ecsv", fwhm=300., instfwhm=120., name="narrow")
broad = LorentzianLines(linetable, fwhm=1500., instfwhm=120., name="broad")
voigt = VoigtLines(linetable, fwhm_G=300., fwhm_L=100., name="voigt")
```

These are native constructors returning `LineSet`, with shared kinematics and
one peak amplitude per template. They prepare the same profiles, derivatives
and parameter maps as an explicit custom set; no evaluation layer is added.
Tables/mappings carry spectral units; relative ECSV paths use the current
line-table directory. `redshift` starts fixed. `bounds={"amplitude": (0, 100)}`
applies to every template amplitude; individual parameters remain editable.
Instrumental widths may be scalar or tabulated at observed positions. Gaussian
broadening combines in quadrature only with Gaussian widths; a broadened
Lorentzian is a Voigt. The peak convention does not imply flux preservation.

Use `LineSet(profile, lines, shared, parameters, ...)` when specifying a custom
profile or parameter sharing. This replaces the former native `LineComponent`
name. New sessions emit `!prism.LineSet`; the previous native tag remains readable.
The ready-made constructors are unrelated to the removed Astropy subclasses.
