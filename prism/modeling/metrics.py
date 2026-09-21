"""Small value containers used by modeling and line-analysis results."""

from dataclasses import dataclass

import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class Metric:
    """Measured value with optional uncertainty, limits, and unit."""

    value: object
    std: object = None
    lolim: object = None
    uplim: object = None
    unit: object = None

    def __post_init__(self):
        if self.unit is not None and not isinstance(self.unit, u.UnitBase):
            object.__setattr__(self, 'unit', u.Unit(self.unit))

    @property
    def quantity(self):
        return self.value if self.unit is None else self.value * self.unit

    @property
    def std_quantity(self):
        if self.std is None:
            return None
        return self.std if self.unit is None else self.std * self.unit

    @property
    def lolim_quantity(self):
        if self.lolim is None:
            return None
        return self.lolim if self.unit is None else self.lolim * self.unit

    @property
    def uplim_quantity(self):
        if self.uplim is None:
            return None
        return self.uplim if self.unit is None else self.uplim * self.unit

    def __iter__(self):
        yield self.value
        yield self.std
        yield self.lolim
        yield self.uplim

    def __repr__(self):
        unit_str = '' if self.unit is None else f' {self.unit}'
        if np.asarray(self.value).ndim:
            return f'<Metric: array of shape {np.asarray(self.value).shape}{unit_str}>'
        if self.std is not None:
            return f'{self.value:.4g} +/- {self.std:.2g}{unit_str}'
        return f'{self.value:.4g}{unit_str}'
