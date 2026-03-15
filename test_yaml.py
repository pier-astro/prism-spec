import numpy as np
import astropy.units as u
from prism.modeling.models import GaussianLine
from astropy.io.misc import yaml

model = GaussianLine(amplitude=10, mean=5000, fwhm=10)
print(yaml.dump(model))
