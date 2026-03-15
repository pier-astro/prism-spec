import numpy as np
import astropy.units as u
from prism.modeling.models import GaussianLine
from astropy.io.misc import yaml
import prism.modeling.models.serialization

model = GaussianLine(amplitude=10, position=5000, fwhm=10)
s = yaml.dump(model)
print(s)
print(yaml.load(s))
