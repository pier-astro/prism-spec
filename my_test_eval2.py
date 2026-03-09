import numpy as np
from astropy.modeling import models, Fittable1DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter

class MyModel(Fittable1DModel):
    a = Parameter(default=1.0)
    
    def __init__(self, *args, **kwargs):
        self.my_attr = kwargs.pop('my_attr', 1.0)
        super().__init__(*args, **kwargs)

    def evaluate(self, x, a):
        return x * a * self.my_attr

m = MyModel(a=2.0, my_attr=5.0)

fitter = LevMarLSQFitter()
x = np.linspace(0, 10, 100)
y = x * 3.0 * 5.0

m_fit = fitter(m, x, y)
print(m_fit.a)
print(m_fit.my_attr)
