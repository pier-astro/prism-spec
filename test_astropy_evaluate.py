import numpy as np
from astropy.modeling import models, Fittable1DModel, Parameter

class Base(Fittable1DModel):
    def __init__(self, *args, **kwargs):
        self.my_attr = kwargs.pop('my_attr', 1.0)
        super().__init__(*args, **kwargs)

def evaluate(self, x, a):
    return x * a * self.my_attr

def fit_deriv(self, x, a):
    return [x * self.my_attr]

MyDynModel = type('MyDynModel', (Base,), {
    'a': Parameter(default=1.0),
    'evaluate': evaluate,
    'fit_deriv': fit_deriv
})

m = MyDynModel(a=2.0, my_attr=5.0)
print(m(10))
