import numpy as np
from astropy.modeling import models, Fittable1DModel, Parameter

class MyModel(Fittable1DModel):
    a = Parameter(default=1.0)
    
    def __init__(self, *args, **kwargs):
        self.my_attr = kwargs.pop('my_attr', 0.0)
        super().__init__(*args, **kwargs)

    def evaluate(self, x, a):
        # Let's test if evaluate can be an instance method
        return x * a + self.my_attr

m = MyModel(a=2.0, my_attr=5.0)
print(m(10))
