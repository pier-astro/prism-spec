import numpy as np
from astropy.modeling import models, fitting
import prism  # this triggers the dynamic patch
from prism.modeling.fitting.extension import _wrap_fitter_call

class TestFitterExt:
    def test_single_fit(self):
        fitter = fitting.TRFLSQFitter()
        model = models.Gaussian1D(amplitude=10, mean=5, stddev=1)
        
        x = np.linspace(0, 10, 100)
        y = model(x) + np.random.normal(0, 0.1, 100)
        yerr = np.full(100, 0.1)

        # Before we assert, let's just make sure it runs and returns model
        fitted = fitter(model, x, y, yerr=yerr)
        print("Single fit successful")
        assert hasattr(fitted, 'amplitude')

    def test_multifit(self):
        fitter = fitting.TRFLSQFitter()
        model = models.Gaussian1D(amplitude=10, mean=5, stddev=1)
        
        x = np.linspace(0, 10, 100)
        y_cube = np.random.rand(5, 5, 100)
        
        result = fitter.multifit(model, x, y_cube, spectral_axis=-1)
        print("Multifit successful")
        assert result.shape == (5, 5)
        
if __name__ == '__main__':
    t = TestFitterExt()
    t.test_single_fit()
    t.test_multifit()
