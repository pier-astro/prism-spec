import numpy as np
from astropy.modeling import models, fitting
import astropy.modeling.fitting as ast_fit

print(dir(ast_fit.Fitter))

def multifit(self, *args, **kwargs):
    print("Multifit called!")
    return "multifit result"

ast_fit.Fitter.multifit = multifit

fitter = fitting.LevMarLSQFitter()
fitter.multifit()
