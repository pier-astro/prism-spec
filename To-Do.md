- (DONE!!!) IMPORTANT: for the fitters and multifit we are now only minmally expanding the standard features of astropy fitters. However how the code is now the multifit will work only uisng our fitters!
Instead it would be amazing is when importing prims we extend the astropy class of the fitters adding minimally the yerr and the other additional keywords we are adding (minimal code). And then for all of them we add the .multifit attribute that the user has to call explicitly! This way we remove ambiguity and we force to be explicit and more! this will work generaly with any astropy fitter even native!

- (DONE!) WARNING: The fit may be unsuccessful; check: 
    The maximum number of function evaluations is exceeded. [astropy.modeling.fitting]
    Make the limit of evaluatios easily settable. Now it is not clear to me how it can be done without hardcoding the value in the astropy package. If effectively it cannot be done otherwise natively in astropy, can we make in prism easier to do so minimally patching astropy? We are using this approach to extend some functionalities about the fitters, the models and others task in prism, no?

- (DONE!) When computing the flux we should INCLUDE the instfwhm!! We bias the flux measure otherwise since flux is just redistributed and when we include the instfwhm in quadrature we are getting that the real flux is the one from both fwhm and instfwhm! Convolved model instead is safer on this side since redeistribution only we have corrected both the width and the amplitude! Are we behaving correctly?
    Minimally patch this, if needed. Then minimally test that results from rsp(lines) and lines passing instfwhm agree

- (DONE (Somehow...)!) Verify how feasible is to minimally extend the Compound model to support Convolved model and analytic derivatives in astropy

- (DONE !) Add something that instead of the covariance uses the .lolim and .uplim to compute the derived parameters uncertainties.

- (DONE !!) Add the display module and verify how feasible is to add the html `__repr__` to the model object directly to keep things lightweight

- (DONE !!) EQUIVALENT WIDTH handling! Either at line level as .flux and at line_analysis level for the blended line.




- (DONE !!!!!!) Explore if, instead of saving the response models via a matrix in a file, we save it as a smart model with the recipe on how to build the matrix. Building it on the fly should not be a problem... However does it significantly simplify the code?
Alternatively, explore if, if we give up about the analytic derivatives, we can simply use astropy CompoundModel to handle the convolution and the fit. If this is the case we can save a lot of code making maintainability much easier. Just evaluate the cases. Btw, to use the analytic via we always habe istfwhm in the models so we can use that... If yes, do not waste all this effor but let's save a well documented copy of the parts we plan to remove where we explain all the details and the rationale for the change. This will be useful for future reference and to avoid losing the work done.
Use all the context you have in these deep dive in the code.



- (DONE!) Add units handling in the line models and in the line analysis. This should be done in a way that is consistent with the rest of the package and with the astropy models. We should verify how feasible is to add the "unit" in the Parameter object and not add them via dictionaries as we are doing now. This will make things more consistent and easier to handle and astropy "standard".
Fix the unit handling in line models. See how feasible is to add the "unit" in the Parameter object and not add them via dictionaries as we are doing now
ADD UNIT AS COLUMN TO DISPLAY in the display module. Remeber the display module is meant to work also with native astropy models and not only with enhanced astropy models as in prism.

- Make everyting less WAVELENGHT specific. Our routines are quite general! Just some aspects like the line models might strongly rely on the wavelength. In that case we might add a "domain" attribute to the line models to specify the domain of application (e.g. wavelength, frequency, energy) and then handle the specificities in the line models. Outise this domain simply raise a warning/error or NotImplementedError. This will make the package more general and potentially useful for other applications beyond spectroscopy.

Do this transition to astropy poilicy on:
- fitting. Both single fit and multifit
- data
- display
- io and serialization




- Test if saving, loading, display are working with native astropy models.






- Table models and Template models with and without velocity braodening
- Penalized Pixel Fitting as fitter