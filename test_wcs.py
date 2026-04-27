from astropy.wcs import WCS
import astropy.units as u
import numpy as np

w = WCS(naxis=3)
w.wcs.crval = [10, 20, 4000]
w.wcs.cdelt = [1, 1, 10]
w.wcs.crpix = [1, 1, 1]
w.wcs.cunit = ['deg', 'deg', 'Angstrom']

z = np.linspace(5000, 6000, 100)
w.wcs.crval[2] = z[0]
w.wcs.cdelt[2] = z[1] - z[0]
w.wcs.crpix[2] = 1.0
w.wcs.cunit[2] = u.AA.to_string()

print(w.wcs.crval)
print(w.wcs.cdelt)
print(w.wcs.cunit)
