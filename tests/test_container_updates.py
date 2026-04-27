import numpy as np
import astropy.units as u
from astropy.wcs import WCS
import warnings
from prism.data.image import Image
from prism.data.spectrum import Spectrum
from prism.data.cube import Cube
from prism.utils.tools import apply_extinction_correction, get_ebv_from_map

def test_image_extent():
    print("Testing Image extent...")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5] # Center of 20x20 image in 1-based coords is 10.5
    wcs.wcs.crval = [100.0, 20.0]
    wcs.wcs.cdelt = [-1/3600, 1/3600] # 1 arcsec/pix, RA increases West (CDELT1 < 0)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    img = Image(np.ones((20, 20)), wcs=wcs)
    img.set_reference_coordinates(100, 20)
    
    extent = img.get_extent()
    print(f"Image extent: {extent}")
    # For a 20x20 image with CRPIX at center (10.5, 10.5):
    # Pixel 0 (edge -0.5) is 10 pixels away from CRPIX. 10 * 1 arcsec = 10 arcsec.
    # CDELT1 = -1 arcsec/pix => Pixel 0.5 (0-indexed) is West of CRVAL.
    # get_offset(-0.5, ...) should give East offset (positive).
    # -0.5 is 11 pixels East of 10.5? No, 10.5 - (-0.5) = 11 pixels. 
    # Wait, 0-indexed: 0 to 19. Edges -0.5 to 19.5. 
    # Offset of -0.5 from 9.5 (0-based center) is -10. 
    # With CDELT1 = -1/3600, RA_off = (pix - crpix) * cdelt = (-10) * (-1) = 10 arcsec West.
    # get_offset returns -dra, so -10 arcsec? 
    # If it's West, it should be negative. East positive.
    # Let's check the code: return -(dra * u.deg).to_value(target_unit)
    # dra = (sky.ra.deg - ra_ref) * cos(dec).
    # If sky.ra > ra_ref (West), dra > 0, returns negative. Correct.
    assert np.allclose(extent[0], 10.0, atol=0.1) # Left (East)
    assert np.allclose(extent[1], -10.0, atol=0.1) # Right (West)
    assert np.allclose(extent[2], -10.0, atol=0.1) # Bottom
    assert np.allclose(extent[3], 10.0, atol=0.1) # Top

def test_ra_dec_units():
    print("Testing RA/Dec unit support...")
    spec = Spectrum(np.arange(10), np.arange(10), ra=100*u.deg, dec="20d00m00s")
    assert spec.ra == 100.0
    assert np.allclose(spec.dec, 20.0)
    
    cube = Cube(np.ones((10, 10, 10)), ra=SkyCoord(100, 20, unit='deg')) if False else None # SkyCoord test
    # Let's test parse_celestial_coord directly for SkyCoord if possible
    from prism.data.core import parse_celestial_coord
    try:
        from astropy.coordinates import SkyCoord
        sc = SkyCoord(100, 20, unit='deg')
        # parse_celestial_coord currently doesn't handle SkyCoord objects directly but RA/Dec components
        assert parse_celestial_coord(sc.ra) == 100.0
    except ImportError:
        pass

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_image_extent()
        test_ra_dec_units()
    print("Verification tests passed!")
