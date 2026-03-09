import numpy as np
import pandas as pd
from prism.modeling.models.lines import GaussianLines, set_wavelength_range

def test_filtering():
    # Helper to test filtering
    names = ['Halpha', 'Hbeta', 'OIII_5007', 'OIII_4959']
    pos = [6562.8, 4861.3, 5006.8, 4958.9]
    
    # 1. No filtering (default)
    print("Test 1: Default range (0 to inf)")
    model = GaussianLines.from_arrays(names, pos)
    print(f"Templates in model: {model._templates}")
    assert len(model._templates) == 4

    # 2. Filter some lines
    print("\nTest 2: Set range to [4000, 6000]")
    set_wavelength_range(wmin=4000, wmax=6000)
    model = GaussianLines.from_arrays(names, pos)
    print(f"Templates in model: {model._templates}")
    assert 'halpha' not in [t.lower() for t in model._templates]
    assert len(model._templates) == 3

    # 3. Filter more strictly
    print("\nTest 3: Set range to [6000, 7000]")
    set_wavelength_range(wmin=6000, wmax=7000)
    model = GaussianLines.from_arrays(names, pos)
    print(f"Templates in model: {model._templates}")
    assert len(model._templates) == 1
    assert model._templates[0].lower() == 'halpha'

    # 4. Check error on empty
    print("\nTest 4: Empty range [1000, 2000]")
    set_wavelength_range(wmin=1000, wmax=2000)
    try:
        GaussianLines.from_arrays(names, pos)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Reset
    set_wavelength_range(wmin=0, wmax=np.inf)

if __name__ == "__main__":
    test_filtering()
