"""Projection must use the physical profile span, not arbitrary QR columns."""
import numpy as np
from prism.experimental.separable import _project_from_linear_span


def test_degenerate_profiles_keep_orthogonal_kinematic_direction():
    profile=np.array([1.,0.,0.,0.])
    design=np.column_stack([profile,2.*profile])
    jacobian=np.array([[1.,0.],[1.,1.],[0.,0.],[0.,1.]])
    expected=jacobian.copy();expected[0]=0.
    np.testing.assert_allclose(_project_from_linear_span(design,jacobian),expected,atol=1e-14)


def test_zero_profile_block_removes_no_kinematic_information():
    design=np.zeros((4,3));jacobian=np.arange(8.).reshape(4,2)
    np.testing.assert_array_equal(_project_from_linear_span(design,jacobian),jacobian)
