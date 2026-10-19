"""Independent algebraic checks of additive component extraction."""
import numpy as np
import pytest
from astropy.modeling.models import Const1D,Gaussian1D
from scipy.sparse import eye
from prism.modeling.models.components import get_components
from prism.modeling.operators import LinearOperatorModel


@pytest.mark.parametrize('kind,expected,count',[
    ('division',.5,2),('power',25.,1),('product',50.,4),
])
def test_constant_decomposition_preserves_actual_algebra(kind,expected,count):
    a,b,c,d=[Const1D(value,name=name) for value,name in [(2,'a'),(3,'b'),(4,'c'),(6,'d')]]
    expression={'division':(a+b)/(c+d),'power':(a+b)**Const1D(2),'product':(a+b)*(c+d)}[kind]
    x=np.array([0.,1.,2.])
    terms=get_components(expression,additive=True)
    assert len(terms.names)==count
    np.testing.assert_allclose(sum(term(x) for term in terms),expected)
    np.testing.assert_allclose(expression(x),expected)
    if kind=='division':assert all('/(c+d)' in name for name in terms.names)
    if kind=='power':assert '(a+b)**' in terms.names[0]


@pytest.mark.parametrize('fold',[False,True])
def test_variable_numerator_and_intact_denominator_with_linear_response(fold):
    x=np.linspace(-3,3,101)
    a,b=Gaussian1D(2,-1,.5),Gaussian1D(1,1,.7)
    c,d=Gaussian1D(.3,0,1),Const1D(1)
    expression=(a-b)/(c+d)
    if fold:expression=expression|LinearOperatorModel(eye(len(x),format='csr'),x=x)
    terms=get_components(expression,additive=True)
    assert len(terms.names)==2
    np.testing.assert_allclose(sum(term(x) for term in terms),expression(x),rtol=1e-14,atol=1e-14)


def test_general_variable_power_stays_whole():
    x=np.linspace(-2,2,91)
    base=Gaussian1D(2,0,.7)+Const1D(1)
    exponent=Gaussian1D(.5,0,1)+Const1D(.5)
    expression=base**exponent
    terms=get_components(expression,additive=True)
    assert len(terms.names)==1
    np.testing.assert_array_equal(terms[0](x),expression(x))
