# -*- coding: utf-8 -*-

from pyiga.bspline import *

def test_eval():
    # create random spline
    kv = make_knots(4, 0.0, 1.0, 25)
    n = kv.numdofs
    coeffs = np.random.rand(n)
    # evaluate it, B-spline by B-spline
    x = np.linspace(0.0, 1.0, 100)
    values = sum((coeffs[j] * single_ev(kv, j, x) for j in range(n)))
    # evaluate all at once and compare
    values2 = ev(kv, coeffs, x)
    assert np.linalg.norm(values - values2) < 1e-10
    # evaluate through collocation matrix and compare
    values3 = collocation(kv, x).dot(coeffs)
    assert np.linalg.norm(values - values3) < 1e-10

def test_interpolation():
    kv = make_knots(3, 0.0, 1.0, 10)
    # create random spline
    coeffs = np.random.rand(kv.numdofs)
    def f(x): return ev(kv, coeffs, x)
    # interpolate it at Gréville points and check that result is the same
    result = interpolate(kv, f)
    assert np.allclose(coeffs, result)

def test_L2_projection():
    kv = make_knots(3, 0.0, 1.0, 10)
    def f(x): return np.sin(2*np.pi * x**2)
    x = np.linspace(0.0, 1.0, 100)
    coeffs = project_L2(kv, f)
    assert np.linalg.norm(f(x) - ev(kv, coeffs, x)) / np.sqrt(len(x)) < 1e-3

def test_deriv():
    # create linear spline
    kv = make_knots(4, 0.0, 1.0, 25)
    coeffs = interpolate(kv, lambda x: 1.0 + 2.5*x)
    # check that derivative is 2.5
    x = np.linspace(0.0, 1.0, 100)
    drv = deriv(kv, coeffs, 1, x)
    assert np.linalg.norm(drv - 2.5) < 1e-10

    # create random spline
    coeffs = np.random.rand(kv.numdofs)
    # compare derivatives by two methods
    derivs1 = deriv(kv, coeffs, 1, x)
    derivs2 = deriv(kv, coeffs, 2, x)
    allders = collocation_derivs(kv, x, derivs=2)
    assert np.linalg.norm(derivs1 - allders[1].dot(coeffs), np.inf) < 1e-10
    assert np.linalg.norm(derivs2 - allders[2].dot(coeffs), np.inf) < 1e-10

def test_prolongation():
    # create random spline
    kv = make_knots(3, 0.0, 1.0, 10)
    coeffs = np.random.rand(kv.numdofs)
    # compute a refined knot vector and prolongation matrix
    kv2 = kv.refine()
    P = prolongation(kv, kv2)
    coeffs2 = P.dot(coeffs)
    # check that they evaluate to the same function
    x = np.linspace(0.0, 1.0, 100)
    val1 = ev(kv, coeffs, x)
    val2 = ev(kv2, coeffs2, x)
    assert np.linalg.norm(val1 - val2) < 1e-10

