# -*- coding: utf-8 -*-
"""Methods for approximating functions in spline spaces."""
from __future__ import print_function

from . import bspline
from . import assemble
from . import tensor
from . import operators
from . import utils
from . import hierarchical

import sys
import scipy.sparse.linalg

def interpolate(kvs, f, geo=None, nodes=None):
    """Perform interpolation in a spline space.

    Returns the coefficients for the interpolant of the function `f` in the
    tensor product B-spline basis `kvs`.

    By default, `f` is assumed to be defined in the parameter domain. If a
    geometry is passed in `geo`, interpolation is instead done in physical
    coordinates.

    `nodes` should be a tensor grid (i.e., a sequence of one-dimensional
    arrays) in the parameter domain specifying the interpolation nodes. If not
    specified, the Gréville abscissae are used.
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    if nodes is None:
        nodes = [kv.greville() for kv in kvs]

    # evaluate f at interpolation nodes
    if geo is not None:
        rhs = utils.grid_eval_transformed(f, nodes, geo)
    else:
        rhs = utils.grid_eval(f, nodes)

    Cinvs = [operators.make_solver(bspline.collocation(kvs[i], nodes[i]))
                for i in range(len(kvs))]
    return tensor.apply_tprod(Cinvs, rhs)

def _project_L2_hspace(hs, f, f_physical=False, geo=None):
    from . import vform, geometry
    if geo is None:
        geo = geometry.identity(hs.knotvectors(0))
    M = assemble.assemble(vform.mass_vf(hs.dim), hs, geo=geo)
    rhs = assemble.assemble(vform.L2functional_vf(hs.dim, physical=f_physical),
            hs, geo=geo, f=f)
    return operators.make_solver(M, spd=True).dot(rhs)

def project_L2(kvs, f, f_physical=False, geo=None):
    """Perform :math:`L_2`-projection into a spline space.

    Returns the coefficients for the :math:`L_2`-projection of the function `f`
    into the tensor product B-spline basis `kvs`. Optionally, a geometry
    transform `geo` can be specified to compute the projection in a physical
    domain.

    By default, `f` is assumed to be defined in the parameter domain. If it is
    given in physical coordinates, pass `f_physical=True`. This requires `geo`
    to be specified.

    This function also supports projection into a hierarchical spline space by
    passing a :class:`.HSpace` object in place of `kvs`.
    """
    if isinstance(kvs, hierarchical.HSpace):
        return _project_L2_hspace(kvs, f, f_physical, geo)
    elif isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    Minvs = [operators.make_solver(assemble.mass(kv), spd=True) for kv in kvs]
    rhs = assemble.inner_products(kvs, f, f_physical=f_physical, geo=geo)
    if geo is None:
        assert not f_physical, 'Cannot use physical coordinates without geometry'
        # in the parameter domain, we simply apply the Kronecker product of the M^{-1}
        return tensor.apply_tprod(Minvs, rhs)
    else:
        # in the physical domain, use the Kronecker product as a preconditioner
        M = assemble.mass(kvs, geo=geo)
        b = rhs.ravel()
        assert b.shape[0] == M.shape[1], 'L2 projection with geometry only implemented for scalar functions'
        x, info = scipy.sparse.linalg.cg(M, b, tol=1e-12, atol=1e-12,
                maxiter=100, M=operators.KroneckerOperator(*Minvs))
        if info:
            print('WARNING: L2 projection - CG did not converge:', info, file=sys.stderr)
        return x.reshape(rhs.shape)
