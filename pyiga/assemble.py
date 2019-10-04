"""Matrix assembling functions for B-spline IgA.

This module contains functions to assemble mass and stiffness matrices
for IgA with tensor product B-spline functions.


.. _gauss-asms:

Tensor product Gauss quadrature assemblers
------------------------------------------

Standard Gauss quadrature assemblers for mass and stiffness matrices.
They take one or two arguments:

- `kvs` (list of :class:`pyiga.bspline.KnotVector`):
  Describes the tensor product B-spline basis for which to assemble
  the matrix. One :class:`KnotVector` per coordinate direction.
  In the 1D case, a single :class:`pyiga.bspline.KnotVector` may be passed
  directly.
- `geo` (:class:`pyiga.geometry.BSplinePatch`; optional):
  Geometry transform, mapping from the parameter domain to the
  physical domain. If omitted, assume the identity map; a fast
  Kronecker product implementation is used in this case.

.. autofunction:: mass
.. autofunction:: stiffness


.. _fast-asms:

Fast low-rank assemblers
------------------------

Fast low-rank assemblers based on the paper
"A Black-Box Algorithm for Fast Matrix Assembly in Isogeometric Analysis".
They may achieve significant speedups over the classical Gauss assemblers,
in particular for fine discretizations and higher spline degrees.
They only work well if the geometry transform is rather smooth so that the
resulting matrix has relatively low (numerical) Kronecker rank.

They take the following additional arguments:

- `tol`: the stopping accuracy for the Adaptive Cross Approximation (ACA)
  algorithm
- `maxiter`: the maximum number of ACA iterations
- `skipcount`: terminate after finding this many successive near-zero pivots
- `tolcount`: terminate after finding this many successive pivots below the
  desired accuracy
- `verbose`: the amount of output to display. `0` is silent, `1` prints
  basic information, `2` prints detailed information

.. autofunction:: mass_fast
.. autofunction:: stiffness_fast

Right-hand sides
----------------

.. autofunction:: inner_products

Boundary and initial conditions
-------------------------------

.. autofunction:: compute_dirichlet_bc
.. autofunction:: compute_initial_condition_01
.. autofunction:: combine_bcs

.. autoclass:: RestrictedLinearSystem
    :members:

"""
import numpy as np
import scipy
import scipy.sparse
import itertools
import math

from . import bspline
from . import assemble_tools
from . import assemblers
from . import fast_assemble_cy
from . import tensor
from . import operators
from . import utils
from . import geometry

from .quadrature import make_iterated_quadrature, make_tensor_quadrature
from .mlmatrix import MLStructure

################################################################################
# 1D assembling routines
################################################################################

def _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights):
    assert nspans * nqp == vals1.shape[1]
    assert nspans * nqp == vals2.shape[1]
    assert qweights.shape == (vals1.shape[1],)
    n_act1,n_act2 = vals1.shape[0],vals2.shape[0]
    elMats = np.empty((nspans, n_act1, n_act2)) # contains one n_act1 x n_act2 element matrix per span
    for k in range(nspans):
        f1 = vals1[:,nqp*k:nqp*(k+1)]       # n_act1 x nqp
        f2 = vals2[:,nqp*k:nqp*(k+1)]       # n_act2 x nqp
        w = qweights[nqp*k:nqp*(k+1)]       # nqp-vector of weights
        elMats[k, :, :] = np.dot( f1, (f2 * w).transpose() )
    return elMats       # n_act1*nspans x n_act2

def _create_coo_1d_from_kv(kv):
    n_act1 = n_act2 = kv.p + 1
    nspans = kv.numspans
    grid = np.mgrid[:n_act1, :n_act2] # 2 x n_act1 x n_act2 array which indexes element matrix
    I_ref = grid[0].ravel()          # slowly varying index, first basis
    J_ref = grid[1].ravel()          # fast varying index, second basis

    first_act = kv.first_active(kv.mesh_span_indices())
    first_act = np.repeat(first_act, n_act1*n_act2)
    I = first_act + np.tile(I_ref, nspans)
    J = first_act + np.tile(J_ref, nspans)
    return (I, J)

def _create_coo_1d_custom(nspans, n_act1, n_act2, first_act1, first_act2):
    """Create COO indices for two sequentially numbered bases over `nspans` knot spans"""
    grid = np.mgrid[:n_act1, :n_act2] # 2 x n_act1 x n_act2 array which indexes element matrix
    I_ref = grid[0].ravel()          # slowly varying index, first basis
    J_ref = grid[1].ravel()          # fast varying index, second basis

    I = np.repeat(first_act1, n_act1*n_act2) + np.tile(I_ref, nspans)
    J = np.repeat(first_act2, n_act1*n_act2) + np.tile(J_ref, nspans)
    return (I, J)

def _assemble_matrix_custom(nspans, nqp, vals1, vals2, I, J, qweights):
    n_act1 = vals1.shape[0]
    n_act2 = vals2.shape[0]
    elMats = _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights)
    return scipy.sparse.coo_matrix((elMats.ravel(), (I, J))).tocsr()

def bsp_mass_1d(knotvec, weightfunc=None):
    """Assemble the mass matrix for the B-spline basis over the given knot vector.

    Optionally, a weight function can be passed; by default, it is assumed to be 1.
    """
    return bsp_mixed_deriv_biform_1d(knotvec, 0, 0, weightfunc=weightfunc)

def bsp_stiffness_1d(knotvec, weightfunc=None):
    """Assemble the Laplacian stiffness matrix for the B-spline basis over the given knot vector.

    Optionally, a weight function can be passed; by default, it is assumed to be 1.
    """
    return bsp_mixed_deriv_biform_1d(knotvec, 1, 1, weightfunc=weightfunc)

def bsp_mixed_deriv_biform_1d(knotvec, du, dv, nqp=None, weightfunc=None):
    "Assemble the matrix for a(u,v)=(weight*u^(du),v^(dv)) for the B-spline basis over the given knot vector"""
    nspans = knotvec.numspans
    # default: use that q-term Gauss quadrature is exact up to poly degree 2q-1
    if nqp is None: nqp = int(math.ceil((2 * knotvec.p - du - dv + 1) / 2.0))
    q = make_iterated_quadrature(knotvec.mesh, nqp)
    derivs = bspline.active_deriv(knotvec, q[0], max(du, dv))
    qweights = q[1]
    if weightfunc is not None:
        qweights *= utils.grid_eval(weightfunc, (q[0],))
    I,J = _create_coo_1d_from_kv(knotvec)
    return _assemble_matrix_custom(nspans, nqp, derivs[dv, :, :], derivs[du, :, :], I, J, qweights)

def bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, du, dv, quadgrid=None, nqp=None):
    """Assemble the matrix for a(u,v)=(u^(du),v^(dv)) relating the two B-spline
    bases. By default, uses the first knot vector for quadrature.
    """
    if quadgrid is None:
        quadgrid = knotvec1.mesh

    # create iterated Gauss quadrature rule for each interval
    if nqp is None:
        nqp = int(math.ceil((knotvec1.p + knotvec2.p - du - dv + 1) / 2.0))
    nspans = len(quadgrid) - 1
    q = make_iterated_quadrature(quadgrid, nqp)
    assert len(q[0]) == nspans * nqp

    # evaluate derivatives of basis functions at quadrature nodes
    derivs1 = bspline.active_deriv(knotvec1, q[0], du)[du, :, :]
    derivs2 = bspline.active_deriv(knotvec2, q[0], dv)[dv, :, :]

    first_points = q[0][::nqp]
    assert len(first_points) == nspans
    # map first_active_at over first quadrature points to get first active basis function index
    first_act1 = np.vectorize(knotvec1.first_active_at, otypes=(np.int,))(first_points)
    first_act2 = np.vectorize(knotvec2.first_active_at, otypes=(np.int,))(first_points)
    I,J = _create_coo_1d_custom(nspans, derivs1.shape[0], derivs2.shape[0], first_act1, first_act2)

    return _assemble_matrix_custom(nspans, nqp, derivs1, derivs2, I, J, q[1])

def bsp_mass_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a mass matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    return bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, 0, 0, quadgrid=quadgrid)

def bsp_stiffness_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a stiffness matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    return bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, 1, 1, quadgrid=quadgrid)

################################################################################
# 2D/3D assembling routines (rely on Cython module)
################################################################################

def bsp_mass_2d(knotvecs, geo=None, format='csr'):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        return scipy.sparse.kron(M1, M2, format=format)
    else:
        return assemble(
                assemblers.MassAssembler2D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_stiffness_2d(knotvecs, geo=None, format='csr'):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        K1 = bsp_stiffness_1d(kv1)
        K2 = bsp_stiffness_1d(kv2)
        return scipy.sparse.kron(K1, M2, format=format) + scipy.sparse.kron(M1, K2, format=format)
    else:
        return assemble(
                assemblers.StiffnessAssembler2D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_mass_3d(knotvecs, geo=None, format='csr'):
    if geo is None:
        M = [bsp_mass_1d(kv) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format=format)
        return k(M[0], k(M[1], M[2]))
    else:
        return assemble(
                assemblers.MassAssembler3D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_stiffness_3d(knotvecs, geo=None, format='csr'):
    if geo is None:
        MK = [(bsp_mass_1d(kv), bsp_stiffness_1d(kv)) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format=format)
        M12 = k(MK[1][0], MK[2][0])
        K12 = k(MK[1][1], MK[2][0]) + k(MK[1][0], MK[2][1])
        return k(MK[0][1], M12) + k(MK[0][0], K12)
    else:
        return assemble(
                assemblers.StiffnessAssembler3D(knotvecs, geo),
                symmetric=True, format=format)

################################################################################
# Assembling right-hand sides
################################################################################

def inner_products(kvs, f, f_physical=False, geo=None):
    """Compute the :math:`L_2` inner products between each basis
    function in a tensor product B-spline basis and the function `f`
    (i.e., the load vector).

    Args:
        kvs (seq): a sequence of :class:`pyiga.bspline.KnotVector`,
            representing a tensor product basis
        f: a function or :class:`pyiga.geometry.BSplineFunc` object
        f_physical (bool): whether `f` is given in physical coordinates.
            If `True`, `geo` must be passed as well.
        geo: a :class:`pyiga.geometry.BSplinePatch` which describes
            the integration domain; if not given, the integrals are
            computed in the parameter domain

    Returns:
        ndarray: the inner products as an array of size
        `kvs[0].ndofs x kvs[1].ndofs x ... x kvs[-1].ndofs`.
        Each entry corresponds to the inner product of the
        corresponding basis function with `f`.
        If `f` is not scalar, then each of its components is treated separately
        and the corresponding dimensions are appended to the end of the return
        value.
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    # compute quadrature rules
    nqp = max(kv.p for kv in kvs) + 1
    gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], nqp)

    # evaluate function f on grid or transformed grid
    if f_physical:
        assert geo is not None, 'inner_products in physical domain requires geometry'
        fvals = utils.grid_eval_transformed(f, gaussgrid, geo)
    else:
        fvals = utils.grid_eval(f, gaussgrid)

    # multiply function values with quadrature weights
    fvals = tensor.apply_tprod(
              [operators.DiagonalOperator(gw) for gw in gaussweights], fvals)
    # if geometry was specified, multiply by abs(det(jac))
    if geo is not None:
        geo_jac = geo.grid_jacobian(gaussgrid)
        geo_det = np.abs(assemble_tools.determinants(geo_jac))
        # if f is not scalar, we simply add trivial dimensions on to the end
        extra_dims = fvals.ndim - geo_det.ndim
        if extra_dims > 0:
            geo_det.shape = geo_det.shape + (extra_dims * (1,))
        fvals *= geo_det
    # apply transposed spline collocation matrices (sum over Gauss nodes)
    Ct = [bspline.collocation(kvs[i], gaussgrid[i]).T
            for i in range(len(kvs))]
    return tensor.apply_tprod(Ct, fvals)

################################################################################
# Incorporating essential boundary conditions
################################################################################

def slice_indices(ax, idx, shape, ravel=False):
    """Return dof indices for a slice of a tensor product basis with size
    `shape`. The slice is taken across index `idx` on axis `ax`.

    The indices are returned either as a `N x dim` array of multiindices or,
    with `ravel=True`, as an array of sequential (raveled) indices.
    """
    shape = tuple(shape)
    if idx < 0:
        idx += shape[ax]     # wrap around
    axdofs = [range(n) for n in shape]
    axdofs[ax] = [idx]
    multi_indices = np.array(list(itertools.product(*axdofs)))
    if ravel:
        multi_indices = np.ravel_multi_index(multi_indices.T, shape)
    return multi_indices


def compute_dirichlet_bc(kvs, geo, bdspec, dir_func):
    """Compute indices and values for a Dirichlet boundary condition using
    interpolation.

    Args:
        kvs: a tensor product B-spline basis
        geo (:class:`pyiga.geometry.BSplinePatch`): the geometry transform
        bdspec: a pair `(axis, side)`. `axis` denotes the axis along
            which the boundary condition lies, and `side` is either
            0 for the "lower" boundary or 1 for the "upper" boundary.
        dir_func: a function which will be interpolated to obtain the
            Dirichlet boundary values. Assumed to be given in physical
            coordinates. If it is vector-valued, one Dirichlet dof is
            computed per component, and they are numbered according to
            the "blocked" matrix layout.

    Returns:
        A pair of arrays `(indices, values)` which denote the indices of the
        dofs within the tensor product basis which lie along the Dirichlet
        boundary and their computed values, respectively.
    """
    bdax, bdside = bdspec

    # get basis for the boundary face
    bdbasis = list(kvs)
    assert len(bdbasis) == geo.sdim, 'Invalid dimension of geometry'
    del bdbasis[bdax]

    # get boundary geometry and interpolate dir_func
    bdgeo = geo.boundary(bdax, bdside)
    from .approx import interpolate
    dircoeffs = interpolate(bdbasis, dir_func, geo=bdgeo)

    # compute sequential indices for eliminated dofs
    N = tuple(kv.numdofs for kv in kvs)
    bdindices = slice_indices(bdax, 0 if bdside==0 else -1, N, ravel=True)

    extra_dims = dircoeffs.ndim - len(bdbasis)
    if extra_dims == 0:
        return bdindices, dircoeffs.ravel()
    elif extra_dims == 1:
        # vector function; assume blocked vector discretization
        numcomp = dircoeffs.shape[-1]
        NN = np.prod(N)
        return combine_bcs(
            (bdindices + j*NN, dircoeffs[..., j].ravel())
                for j in range(numcomp))
    else:
        raise ValueError('invalid dimension of Dirichlet coefficients: %s' % dircoeffs.shape)


def compute_initial_condition_01(kvs, geo, bdspec, g0, g1, physical=True):
    r"""Compute indices and values for an initial condition including function
    value and derivative for a space-time discretization using interpolation.
    This only works for a space-time cylinder with constant (in time) geometry.
    To be precise, the space-time geometry transform `geo` should have the form

    .. math:: G(\vec x, t) = (\widetilde G(\vec x), t).

    Args:
        kvs: a tensor product B-spline basis
        geo (:class:`pyiga.geometry.BSplinePatch`): the geometry transform of
            the space-time cylinder
        bdspec: a pair `(axis, side)`. `axis` denotes the time axis of `geo`,
            and `side` is either 0 for the "lower" boundary or 1 for the
            "upper" boundary.
        g0: a function which will be interpolated to obtain the initial
            function values
        g1: a function which will be interpolated to obtain the initial
            derivatives.
        physical (bool): whether the functions `g0` and `g1` are given in
            physical (True) or parametric (False) coordinates. Physical
            coordinates are assumed by default.

    Returns:
        A pair of arrays `(indices, values)` which denote the indices of the
        dofs within the tensor product basis which lie along the initial face
        of the space-time cylinder and their computed values, respectively.
    """
    bdax, bdside = bdspec

    bdbasis = list(kvs)
    del bdbasis[bdax]

    bdgeo = geo.boundary(bdax, bdside) if physical else None
    from .approx import interpolate
    coeffs01 = np.stack((  # coefficients for 0th and 1st derivatives, respectively
        interpolate(bdbasis, g0, geo=bdgeo).ravel(),
        interpolate(bdbasis, g1, geo=bdgeo).ravel()
    ))

    # compute 2x2 matrix which maps the two boundary coefficients to 0-th and 1-st derivative
    # at the boundary (only two basis functions have contributions there!)
    if bdside == 0:
        bdcolloc = bspline.active_deriv(kvs[bdax], 0.0, 1)[:2, :2] # first two basis functions
    else:
        bdcolloc = bspline.active_deriv(kvs[bdax], 1.0, 1)[:2, -2:] # last two basis functions

    # note: this only works for a space-time cylinder with constant geometry!
    coll_coeffs = np.linalg.solve(bdcolloc, coeffs01)

    # compute indices for the two boundary slices
    N = tuple(kv.numdofs for kv in kvs)
    firstidx = (0 if bdside==0 else -2)
    bdindices = np.concatenate((
        slice_indices(bdax, firstidx,   N, ravel=True),
        slice_indices(bdax, firstidx+1, N, ravel=True)
    ))

    return bdindices, coll_coeffs.ravel()


def combine_bcs(bcs):
    """Given a sequence of `(indices, values)` pairs such as returned by
    :func:`compute_dirichlet_bc`, combine them into a single pair
    `(indices, values)`.

    Dofs which occur in more than one `indices` array take their
    value from an arbitrary corresponding `values` array.
    """
    bcs = list(bcs)
    indices = np.concatenate([ind for ind,_ in bcs])
    values  = np.concatenate([val for _,val in bcs])
    assert indices.shape == values.shape, 'Inconsistent BC sizes'

    uidx, lookup = np.unique(indices, return_index=True)
    return uidx, values[lookup]


class RestrictedLinearSystem:
    """Represents a linear system with some of its dofs eliminated.

    Args:
        A: the full matrix
        b: the right-hand side (may be 0)
        bcs: a pair of arrays `(indices, values)` which contain the
            indices and values, respectively, of dofs to be eliminated
            from the system

    Once constructed, the restricted linear system can be accessed through
    the following attributes:

    Attributes:
        A: the restricted matrix
        b: the restricted and updated right-hand side
    """
    def __init__(self, A, b, bcs):
        indices, values = bcs
        if np.isscalar(b):
            b = np.broadcast_to(b, A.shape[0])
        if np.isscalar(values):
            values = np.broadcast_to(values, indices.shape[0])

        I = scipy.sparse.eye(A.shape[0], format='csr')
        # compute mask which contains non-eliminated dofs
        mask = np.ones(A.shape[0], dtype=bool)
        mask[list(indices)] = False

        # TODO/BUG: this may require the indices to be in increasing order?
        self.R_free = I[mask]
        self.R_elim = I[np.logical_not(mask)]
        self.values = values

        self.A = self.restrict_matrix(A)
        self.b = self.restrict(b - A.dot(self.R_elim.T.dot(values)))

    def restrict(self, u):
        """Given a vector `u` containing all dofs, return its restriction to the free dofs."""
        return self.R_free.dot(u)

    def restrict_matrix(self, B):
        """Given a matrix `B` which operates on all dofs, return its restriction to the free dofs."""
        return self.R_free.dot(B).dot(self.R_free.T)

    def extend(self, u):
        """Given a vector `u` containing only the free dofs, pad it with zeros to all dofs."""
        return self.R_free.T.dot(u)

    def complete(self, u):
        """Given a solution `u` of the restricted linear system, complete it
        with the values of the eliminated dofs to a solution of the original
        system.
        """
        return self.extend(u) + self.R_elim.T.dot(self.values)

################################################################################
# Integration
################################################################################

def integrate(kvs, f, f_physical=False, geo=None):
    """Compute the integral of the function `f` over the geometry
    `geo` or a simple tensor product domain.

    Args:
        kvs (seq): a sequence of :class:`pyiga.bspline.KnotVector`;
            determines the parameter domain and the quadrature rule
        f: a function or :class:`pyiga.geometry.BSplineFunc` object
        f_physical (bool): whether `f` is given in physical coordinates.
            If `True`, `geo` must be passed as well.
        geo: a :class:`pyiga.geometry.BSplinePatch` which describes
            the integration domain; if not given, the integral is
            computed in the parameter domain

    Returns:
        float: the integral of `f` over the specified domain
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    # compute quadrature rules
    nqp = max(kv.p for kv in kvs) + 1
    gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], nqp)

    # evaluate function f on grid or transformed grid
    if f_physical:
        assert geo is not None, 'integrate in physical domain requires geometry'
        fvals = utils.grid_eval_transformed(f, gaussgrid, geo)
    else:
        fvals = utils.grid_eval(f, gaussgrid)

    # multiply function values with quadrature weights
    fvals = tensor.apply_tprod(
              [operators.DiagonalOperator(gw) for gw in gaussweights], fvals)
    # if geometry was specified, multiply by abs(det(jac))
    if geo is not None:
        geo_jac = geo.grid_jacobian(gaussgrid)
        geo_det = np.abs(assemble_tools.determinants(geo_jac))
        fvals *= geo_det
    # sum over all coordinate axes (leave vector components intact, if any)
    return fvals.sum(axis=tuple(range(len(kvs))))

################################################################################
# Driver routines for assemblers
################################################################################

def assemble(asm, symmetric=False, format='csr'):
    kvs0, kvs1 = asm.kvs
    X = MLStructure.from_kvs(kvs0, kvs1).make_mlmatrix()

    if isinstance(asm, assemble_tools.BaseAssembler2D):
        X.data = assemble_tools.generic_assemble_core_2d(asm, X.structure.bidx, symmetric=symmetric)
    elif isinstance(asm, assemble_tools.BaseAssembler3D):
        X.data = assemble_tools.generic_assemble_core_3d(asm, X.structure.bidx, symmetric=symmetric)
    else:
        assert False, 'Unknown assembler type'
    if format == 'mlb':
        return X
    else:
        return X.asmatrix(format)

def assemble_vector(asm, symmetric=False, format='csr', layout='blocked'):
    assert layout in ('packed', 'blocked')

    kvs0, kvs1 = asm.kvs
    dim = len(kvs0)
    nc = asm.num_components()[::-1]  # reverse axes (u = kv0 = columns)
    struc = MLStructure.from_kvs(kvs0, kvs1).join(MLStructure.dense(nc))
    X = struc.make_mlmatrix()

    if dim == 2:
        X.data = assemble_tools.generic_assemble_core_vec_2d(asm, X.structure.bidx[:dim], symmetric)
    elif dim == 3:
        X.data = assemble_tools.generic_assemble_core_vec_3d(asm, X.structure.bidx[:dim], symmetric)
    else:
        assert False, 'dimension %d not implemented' % dim

    if layout == 'blocked':
        axes = (dim,) + tuple(range(dim))    # bring last axis to the front
        X = X.reorder(axes)
    if format == 'mlb':
        return X
    else:
        return X.asmatrix(format)

################################################################################
# Convenience functions
################################################################################

def _detect_dim(kvs):
    if isinstance(kvs, bspline.KnotVector):
        return 1
    else:
        return len(kvs)

def mass(kvs, geo=None, format='csr'):
    """Assemble a mass matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_mass_1d(kvs)
    elif dim == 2:
        return bsp_mass_2d(kvs, geo, format)
    elif dim == 3:
        return bsp_mass_3d(kvs, geo, format)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def stiffness(kvs, geo=None, format='csr'):
    """Assemble a stiffness matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_stiffness_1d(kvs)
    elif dim == 2:
        return bsp_stiffness_2d(kvs, geo, format)
    elif dim == 3:
        return bsp_stiffness_3d(kvs, geo, format)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def divdiv(kvs, geo=None, layout='blocked', format='csr'):
    dim = _detect_dim(kvs)
    if geo is None:
        geo = geometry.unit_cube(dim=dim)   # TODO: fast assembling for div-div?
    if dim == 2:
        asm = assemblers.DivDivAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.DivDivAssembler3D(kvs, geo)
    else:
        assert False, 'dimension %d not implemented' % dim
    return assemble_vector(asm, symmetric=True, layout=layout, format=format)

def mass_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a mass matrix for the given tensor product B-spline basis with
    an optional geometry transform, using the fast low-rank assembling
    algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return mass(kvs)
    dim = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        asm = assemblers.MassAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.MassAssembler3D(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."
    return fast_assemble_cy.fast_assemble(asm, kvs, tol, maxiter, skipcount, tolcount, verbose)

def stiffness_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a stiffness matrix for the given tensor product B-spline basis
    with an optional geometry transform, using the fast low-rank assembling
    algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return stiffness(kvs)
    dim = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        asm = assemblers.StiffnessAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.StiffnessAssembler3D(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."
    return fast_assemble_cy.fast_assemble(asm, kvs, tol, maxiter, skipcount, tolcount, verbose)

