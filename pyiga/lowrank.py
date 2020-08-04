import numpy as np
import numpy.random

from .lowrank_cy import *
from . import tensor

################################################################################
# Utility classes for entrywise matrix/tensor generation
################################################################################

class MatrixGenerator:
    def __init__(self, m, n, entryfunc=None, multientryfunc=None):
        """Set up a matrix generator with shape (m,n).

        Arguments:
            m: number of rows
            n: number of columns
            `entryfunc`: a function taking i,j and producing an entry.
            `multientryfunc`: a function taking a list of (i,j) pairs
                and producing an ndarray of the requested entries.

        At least one of the two functions must be provided.
        """
        self.shape = (m,n)
        assert entryfunc is not None or multientryfunc is not None, \
            "At least one of entryfunc and multientryfunc must be specified"
        if entryfunc is not None:
            self.entry = entryfunc
        if multientryfunc is not None:
            self.compute_entries = multientryfunc

    @staticmethod
    def from_array(X):
        """Create a MatrixGenerator which just returns the entries of a 2D array"""
        assert X.ndim == 2
        return MatrixGenerator(X.shape[0], X.shape[1],
                lambda i,j: X[i,j])

    def entry(self, i, j):
        """Generate the entry at row i and column j"""
        return self.compute_entries([(i, j)])[0]

    def compute_entries(self, indices):
        """Compute all entries given by the list of pairs `indices`."""
        indices = list(indices)
        n = len(indices)
        result = np.empty(n)
        for i in range(n):
            result[i] = self.entry(*indices[i])
        return result

    def row(self, i):
        """Generate the i-th row"""
        return self.compute_entries((i,j) for j in range(self.shape[1]))

    def column(self, j):
        """Generate the j-th column"""
        return self.compute_entries((i,j) for i in range(self.shape[0]))

    def asarray(self):
        """Generate the entire matrix as an np.ndarray"""
        return self.compute_entries(
                (i,j) for i in range(self.shape[0])
                      for j in range(self.shape[1])).reshape(self.shape, order='C')


class TensorGenerator:
    def __init__(self, shape, entryfunc=None, multientryfunc=None):
        self.shape = tuple(shape)
        assert entryfunc is not None or multientryfunc is not None, \
            "At least one of entryfunc and multientryfunc must be specified"
        if entryfunc is not None:
            self.entry = entryfunc
        if multientryfunc is not None:
            self.compute_entries = multientryfunc

    @staticmethod
    def from_array(X):
        return TensorGenerator(X.shape, lambda I: X[tuple(I)])

    def entry(self, I):
        """Generate the entry at index I"""
        return self.compute_entries([I])[0]

    def compute_entries(self, indices):
        """Compute all entries given by the list of tuples `indices`."""
        indices = list(indices)
        n = len(indices)
        result = np.empty(n)
        for i in range(n):
            result[i] = self.entry(indices[i])
        return result

    def fiber_at(self, I, axis):
        """Generate the fiber (vector) passing through index I along the given axis"""
        assert len(I) == len(self.shape)
        I = list(I)
        m = self.shape[axis]
        indices = []
        for i in range(m):
            I[axis] = i
            indices.append(tuple(I))
        return self.compute_entries(indices)

    def matrix_at(self, I, axes):
        """Return a MatrixGenerator for the matrix slice of this tensor which
        passes through index I along the given two axes."""
        assert len(axes) == 2
        assert len(I) == len(self.shape)
        I = list(I)
        def multientryfunc(indices):
            indices = list(indices)
            for k in range(len(indices)):
                I[axes[0]], I[axes[1]] = indices[k]
                indices[k] = tuple(I)
            return self.compute_entries(indices)

        return MatrixGenerator(self.shape[axes[0]],
                               self.shape[axes[1]],
                               multientryfunc=multientryfunc)

    def asarray(self):
        """Generate the full tensor as an np.ndarray"""
        return self.compute_entries(
            (i,j,k) for i in range(self.shape[0])
                    for j in range(self.shape[1])
                    for k in range(self.shape[2])).reshape(self.shape, order='C')


################################################################################
# Adaptive cross approximation (ACA)
################################################################################

def aca(A, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2, startval=None):
    """Adaptive Cross Approximation (ACA) algorithm with row pivoting"""
    if not isinstance(A, MatrixGenerator):
        A = MatrixGenerator.from_array(A)  # assume it's an array
    if startval is not None:
        X = np.array(startval, order='C')
        assert X.shape == A.shape
    else:
        X = np.zeros(A.shape, order='C')
    i = A.shape[0] // 2   # starting row
    k = 0
    skipcount, max_skipcount = 0, skipcount
    tolcount,  max_tolcount  = 0, tolcount

    while True:
        E_row = X[i,:] - A.row(i)
        j0 = abs(E_row).argmax()
        e = abs(E_row[j0])
        if e < 1e-15:
            if verbose >= 2:
                print('skipping', i)
            i = np.random.randint(A.shape[0])
            skipcount += 1
            if skipcount >= max_skipcount:
                if verbose >= 1:
                    print('maximum skip count reached; stopping (%d it.)' % k)
                break
            else:
                continue
        elif e < tol:
            tolcount += 1
            if tolcount >= max_tolcount:
                if verbose >= 1:
                    print('desired tolerance reached', tolcount, 'times; stopping (%d it.)' % k)
                break
        else:   # error is large
            skipcount = tolcount = 0   # reset the counters

        if verbose >= 2:
            print(i, '\t', j0, '\t', e)

        col = A.column(j0) - X[:,j0]
        rank_1_update(X, 1 / E_row[j0], col, E_row)

        col[i] = 0  # error is now 0 there
        i = abs(col).argmax()   # choose next row to pivot on
        k += 1
        if k >= maxiter:
            if verbose >= 1:
                print('Maximum iteration count reached; aborting (%d it.)' % k)
            break
    return X

def aca_lr(A, tol=1e-10, maxiter=100, verbose=2):
    """ACA which returns the crosses rather than the full matrix"""
    if not isinstance(A, MatrixGenerator):
        A = MatrixGenerator.from_array(A)  # assume it's an array
    crosses = []

    def X_row(i):
        return sum(c[i]*r for (c,r) in crosses)
    def X_col(j):
        return sum(c*r[j] for (c,r) in crosses)

    i = A.shape[0] // 2   # starting row
    k = 0
    skipcount, max_skipcount = 0, 3
    tolcount,  max_tolcount  = 0, 3

    while k < maxiter:
        err_i = X_row(i) - A.row(i)
        j0 = abs(err_i).argmax()
        e = abs(err_i[j0])
        if e < 1e-15:
            if verbose >= 2:
                print('skipping', i) #, '  error=', abs(err_i[j0]))
            #i = (i + 1) % A.shape[0]
            i = np.random.randint(A.shape[0])
            skipcount += 1
            if skipcount >= max_skipcount:
                if verbose >= 1:
                    print('maximum skip count reached; stopping (%d it.)' % k)
                break
            else:
                continue
        elif e < tol:
            tolcount += 1
            if tolcount >= max_tolcount:
                if verbose >= 1:
                    print('desired tolerance reached', tolcount, 'times; stopping (%d it.)' % k)
                break
        else:   # error is large
            skipcount = tolcount = 0   # reset the counters

        if verbose >= 2:
            print(i, '\t', j0, '\t', e)
        c = (A.column(j0) - X_col(j0)) / err_i[j0]
        crosses.append((c, err_i))
        i = abs(c).argmax()
        k += 1
    return crosses


def _tensor_slice(A, I):
    I = tuple(I)
    assert len(I) == A.ndim, 'invalid slice index'
    if isinstance(A, np.ndarray):
        sl = tuple((slice(None) if ii is None else ii) for ii in I)
        return A[sl]
    else:
        def slice_op(j, n):
            C = scipy.sparse.csr_matrix((1,n))
            C[0,j] = 1.0
            return C
        sl_ops = tuple(slice_op(I[j], A.shape[j]) if (I[j] is not None) else None
                for j in range(len(I)))
        S = tensor.apply_tprod(sl_ops, A)
        return np.squeeze(tensor.asarray(S))


def aca_3d(A, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2, lr=False):
    if not isinstance(A, TensorGenerator):
        A = TensorGenerator.from_array(A)  # assume it's an array

    X = np.zeros(A.shape)
    if lr: X_lr = tensor.TensorSum(tensor.CanonicalTensor.zeros(A.shape))

    I = list(m//2 for m in A.shape)  # starting index
    def randomize():
        for j in range(len(A.shape)):
            I[j] = np.random.randint(A.shape[j])
    k = 0
    skipcount, max_skipcount = 0, skipcount
    tolcount,  max_tolcount  = 0, tolcount

    while k < maxiter:
        E_col = A.fiber_at(I, axis=0) - X[:,I[1],I[2]]
        i0 = abs(E_col).argmax()
        e = abs(E_col[i0])
        if e < 1e-15:
            if verbose >= 2:
                print('skipping', I)
            randomize()
            skipcount += 1
            if skipcount >= max_skipcount:
                if verbose >= 1:
                    print('maximum skip count reached; stopping (%d outer it.)' % k)
                break
            else:
                continue
        elif e < tol:
            tolcount += 1
            if tolcount >= max_tolcount:
                if verbose >= 1:
                    print('desired tolerance reached', tolcount, 'times; stopping (%d outer it.)' % k)
                break
        else:   # error is large
            skipcount = tolcount = 0   # reset the counters

        I[0] = i0
        if verbose >= 2:
            print(I, '\t', e)

        A_mat = aca(A.matrix_at(I, axes=(1,2)), startval=X[i0,:,:],
                    tol=tol, maxiter=maxiter,
                    skipcount=max_skipcount, tolcount=max_tolcount,
                    verbose=min(verbose, 1))
        E_mat = A_mat - X[i0,:,:]

        # add the scaled tensor product E_col * E_mat
        aca3d_update(X, 1.0 / E_col[i0], E_col, E_mat)
        if lr:
            X_lr += tensor.TensorProd(E_col / E_col[i0], E_mat.copy())

        E_mat[I[1:]] = 0  # error is now (close to) zero there
        I[1:] = np.unravel_index(abs(E_mat).argmax(), E_mat.shape)
        k += 1
        if k >= maxiter:
            if verbose >= 1:
                print('Maximum iteration count reached; aborting (%d outer it.)' % k)
            break
    if lr:
        return tensor.TensorSum(*X_lr.Xs[1:])    # skip the zero CanonicalTensor
    else:
        return X


#from .lowrank_cy import *

