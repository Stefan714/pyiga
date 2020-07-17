# cython: profile=False

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

import scipy.sparse

################################################################################
# Reindexing
################################################################################

@cython.cdivision(True)
def reindex_from_reordered(size_t i, size_t j, size_t m1, size_t n1, size_t m2, size_t n2):
    """Convert (i,j) from an index into reorder(X, m1, n1) into the
    corresponding index into X (reordered to original).

    Arguments:
        i = row = block index           (0...m1*n1)
        j = column = index within block (0...m2*n2)

    Returns:
        a pair of indices with ranges `(0...m1*m2, 0...n1*n2)`
    """
    cdef size_t bi0, bi1, ii0, ii1
    bi0, bi1 = i // n1, i % n1      # range: m1, n1
    ii0, ii1 = j // n2, j % n2      # range: m2, n2
    return (bi0*m2 + ii0, bi1*n2 + ii1)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void from_seq2(np.int_t i, np.int_t[:] dims, size_t[2] out):
    out[0] = i // dims[1]
    out[1] = i % dims[1]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object from_seq(np.int_t i, np.int_t[:] dims):
    """Convert sequential (lexicographic) index into multiindex.

    Same as np.unravel_index(i, dims) except for returning a list.
    """
    cdef np.int_t L, k
    L = len(dims)
    I = L * [0]
    for k in reversed(range(L)):
        mk = dims[k]
        I[k] = i % mk
        i //= mk
    return I

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int_t to_seq(np.int_t* I, np.int_t* dims, int n) nogil:
    """Convert multiindex into sequential (lexicographic) index.

    Same as np.ravel_multiindex(I, dims).
    """
    cdef np.int_t i, k
    i = 0
    for k in range(n):
        i *= dims[k]
        i += I[k]
    return i

# this doesn't work currently
#def reindex_to_multilevel(i, j, bs):
#    bs = np.array(bs, copy=False)   # bs has shape L x 2
#    I, J = from_seq(i, bs[:,0]), from_seq(j, bs[:,1])  # multilevel indices
#    return tuple(to_seq((I[k],J[k]), bs[k,:]) for k in range(bs.shape[0]))   # <<- to_seq() expects buffers

def reindex_from_multilevel(M, np.int_t[:,:] bs):
    """Convert a multiindex M with length L into sequential indices (i,j)
    of a multilevel matrix with L levels and block sizes bs.

    This is the multilevel version of reindex_from_reordered, which does the
    same for the two-level case.

    Arguments:
        M: the multiindex to convert
        bs: the block sizes; ndarray of shape Lx2. Each row gives the sizes
            (rows and columns) of the blocks on the corresponding matrix level.
    """
    cdef size_t L = len(M), k
    cdef size_t[2] ij
    cdef size_t ii=0, jj=0

    for k in range(L):
        from_seq2(M[k], bs[k,:], ij)
        ii *= bs[k,0]
        ii += ij[0]
        jj *= bs[k,1]
        jj += ij[1]
    return (ii, jj)


# computes the Cartesian product of indices and ravels them according to the size `dims`
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pyx_raveled_cartesian_product(arrays, np.int_t[::1] dims):
    cdef int L = len(arrays)
    cdef np.int_t[8] I      # iteration index
    cdef np.int_t[8] K      # corresponding indices K[k] = arrays[k][I[k]]
    cdef np.int_t[8] shp    # size of Cartesian product
    cdef int i, k, N

    cdef (np.int_t*)[8] arr_ptrs
    cdef np.int_t[:] arr

    N = 1
    for k in range(L):
        # initialize shape
        shp[k] = arrays[k].shape[0]
        if shp[k] == 0:
            return np.zeros(0, dtype=np.int)
        N *= shp[k]
        # initialize pointer
        arr = arrays[k]
        arr_ptrs[k] = &arr[0]
        # initialize I and K
        I[k] = 0
        K[k] = arr_ptrs[k][I[k]]

    out_buf = np.empty(N, dtype=np.int)
    cdef np.int_t[::1] out = out_buf

    with nogil:
        for i in range(N):
            # compute raveled index at current multi-index K
            out[i] = to_seq(&K[0], &dims[0], L)

            # increment multi-index I and update K
            for k in reversed(range(L)):
                I[k] += 1
                if I[k] < shp[k]:
                    K[k] = arr_ptrs[k][I[k]]
                    break
                else:
                    I[k] = 0
                    K[k] = arr_ptrs[k][I[k]]
                    # go on to increment previous one
    return out_buf

# helper function for nonzeros_for_rows
@cython.boundscheck(False)
@cython.wraparound(False)
def pyx_rowwise_cartesian_product(lvia, np.int_t[:, ::1] ix, np.int_t[::1] block_sizes):
    cdef int N = ix.shape[0]
    cdef int L = ix.shape[1]
    cdef int i

    Js = N * [None]
    for i in range(N):      # loop over all row_indices
        # obtain the levelwise interactions for each index i_k = ix[i,k]
        ia_k = tuple(lvia[k][ix[i,k]] for k in range(L))
        # compute global interactions by taking the Cartesian product
        Js[i] = pyx_raveled_cartesian_product(ia_k, block_sizes)
    return Js

################################################################################
# Inflation and matvec
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_2d(bidx, block_sizes, bint lower_tri=False):
    cdef unsigned[:,::1] bidx1, bidx2
    cdef int m2,n2
    cdef size_t i, j, Ni, Nj, N, idx, I, J
    cdef unsigned xi0, xi1, yi0, yi1

    bidx1, bidx2 = bidx
    m2, n2 = block_sizes[1]
    Ni, Nj = len(bidx1), len(bidx2)
    N = Ni * Nj
    results = np.empty((2, N), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2

                I = xi0 * m2 + yi0      # range: m1*m2*m3
                J = xi1 * n2 + yi1      # range: n1*n2*n3
                if not lower_tri or J <= I:
                    IJ[0,idx] = I
                    IJ[1,idx] = J
                    idx += 1
    if idx < N: # TODO: what's the closed formula for N in the triangular case?
        return results[:, :idx]
    else:
        return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_2d(object X, np.int_t[:] sparsidx1, np.int_t[:] sparsidx2,
        int m1, int n1, int m2, int n2):
    """Convert the dense ndarray X from reordered, compressed two-level
    banded form back to a standard sparse matrix format.
    """
    cdef long[::1] entries_i, entries_j
    cdef size_t i, j, si, sj, M, N, k=0
    M, N = X.shape[0], X.shape[1]

    cdef size_t bi0, bi1, ii0, ii1

    assert len(sparsidx1) == M
    assert len(sparsidx2) == N

    entries_i = np.empty(M*N, dtype=int)
    entries_j = np.empty(M*N, dtype=int)

    for i in range(M):
        si = sparsidx1[i]                   # range: m1*n1
        bi0, bi1 = si // n1, si % n1        # range: m1, n1
        for j in range(N):
            sj = sparsidx2[j]               # range: m2*n2
            ii0, ii1 = sj // n2, sj % n2    # range: m2, n2

            entries_i[k] = bi0*m2 + ii0     # range: m1*m2
            entries_j[k] = bi1*n2 + ii1     # range: n1*n2
            k += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2, n1*n2))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_2d_bidx(object X, bidx, block_sizes):
    """Convert the dense ndarray X from reordered, compressed two-level
    banded form back to a standard sparse matrix format.
    """
    cdef unsigned[:,::1] bidx0, bidx1
    bidx0, bidx1 = bidx

    cdef int m1,n1, m2,n2
    m1,n1 = block_sizes[0]
    m2,n2 = block_sizes[1]

    cdef long[::1] entries_i, entries_j
    cdef size_t i, j, MU0, MU1, k=0
    MU0, MU1 = X.shape[0], X.shape[1]

    cdef size_t I[2]
    cdef size_t J[2]

    assert len(bidx0) == MU0
    assert len(bidx1) == MU1

    entries_i = np.empty(MU0*MU1, dtype=int)
    entries_j = np.empty(MU0*MU1, dtype=int)

    for i in range(MU0):
        I[0], J[0] = bidx0[i,0], bidx0[i,1]       # range: m1, n1
        for j in range(MU1):
            I[1], J[1] = bidx1[j,0], bidx1[j,1]   # range: m2, n2

            entries_i[k] = I[0]*m2 + I[1]         # range: m1*m2
            entries_j[k] = J[0]*n2 + J[1]         # range: n1*n2
            k += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2, n1*n2))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_2d(double[:,::1] X, bidx, block_sizes, double[::1] x, double[::1] y):
    cdef unsigned[:,::1] bidx1, bidx2
    bidx1,bidx2 = bidx

    cdef int m2, n2
    m2,n2 = block_sizes[1]

    cdef size_t i, j, M, N, I, J
    M, N = X.shape[0], X.shape[1]

    cdef unsigned bi0, bi1, ii0, ii1

    assert len(bidx1) == M
    assert len(bidx2) == N

    #for i in prange(M, schedule='static', nogil=True):
    # the += update is not thread safe! need atomics
    for i in range(M):
        bi0, bi1 = bidx1[i,0], bidx1[i,1]        # range: m1, n1
        for j in range(N):
            ii0, ii1 = bidx2[j,0], bidx2[j,1]    # range: m2, n2

            I = bi0*m2 + ii0     # range: m1*m2
            J = bi1*n2 + ii1     # range: n1*n2

            y[I] += X[i,j] * x[J]


def get_transpose_idx_for_bidx(bidx):
    cdef dict jidict
    jidict = {}
    for (k,(i,j)) in enumerate(bidx):
        jidict[(j,i)] = k

    transpose_bidx = np.empty(len(bidx), dtype=np.uintp)

    for k in range(len(bidx)):
        transpose_bidx[k] = jidict[ tuple(bidx[k]) ]
    return transpose_bidx

def symmetrize_X_2d(X, bidx):
    cdef unsigned[:, ::1] bidx0, bidx1
    cdef unsigned MU0, MU1
    cdef size_t[::1] transp0, transp1
    cdef double[:, ::1] values = X
    cdef size_t[2] i, j

    bidx0, bidx1 = bidx
    MU0, MU1 = bidx0.shape[0], bidx1.shape[0]

    transp0 = get_transpose_idx_for_bidx(bidx0)
    transp1 = get_transpose_idx_for_bidx(bidx1)

    entries = np.zeros((MU0, MU1))

    for mu0 in range(MU0):
        i[0] = bidx0[mu0, 0]
        j[0] = bidx0[mu0, 1]

        diag0 = <int>j[0] - <int>i[0]
        if diag0 > 0:       # block is above diagonal?
            continue

        for mu1 in range(MU1):
            i[1] = bidx1[mu1, 0]
            j[1] = bidx1[mu1, 1]

            diag1 = <int>j[1] - <int>i[1]
            if (diag0 == 0) and (diag1 > 0):
                continue

            if diag0 != 0 or diag1 != 0:
                values[ transp0[mu0], transp1[mu1] ] = values[ mu0, mu1 ]



## 3D

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_3d(bidx, block_sizes, bint lower_tri=False):
    cdef unsigned[:,::1] bidx1, bidx2, bidx3
    cdef int m2,n2, m3,n3
    cdef size_t i, j, k, Ni, Nj, Nk, N, idx, I, J
    cdef unsigned xi0, xi1, yi0, yi1, zi0, zi1

    bidx1, bidx2, bidx3 = bidx
    m2, n2 = block_sizes[1]
    m3, n3 = block_sizes[2]
    Ni, Nj, Nk = len(bidx1), len(bidx2), len(bidx3)
    N = Ni * Nj * Nk
    results = np.empty((2, N), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2
                for k in range(Nk):
                    zi0, zi1 = bidx3[k,0], bidx3[k,1]  # range: m3, n3

                    I = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                    J = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3
                    if not lower_tri or J <= I:
                        IJ[0,idx] = I
                        IJ[1,idx] = J
                        idx += 1
    if idx < N: # TODO: what's the closed formula for N in the triangular case?
        return results[:, :idx]
    else:
        return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_3d(object X, sparsidx, block_sizes):
    """Convert the dense ndarray X from reordered, compressed three-level
    banded form back to a standard sparse matrix format.

    Returns:
        sparse matrix of size `m1*m2*m3 x n1*n2*n3`
    """
    cdef np.int_t[:] sparsidx1, sparsidx2, sparsidx3
    sparsidx1,sparsidx2,sparsidx3 = sparsidx

    cdef int m1,n1, m2,n2, m3,n3
    m1,n1 = block_sizes[0]
    m2,n2 = block_sizes[1]
    m3,n3 = block_sizes[2]

    cdef long[::1] entries_i, entries_j
    cdef size_t i, j, k, si, sj, sk, Ni, Nj, Nk
    cdef size_t idx=0
    Ni,Nj,Nk = X.shape

    assert len(sparsidx1) == Ni
    assert len(sparsidx2) == Nj
    assert len(sparsidx3) == Nk

    cdef size_t xi0, xi1, yi0, yi1, zi0, zi1

    entries_i = np.empty(Ni*Nj*Nk, dtype=int)
    entries_j = np.empty(Ni*Nj*Nk, dtype=int)

    for i in range(Ni):
        si = sparsidx1[i]                       # range: m1*n1
        xi0, xi1 = si // n1, si % n1            # range: m1, n1

        for j in range(Nj):
            sj = sparsidx2[j]                   # range: m2*n2
            yi0, yi1 = sj // n2, sj % n2        # range: m2, n2

            for k in range(Nk):
                sk = sparsidx3[k]               # range: m3*n3
                zi0, zi1 = sk // n3, sk % n3    # range: m3, n3

                entries_i[idx] = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                entries_j[idx] = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3
                idx += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2*m3, n1*n2*n3))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_3d_bidx(object X, bidx, block_sizes):
    """Convert the dense ndarray X from reordered, compressed three-level
    banded form back to a standard sparse matrix format.

    Returns:
        sparse matrix of size `m1*m2*m3 x n1*n2*n3`
    """
    cdef unsigned[:,::1] bidx0, bidx1, bidx2
    bidx0, bidx1, bidx2 = bidx

    cdef int m1,n1, m2,n2, m3,n3
    m1,n1 = block_sizes[0]
    m2,n2 = block_sizes[1]
    m3,n3 = block_sizes[2]

    cdef long[::1] entries_i, entries_j
    cdef size_t mu0, mu1, mu2, MU0, MU1, MU2
    cdef size_t idx=0
    MU0,MU1,MU2 = X.shape

    assert len(bidx0) == MU0
    assert len(bidx1) == MU1
    assert len(bidx2) == MU2

    cdef size_t I[3]
    cdef size_t J[3]

    entries_i = np.empty(MU0*MU1*MU2, dtype=int)
    entries_j = np.empty(MU0*MU1*MU2, dtype=int)

    for mu0 in range(MU0):
        I[0], J[0] = bidx0[mu0,0], bidx0[mu0,1]            # range: m1, n1

        for mu1 in range(MU1):
            I[1], J[1] = bidx1[mu1,0], bidx1[mu1,1]        # range: m2, n2

            for mu2 in range(MU2):
                I[2], J[2] = bidx2[mu2,0], bidx2[mu2,1]    # range: m3, n3

                entries_i[idx] = (I[0] * m2 + I[1]) * m3 + I[2]    # range: m1*m2*m3
                entries_j[idx] = (J[0] * n2 + J[1]) * n3 + J[2]    # range: n1*n2*n3
                idx += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2*m3, n1*n2*n3))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_3d(double[:,:,::1] X, bidx, block_sizes, double[::1] x, double[::1] y):
    cdef unsigned[:,::1] bidx1, bidx2, bidx3
    bidx1,bidx2,bidx3 = bidx

    cdef int m2,n2, m3,n3
    m2,n2 = block_sizes[1]
    m3,n3 = block_sizes[2]

    cdef size_t i, j, k, Ni, Nj, Nk, I, J
    cdef unsigned xi0, xi1, yi0, yi1, zi0, zi1

    Ni,Nj,Nk = X.shape[:3]
    assert len(bidx1) == Ni
    assert len(bidx2) == Nj
    assert len(bidx3) == Nk

    #for i in prange(Ni, schedule='static', nogil=True):
    # the += update is not thread safe! need atomics
    for i in range(Ni):
        xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1

        for j in range(Nj):
            yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2

            for k in range(Nk):
                zi0, zi1 = bidx3[k,0], bidx3[k,1]  # range: m3, n3

                I = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                J = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3

                y[I] += X[i,j,k] * x[J]

