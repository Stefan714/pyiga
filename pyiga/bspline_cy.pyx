# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libcpp.string cimport string
from libc.stdio cimport snprintf

cdef class KnotVector:
    cdef int _p
    cdef double[:] _knots  # typed memoryview
    cdef int size

    def __cinit__(self, int p, np.ndarray[np.float64_t, ndim=1] knots):
        self._p = p
        if not knots.flags['C_CONTIGUOUS']:
            raise ValueError("knots array must be contiguous")
        self._knots = knots
        self.size = len(knots)

        if not self.sanity_check(): raise AssertionError("not a p-open knot vector")

    def __richcmp__(self, other, int op):
        if not isinstance(other, KnotVector):
            return NotImplemented
    
        if op == 0:  # <
            return self.leq(other) and not self.eq(other)
        elif op == 1:  # <=
            return self.leq(other)
        elif op == 2:  # ==
            return self.eq(other)
        elif op == 3:  # !=
            return not self.eq(other)
        elif op == 4:  # >
            return not self.leq(other)
        elif op == 5:  # >=
            return self.eq(other) or not self.leq(other)
        else:
            return NotImplemented

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint eq(self, KnotVector other):
        cdef double[:] knots1 = self._knots
        cdef double[:] knots2 = other._knots
        cdef int i, n = self.size
        cdef double tol = 1e-12
    
        if self._p != other._p or n != other.size:
            return False
        for i in range(n):
            if fabs(knots1[i] - knots2[i]) > tol:
                return False
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint leq(self, KnotVector other):
        cdef double[:] knots1 = self._knots
        cdef double[:] knots2 = other._knots

        cdef double a1 = knots1[0], b1 = knots1[self.size-1], a2 = knots2[0], b2 = knots2[other.size-1] 
        
        cdef int i1=0, i2=0
        cdef int m1, m2, delta_p=other._p - self._p
        cdef int n1=self.size, n2=other.size
        cdef double tol=1e-12
        if delta_p<0: return False
        if a1 > a2 + tol or b2 > b1 + tol: return False
    
        while knots1[i1] < a2 - tol: 
            i1 += 1
    
        while i1 < n1 and i2 < n2:  
            while i2 < n2 and knots2[i2] < knots1[i1] - tol:
                i2 += 1
            if i2 == n2:
                break
    
            while i1 < n1 and knots1[i1] < knots2[i2] - tol:
                i1 += 1
            if i1 == n1:
                break
    
            m1 = 1
            while (i1 + m1) < n1 and fabs(knots1[i1 + m1] - knots1[i1]) < tol:
                m1 += 1
    
            m2 = 1
            while (i2 + m2) < n2 and fabs(knots2[i2 + m2] - knots2[i2]) < tol:
                m2 += 1
    
            if m2 < m1 + delta_p:
                return False
    
            i1 += m1
            i2 += m2
        return True
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __str__(self):
        cdef double[:] knots = self._knots
        cdef:
            int n = self.size
            int i, pos = 0
            char buffer[2048]  # adjust if needed
            int written
    
        written = snprintf(buffer, sizeof(buffer), b"Knot vector (degree %d): [", self._p)
        pos += written
    
        for i in range(n):
            written = snprintf(buffer + pos, sizeof(buffer) - pos, b"%.3f", knots[i])
            pos += written
    
            if i < n - 1:
                buffer[pos] = ord(',')
                buffer[pos+1] = ord(' ')
                pos += 2
    
        buffer[pos] = ord(']')
        pos += 1
        buffer[pos] = 0
    
        return buffer[:pos].decode('ascii')

    @property
    def p(self):
        return self._p

    @property
    def knots(self):
        return self._knots.base

    @property
    def support(self):
        return (self._knots[0], self._knots[self.size-1])

    @property
    def mesh(self):
        return np.unique(self._knots.base)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[double, ndim=1] greville(self):
        cdef double[:] knots = self._knots
        cdef int n = self.size - self._p - 1
        cdef np.ndarray[double, ndim=1] grev_pts = np.empty(n, dtype=np.float64)
        cdef int i, j
        cdef double s

        if self._p == 0:
            for i in range(n):
                grev_pts[i] = (knots[i] + knots[i+1])/2
            return grev_pts

        for i in range(n):
            s = 0.0
            for j in range(1, self._p + 1):  # sum from kv[i+1] to kv[i+p]
                s += knots[i + j]
            grev_pts[i] = s / self._p
        return grev_pts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint sanity_check(self):
        cdef double[:] knots = self._knots
        cdef int i, n = self.size
        cdef double first_val, last_val
    
        # Quick check on knot vector length
        if n < 2 * self._p + 2:
            return False
    
        first_val = knots[0]
        last_val = knots[n - 1]
    
        # Check first p+1 knots equal
        for i in range(1, self._p + 1):
            if knots[i] != first_val:
                return False
    
        # Check last p+1 knots equal
        for i in range(n - self._p - 1, n):
            if knots[i] != last_val:
                return False
    
        # Check nondecreasing sequence
        for i in range(n - 1):
            if knots[i] > knots[i + 1]:
                return False
    
        return True

cpdef KnotVector make_knots(int p, double a, double b, int n, int mult=1):
    print(1)

##################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint pyx_knots_leq(double[:] kv1, int n1, int p1, double a1, double b1,
                         double[:] kv2, int n2, int p2, double a2, double b2):
    cdef int i1=0, i2=0
    cdef int m1, m2, delta_p=p2-p1
    cdef double tol=1e-12
    if delta_p<0: return 0
    if a1 > a2 + tol or b2 > b1 + tol: return 0
    while kv1[i1] < a2 - tol: 
        i1 += 1

    while i1 < n1 and i2 < n2:  
        while i2 < n2 and kv2[i2] < kv1[i1] - tol:
            i2 += 1
        if i2 == n2:
            break

        while i1 < n1 and kv1[i1] < kv2[i2] - tol:
            i1 += 1
        if i1 == n1:
            break

        m1 = 1
        while (i1 + m1) < n1 and fabs(kv1[i1 + m1] - kv1[i1]) < tol:
            m1 += 1

        m2 = 1
        while (i2 + m2) < n2 and fabs(kv2[i2 + m2] - kv2[i2]) < tol:
            m2 += 1

        if m2 < m1 + delta_p:
            return 0

        i1 += m1
        i2 += m2
    return 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int pyx_findspan(double[::1] kv, int p, double u) noexcept nogil:
    cdef int n = kv.shape[0]

    if u >= kv[n - p - 1]:
        return n - p - 2  # last interval

    cdef int a = 0, b = n - 1, c

    while b - a > 1:
        c = a + (b - a) // 2
        if kv[c] > u:
            b = c
        else:
            a = c
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_findspans(double[::1] kv, int p, double[::1] u):
    out = np.empty(u.shape[0], dtype=np.long)
    cdef long[::1] result = out
    cdef int i
    for i in range(u.shape[0]):
        result[i] = pyx_findspan(kv, p, u[i])
    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] bspline_active_deriv_single(object knotvec, double u, int numderiv, double[:,:] result=None) noexcept:
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at a single point `u`"""
    cdef double[::1] kv
    cdef int p, j, r, k, span, rk, pk, fac, j1, j2
    cdef double[:,::1] NDU
    cdef double saved, temp, d
    cdef double[64] left, right, a1buf, a2buf
    cdef double* a1
    cdef double* a2

    kv, p = knotvec.kv, knotvec.p
    assert p < 64, "Spline degree too high"  # need to change constant array sizes above (p+1)

    NDU = np.empty((p+1, p+1), order='C')
    if result is None:
        result = np.empty((numderiv+1, p+1))
    else:
        assert result.shape[0] is numderiv+1 and result.shape[1] is p+1

    span = pyx_findspan(kv, p, u)

    NDU[0,0] = 1.0

    for j in range(1, p+1):
        # Compute knot splits
        left[j-1]  = u - kv[span+1-j]
        right[j-1] = kv[span+j] - u
        saved = 0.0

        for r in range(j):     # For all but the last basis functions of degree j (ndu row)
            # Strictly lower triangular part: Knot differences of distance j
            NDU[j, r] = right[r] + left[j-r-1]
            temp = NDU[r, j-1] / NDU[j, r]
            # Upper triangular part: Basis functions of degree j
            NDU[r, j] = saved + right[r] * temp  # r-th function value of degree j
            saved = left[j-r-1] * temp

        # Diagonal: j-th (last) function value of degree j
        NDU[j, j] = saved

    # copy function values into result array
    for j in range(p+1):
        result[0, j] = NDU[j, p]

    (a1,a2) = a1buf, a2buf

    for r in range(p+1):    # loop over basis functions
        a1[0] = 1.0

        fac = p        # fac = fac(p) / fac(p-k)

        # Compute the k-th derivative of the r-th basis function
        for k in range(1, numderiv+1):
            rk = r - k
            pk = p - k
            d = 0.0

            if r >= k:
                a2[0] = a1[0] / NDU[pk+1, rk]
                d = a2[0] * NDU[rk, pk]

            j1 = 1 if rk >= -1  else -rk
            j2 = k-1 if r-1 <= pk else p - r

            for j in range(j1, j2+1):
                a2[j] = (a1[j] - a1[j-1]) / NDU[pk+1, rk+j]
                d += a2[j] * NDU[rk+j, pk]

            if r <= pk:
                a2[k] = -a1[k-1] / NDU[pk+1, r]
                d += a2[k] * NDU[r, pk]

            result[k, r] = d * fac
            fac *= pk          # update fac = fac(p) / fac(p-k) for next k

            # swap rows a1 and a2
            (a1,a2) = (a2,a1)

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def active_deriv(object knotvec, u, int numderiv):
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at the points `u`.

    Returns an array with shape (numderiv+1, p+1) if `u` is scalar or
    an array with shape (numderiv+1, p+1, len(u)) otherwise.
    """
    cdef double[:,:,:] result
    cdef double[:] u_arr
    cdef int i, n

    if np.isscalar(u):
        return bspline_active_deriv_single(knotvec, u, numderiv)
    else:
        u_arr = u
        n = u.shape[0]
        result = np.empty((numderiv+1, knotvec.p+1, n))
        for i in range(n):
            bspline_active_deriv_single(knotvec, u_arr[i], numderiv, result=result[:,:,i])
        return result

