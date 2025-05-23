cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libc.stdlib cimport rand, srand

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] pyx_doerfler_mark(double[:] x, int n, double theta, double TOL):
    #cdef int[:] idx = np.arange(n, dtype=np.int32)
    #quicksort(x, idx, 0, n-1)
    cdef int[:] idx = np.flip(np.argsort(x).astype(np.int32))
    cdef int i, k
    cdef double total=0, S=0

    for i in range(n):
        total += x[i]**2

    for i in range(n):
        S += x[idx[i]]**2
        k = i
        if S > theta * total:
            while k>0 and (fabs((x[idx[i]]-x[idx[k+1]])/x[idx[i]]) < TOL):       #we go on adding entries that are just 100*TOL% off from the breakpoint entry.
                k+=1
            break
    return idx.base[:(k+1)]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int64_t, ndim=1] pyx_quick_mark(double[:] x, double theta, int[:] idx, int l, int u, double v):
    cdef int i, p
    cdef double sigma = 0.0

    p = (u + l) / 2  # integer division, choosing the slot of the median (rounded to the lower slot of the array)

    quickselect(x, idx, l, u, p)

    for i in range(l,p):
        sigma += x[idx[i]]**2
    if sigma > v:                                                             #If the norm of the larger entries exceeds the total norm we did not find the minimal                                                                                set of entries yet.                  
        return pyx_quick_mark(x, theta, idx, l, p-1, v)                       
    elif sigma + x[idx[p]]**2 > v:                                            #If adding the p-th value (the next biggest entry we can add) suddenly satisfies the                                                                                 condition, we are done.
        return idx.base[:(p+1)]                                                    
    else:                                                                     #We have not reached the desired norm so we have to look further.
        return pyx_quick_mark(x, theta, idx, p+1, u, v-sigma-x[idx[p]]**2)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void swap(int[:] idx, int i, int j):
    cdef int temp = idx[i]
    idx[i] = idx[j]
    idx[j] = temp
    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int partition(double[:] x, int[:] idx, int l, int u, int p):
    cdef double pivot_value = x[idx[p]]
    if p!=u:
        swap(idx, p, u)
    cdef int store_index = l
    cdef int i
    for i in range(l, u):
        if x[idx[i]] > pivot_value:
            swap(idx, store_index, i)
            store_index += 1
    swap(idx, store_index, u)
    return store_index

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quickselect(double[:] x, int[:] idx, int l, int u, int q):
    cdef int p, k
    while l < u:
        p = l + rand() % (u - l + 1)
        #p =  (u + l)/2
        k = partition(x, idx, l, u, p)
        if k == q:
            return
        elif q < k:
            u = k - 1
        else:
            l = k + 1
    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void quicksort(double[:] x, int[:] idx, int l, int u):
    cdef int k, p
    if l < u:
        p = l + rand() % (u - l + 1)
        k = partition(x, idx, l, u, p)
        quicksort(x, idx, l, k-1)
        quicksort(x, idx, k+1, u)
    return