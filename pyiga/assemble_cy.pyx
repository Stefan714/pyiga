cimport cython

import numpy as np
cimport numpy as np

import scipy

cpdef object pyx_right_inverse_C0_Basis(int[:] indptr, int[:] indices, double[:] data, int n, int m):
    cdef int j,r,k=0

    cdef int[:] ii = np.empty(m, dtype=np.int32)        
    cdef int[:] jj = np.empty(m, dtype=np.int32)        
    cdef double[:] vv = np.empty(m, dtype=np.float64) 
    
    for j in range(m):
        jj[j]=j
        vv[k]=1.
        for r in range(indptr[j],indptr[j+1]):
            if abs(1-data[r])<1e-12:
                ii[k]=indices[r]
                k+=1
                break
    return scipy.sparse.coo_matrix((vv.base,(jj.base,ii.base)),(m,n))

#cpdef object pyx_eval_nodal()
                
                
                
            
            