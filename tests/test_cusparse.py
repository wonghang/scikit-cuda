#!/usr/bin/env python

"""
Unit tests for skcuda.cublas
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as slinalg

drv.init()

_SEPS = np.finfo(np.float32).eps
_DEPS = np.finfo(np.float64).eps

import skcuda.cusparse as cusparse
import skcuda.misc as misc

class test_cusparse(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = make_default_context()
        cls.cusparse_handle = cusparse.cusparseCreate()
        cusparse.cusparseSetPointerMode(cls.cusparse_handle,cusparse.CUSPARSE_POINTER_MODE_HOST)

    @classmethod
    def tearDownClass(cls):
        cusparse.cusparseDestroy(cls.cusparse_handle)
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(21314)

    def test_cusparseDcsrmv(self):
        # original example on scipy
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6],dtype=np.float64)
        y = np.random.rand(3)
        
        A = csr_matrix((data,indices,indptr),shape=(3,3))
        A = slinalg.aslinearoperator(A)

        result_matvec = A.matvec(y)
        result_rmatvec = A.rmatvec(y)

        m = 3
        n = 3
        gpu_indptr = gpuarray.to_gpu(indptr.astype(np.int32))
        gpu_indices = gpuarray.to_gpu(indices.astype(np.int32))
        gpu_data = gpuarray.to_gpu(data)
        gpu_y = gpuarray.to_gpu(y)
        gpu_result_matvec = gpuarray.empty((m,),dtype=np.float64)
        gpu_result_rmatvec = gpuarray.empty((m,),dtype=np.float64)
        
        descrA = cusparse.cusparseCreateMatDescr()
        cusparse.cusparseSetMatType(descrA,cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparse.cusparseSetMatFillMode(descrA,cusparse.CUSPARSE_FILL_MODE_LOWER) # not important?
        cusparse.cusparseSetMatDiagType(descrA,cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)
        cusparse.cusparseSetMatIndexBase(descrA,cusparse.CUSPARSE_INDEX_BASE_ZERO)

        nnz = gpu_data.shape[0]
        cusparse.cusparseDcsrmv(self.cusparse_handle,
                                cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                m,
                                n,
                                nnz,
                                1.0,
                                descrA,
                                gpu_data.gpudata,
                                gpu_indptr.gpudata,
                                gpu_indices.gpudata,
                                gpu_y.gpudata,
                                0.0,
                                gpu_result_matvec.gpudata)
        cusparse.cusparseDcsrmv(self.cusparse_handle,
                                cusparse.CUSPARSE_OPERATION_TRANSPOSE,
                                m,
                                n,
                                nnz,
                                1.0,
                                descrA,
                                gpu_data.gpudata,
                                gpu_indptr.gpudata,
                                gpu_indices.gpudata,
                                gpu_y.gpudata,
                                0.0,
                                gpu_result_rmatvec.gpudata)
        
        cusparse.cusparseDestroyMatDescr(descrA)
        assert np.allclose(result_matvec,gpu_result_matvec.get())
        assert np.allclose(result_rmatvec,gpu_result_rmatvec.get())

def suite():
    context = make_default_context()
    device = context.get_device()
    context.pop()

    s = TestSuite()
    if misc.get_compute_capability(device) >= 1.3:
        s.addTest(test_cusparse('test_cusparseDcsrmv'))
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
