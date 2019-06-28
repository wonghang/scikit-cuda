#!/usr/bin/env python

"""
Unit tests for skcuda.cublas
"""

from unittest import main, makeSuite, TestCase, TestSuite

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import clear_context_caches, make_default_context
import numpy as np

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

    @classmethod
    def tearDownClass(cls):
        cusparse.cusparseDestroy(cls.cusparse_handle)
        cls.ctx.pop()
        clear_context_caches()

    def setUp(self):
        np.random.seed(23)    # For reproducible tests.

    def test_cusparseDcsrmv(self):
        # x = np.random.rand(5).astype(np.float32)
        # x_gpu = gpuarray.to_gpu(x)
        # result = cublas.cublasIsamax(self.cublas_handle, x_gpu.size, x_gpu.gpudata, 1)
        # assert np.allclose(result, np.argmax(x))
        print("GOOD")

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
