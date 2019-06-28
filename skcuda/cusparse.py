#!/usr/bin/env python

"""
Python interface to CUSPARSE functions.

Note: this module does not explicitly depend on PyCUDA.
"""

import atexit
import ctypes.util
import platform
from string import Template
import sys
import warnings

import numpy as np

import cuda

# Load library:
_version_list = [10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
if 'linux' in sys.platform:
    _libcusparse_libname_list = ['libcusparse.so'] + \
                                ['libcusparse.so.%s' % v for v in _version_list]
elif sys.platform == 'darwin':
    _libcusparse_libname_list = ['libcusparse.dylib']
elif sys.platform == 'win32':
    if platform.machine().endswith('64'):
        _libcusparse_libname_list = ['cusparse.dll'] + \
            ['cusparse64_%s.dll' % (int(v) if v >= 10 else int(10*v))for v in _version_list]
    else:
        _libcusparse_libname_list = ['cusparse.dll'] + \
            ['cusparse32_%s.dll' % (int(v) if v >= 10 else int(10*v))for v in _version_list]
else:
    raise RuntimeError('unsupported platform')

# Print understandable error message when library cannot be found:
_libcusparse = None
for _libcusparse_libname in _libcusparse_libname_list:
    try:
        if sys.platform == 'win32':
            _libcusparse = ctypes.windll.LoadLibrary(_libcusparse_libname)
        else:
            _libcusparse = ctypes.cdll.LoadLibrary(_libcusparse_libname)
    except OSError:
        pass
    else:
        break
if _libcusparse == None:
    OSError('CUDA sparse library not found')

class cusparseError(Exception):
    """CUSPARSE error"""
    pass

class cusparseStatusNotInitialized(cusparseError):
    """CUSPARSE library not initialized"""
    pass

class cusparseStatusAllocFailed(cusparseError):
    """CUSPARSE resource allocation failed"""
    pass

class cusparseStatusInvalidValue(cusparseError):
    """Unsupported value passed to the function"""
    pass

class cusparseStatusArchMismatch(cusparseError):
    """Function requires a feature absent from the device architecture"""
    pass

class cusparseStatusMappingError(cusparseError):
    """An access to GPU memory space failed"""
    pass

class cusparseStatusExecutionFailed(cusparseError):
    """GPU program failed to execute"""
    pass

class cusparseStatusInternalError(cusparseError):
    """An internal CUSPARSE operation failed"""
    pass

class cusparseStatusMatrixTypeNotSupported(cusparseError):
    """The matrix type is not supported by this function"""
    pass

class cusparseStatusZeroPivot(cusparseError):
    """FIXME"""
    pass

cusparseExceptions = {
    1: cusparseStatusNotInitialized,
    2: cusparseStatusAllocFailed,
    3: cusparseStatusInvalidValue,
    4: cusparseStatusArchMismatch,
    5: cusparseStatusMappingError,
    6: cusparseStatusExecutionFailed,
    7: cusparseStatusInternalError,
    8: cusparseStatusMatrixTypeNotSupported,
    9: cusparseStatusZeroPivot,
    }

# cudaDatatype_t
CUDA_R_16F= 2
CUDA_C_16F= 6
CUDA_R_32F= 0
CUDA_C_32F= 4
CUDA_R_64F= 1
CUDA_C_64F= 5
CUDA_R_8I = 3
CUDA_C_8I = 7
CUDA_R_8U = 8
CUDA_C_8U = 9
CUDA_R_32I= 10
CUDA_C_32I= 11
CUDA_R_32U= 12
CUDA_C_32U= 1

# cusparsePointerMode_t
CUSPARSE_POINTER_MODE_HOST = 0
CUSPARSE_POINTER_MODE_DEVICE = 1

# cusparseAction_t
CUSPARSE_ACTION_SYMBOLIC = 0
CUSPARSE_ACTION_NUMERIC = 1

# Matrix types
# cusparseMatrixType_t
CUSPARSE_MATRIX_TYPE_GENERAL = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

# cusparseFillMode_t
CUSPARSE_FILL_MODE_LOWER = 0
CUSPARSE_FILL_MODE_UPPER = 1

# Whether or not a matrix' diagonal entries are unity:
# cusparseDisagType_t
CUSPARSE_DIAG_TYPE_NON_UNIT = 0
CUSPARSE_DIAG_TYPE_UNIT = 1

# Matrix index bases:
# cusparseIndexBase_t
CUSPARSE_INDEX_BASE_ZERO = 0
CUSPARSE_INDEX_BASE_ONE = 1

# Operation types:
# cusparseOperation_t
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_OPERATION_TRANSPOSE = 1
CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

# Whether or not to parse elements of a dense matrix row or column-wise.
# cusparseDirection_t
CUSPARSE_DIRECTION_ROW = 0
CUSPARSE_DIRECTION_COLUMN = 1

# cusparseHybPartition_t
CUSPARSE_HYB_PARTITION_AUTO = 0
CUSPARSE_HYB_PARTITION_USER = 1
CUSPARSE_HYB_PARTITION_MAX = 2

# cusparseSolvePolicy_t
CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

# cusparseSideMode_t
CUSPARSE_SIDE_LEFT = 0
CUSPARSE_SIDE_RIGHT = 1

# cusparseColorAlg_t
CUSPARSE_COLOR_ALG0 = 0
CUSPARSE_COLOR_ALG1 = 1

# cusparseAlgMode_t
CUSPARSE_ALG0 = 0
CUSPARSE_ALG1 = 1
CUSPARSE_ALG_NAIVE = 0,
CUSPARSE_ALG_MERGE_PATH = 1

# Helper functions:
class cusparseMatDescr(ctypes.Structure):
    _fields_ = [
        ('MatrixType', ctypes.c_int),
        ('FillMode', ctypes.c_int),
        ('DiagType', ctypes.c_int),
        ('IndexBase', ctypes.c_int)
        ]

def cusparseCheckStatus(status):
    """
    Raise CUSPARSE exception

    Raise an exception corresponding to the specified CUSPARSE error
    code.

    Parameters
    ----------
    status : int
        CUSPARSE error code.

    See Also
    --------
    cusparseExceptions

    """

    if status != 0:
        try:
            raise cusparseExceptions[status]
        except KeyError:
            raise cusparseError

_libcusparse.cusparseCreate.restype = int
_libcusparse.cusparseCreate.argtypes = [ctypes.c_void_p]
def cusparseCreate():
    """
    Initialize CUSPARSE.

    Initializes CUSPARSE and creates a handle to a structure holding
    the CUSPARSE library context.

    Returns
    -------
    handle : int
        CUSPARSE library context.

    """

    handle = ctypes.c_int()
    status = _libcusparse.cusparseCreate(ctypes.byref(handle))
    cusparseCheckStatus(status)
    return handle.value

_libcusparse.cusparseDestroy.restype = int
_libcusparse.cusparseDestroy.argtypes = [ctypes.c_int]
def cusparseDestroy(handle):
    """
    Release CUSPARSE resources.

    Releases hardware resources used by CUSPARSE

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    """

    status = _libcusparse.cusparseDestroy(handle)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetVersion.restype = int
_libcusparse.cusparseGetVersion.argtypes = [ctypes.c_int,
                                            ctypes.c_void_p]
def cusparseGetVersion(handle):
    """
    Return CUSPARSE library version.

    Returns the version number of the CUSPARSE library.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.

    Returns
    -------
    version : int
        CUSPARSE library version number.

    """

    version = ctypes.c_int()
    status = _libcusparse.cusparseGetVersion(handle,
                                             ctypes.byref(version))
    cusparseCheckStatus(status)
    return version.value

_libcusparse.cusparseSetStream.restype = int
_libcusparse.cusparseSetStream.argtypes = [ctypes.c_int,
                                                 ctypes.c_int]
def cusparseSetStream(handle, id):
    """
    Sets the CUSPARSE stream in which kernels will run.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    id : int
        Stream ID.

    """

    status = _libcusparse.cusparseSetStream(handle, id)
    cusparseCheckStatus(status)

_libcusparse.cusparseCreateMatDescr.restype = int
_libcusparse.cusparseCreateMatDescr.argtypes = [cusparseMatDescr]
def cusparseCreateMatDescr():
    """
    Initialize a sparse matrix descriptor.

    Initializes the `MatrixType` and `IndexBase` fields of the matrix
    descriptor to the default values `CUSPARSE_MATRIX_TYPE_GENERAL`
    and `CUSPARSE_INDEX_BASE_ZERO`.

    Returns
    -------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    desc = cusparseMatrixDesc()
    status = _libcusparse.cusparseCreateMatDescr(ctypes.byref(desc))
    cusparseCheckStatus(status)
    return desc

_libcusparse.cusparseDestroyMatDescr.restype = int
_libcusparse.cusparseDestroyMatDescr.argtypes = [ctypes.c_int]
def cusparseDestroyMatDescr(desc):
    """
    Releases the memory allocated for the matrix descriptor.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    """

    status = _libcusparse.cusparseDestroyMatDescr(desc)
    cusparseCheckStatus(status)

_libcusparse.cusparseSetMatType.restype = int
_libcusparse.cusparseSetMatType.argtypes = [cusparseMatDescr,
                                            ctypes.c_int]
def cusparseSetMatType(desc, type):
    """
    Sets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.
    type : int
        Matrix type.

    """

    status = _libcusparse.cusparseSetMatType(desc, type)
    cusparseCheckStatus(status)

_libcusparse.cusparseGetMatType.restype = int
_libcusparse.cusparseGetMatType.argtypes = [cusparseMatDescr]    
def cusparseGetMatType(desc):
    """
    Gets the matrix type of the specified matrix.

    Parameters
    ----------
    desc : cusparseMatDescr
        Matrix descriptor.

    Returns
    -------
    type : int
        Matrix type.

    """

    return _libcusparse.cusparseGetMatType(desc)

# Format conversion functions:
_libcusparse.cusparseSnnz.restype = int
_libcusparse.cusparseSnnz.argtypes = [ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      cusparseMatDescr,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p]
def cusparseSnnz(handle, dirA, m, n, descrA, A, lda, 
                 nnzPerRowColumn, nnzTotalDevHostPtr):
    """
    Compute number of non-zero elements per row, column, or dense matrix.

    Parameters
    ----------
    handle : int
        CUSPARSE library context.
    dirA : int
        Data direction of elements.
    m : int
        Rows in A.
    n : int
        Columns in A.
    descrA : cusparseMatDescr
        Matrix descriptor.
    A : pycuda.gpuarray.GPUArray
        Dense matrix of dimensions (lda, n).
    lda : int
        Leading dimension of A.
    
    Returns
    -------
    nnzPerRowColumn : pycuda.gpuarray.GPUArray
        Array of length m or n containing the number of 
        non-zero elements per row or column, respectively.
    nnzTotalDevHostPtr : pycuda.gpuarray.GPUArray
        Total number of non-zero elements in device or host memory.

    """

    # Unfinished:
    if dirA == CUSPARSE_DIRECTION_ROW:
        nnzPerRowColumn = gpuarray.empty((m,),dtype=np.int32)
    elif dirA == CUSPARSE_DIRECTION_COLUMN:
        nnzPerRowColumn = gpuarray.empty((n,),dtype=np.int32)
    else:
        raise ValueError("Unknown dirA")

    status = _libcusparse.cusparseSnnz(handle, dirA, m, n, 
                                       descrA, int(A), lda,
                                       int(nnzPerRowColumn), int(nnzTotalDevHostPtr))
    cusparseCheckStatus(status)
    return nnzPerVector, nnzHost

_libcusparse.cusparseSdense2csr.restype = int
_libcusparse.cusparseSdense2csr.argtypes = [ctypes.c_int,
                                            ctypes.c_int,
                                            ctypes.c_int,
                                            cusparseMatDescr,
                                            ctypes.c_void_p,
                                            ctypes.c_int,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p,
                                            ctypes.c_void_p]
def cusparseSdense2csr(handle, m, n, descrA, A, lda, 
                       nnzPerRow, csrValA, csrRowPtrA, csrColIndA):
    # Unfinished
    pass

_libcusparse.cusparseDcsrmv.restype = int
_libcusparse.cusparseDcsrmv.argtypes = [ctypes.c_int, # handle
                                        ctypes.c_int, # transA
                                        ctypes.c_int, # m
                                        ctypes.c_int, # n
                                        ctypes.c_int, # nnz
                                        ctypes.c_double_p, # alpha
                                        cusparseMatDescr, # descrA
                                        ctypes.c_double_p, # csrValA
                                        ctypes.c_int_p, # csrRowPtrA
                                        ctypes_c_int_p, # csrColIndA
                                        ctypes_c_double_p, # x
                                        ctypes_c_double_p, # beta
                                        ctypes_c_double_p] # y

def cusparseDcsrmv(handle, transA, m, n, nnz, alpha,
                   descrA, csrValA, csrRowPtrA, csrColIndA,
                   x,beta,y):
    status = _libcusparse.cusparseDcsrmv(handle, transA, m, n,  nnz,
                                         int(alpha), descrA, int(csrValA),
                                         int(csrRowPtrA), int(csrColIndA),
                                         int(x),int(beta),int(y))
    cusparseCheckStatus(status)
