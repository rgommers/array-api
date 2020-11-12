"""
Use many-libs conda env, see: https://gist.github.com/rgommers/347b40695b526ff3993a61d36bdb1c6e

- Overview of APIs
- Semantics
- Issues:
  - bool not supported (JAX doesn't support bool, PyTorch converts to int8)
  - complex not supported
- Advantages
- Possibility to add refcounting (see message Tianqi)
- Python API

"""

# Single library round-trip
# -------------------------

# JAX
import jax
import jax.dlpack

x = jax.numpy.arange(3)
# Note: take_ownership=False (default) requires jaxlib 0.1.57, released 11 Nov 2020
#       this is a mode where the user guarantees not to mutate the buffer
#       see https://github.com/google/jax/issues/4636
capsule = jax.dlpack.to_dlpack(x, take_ownership=True)
x2 = jax.dlpack.from_dlpack(capsule)


# PyTorch
import torch
import torch.utils.dlpack

x = torch.arange(3)
capsule = torch.utils.dlpack.to_dlpack(x)
x2 = torch.utils.dlpack.from_dlpack(capsule)


 # CuPy
import cupy as cp

x = cp.arange(3)
capsule = x.toDlpack()
x2 = cp.fromDlpack(capsule)


# TensorFlow
import tensorflow as tf

x = tf.range(3)
capsule = tf.experimental.dlpack.to_dlpack(x)
x2 = tf.experimental.dlpack.from_dlpack(capsule)


# MXNet
import mxnet

x = mxnet.nd.arange(3)
# MXNet also has to_dlpack_for_write(), with identical docs (?)
# Looks like the same idea as JAX: keep ownership if _for_read(),
#                                  consume if _for_write().
capsule = x.to_dlpack_for_read()
x2 = mxnet.nd.from_dlpack(capsule)


# Moving between libraries
# ------------------------

# JAX to PyTorch
# ``````````````

j = jax.numpy.arange(3)
capsule = jax.dlpack.to_dlpack(j, take_ownership=True)
t = torch.utils.dlpack.from_dlpack(capsule)

# This will raise a RuntimeError, can only consume a capsule once:
import pytest
with pytest.raises(RuntimeError):
    t2 = torch.utils.dlpack.from_dlpack(capsule)


with pytest.raises(RuntimeError):
    # Will raise: RuntimeError: Invalid argument: Invalid buffer passed to
    #             Execute() as argument 0 to replica 0: Invalid argument: Hold
    #             requested on deleted or donated buffer
    j[0]


# PyTorch to CuPy
# ```````````````
x = cp.arange(3)
capsule = x.toDlpack()
t2 = torch.utils.dlpack.from_dlpack(capsule)

# This will actually share memory:
t2[0] = 3
assert x[0] == 3

# Now see if `x` is still available after `t2` goes out of scope:
x = cp.arange(3)

def somefunc(x):
    capsule = x.toDlpack()
    t2 = torch.utils.dlpack.from_dlpack(capsule)
    t2[0] = 3
    return None

somefunc(x)
x += 1
assert x[0] == 4
# Note: this appears to work - but the deleter should have been called when t2
#       goes out of scope (or is garbage collected?), so this may result in
#       flaky behaviour.


# TensorFlow - PyTorch interop
# ----------------------------
import tensorflow as tf
import torch.utils.dlpack

x = tf.range(3)
capsule = tf.experimental.dlpack.to_dlpack(x)
x2 = torch.utils.dlpack.from_dlpack(capsule)

x2 += 1
assert x2[2] == 3  # sanity check we got the data

capsule2 = torch.utils.dlpack.to_dlpack(x2)
x3 = tf.experimental.dlpack.from_dlpack(capsule2)

assert x3[2] == 3



"""
# CuPy implementation snippet (https://github.com/cupy/cupy/blob/master/cupy/core/dlpack.pyx):
cdef class DLPackMemory(memory.BaseMemory):

    # Memory object for a dlpack tensor.
    # This does not allocate any memory.

    cdef DLManagedTensor* dlm_tensor
    cdef object dltensor

    def __init__(self, object dltensor):
        self.dltensor = dltensor
        ...
        # Make sure this capsule will never be used again.
        cpython.PyCapsule_SetName(dltensor, 'used_dltensor')

    def __dealloc__(self):
        self.dlm_tensor.deleter(self.dlm_tensor)

cpdef ndarray fromDlpack(object dltensor):
   mem = DLPackMemory(dltensor)
   ...
"""


