# Add with a single thread on the GPU

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# Define CUDA function
mod = SourceModule("""
__global__ void add(int *a, int *b, int *c)  {
  int id = blockIdx.x;

  c[id] = a[id] + b[id];
}""")

func = mod.get_function("add")

# Vector size
N = 10

# Host vectors
a = numpy.array(range(0,N))
b = 1 - a
c = numpy.zeros(N)

a = a.astype(numpy.int32)
b = b.astype(numpy.int32)
c = c.astype(numpy.int32)

# Allocate on device
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)

# Copy from host to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

func(a_gpu, b_gpu, c_gpu, block=(1,1,1), grid=(N,1))

# Copy result to host
cuda.memcpy_dtoh(c, c_gpu)

# Display results
print("Should be %d" % N)
print("Results: %d" % numpy.sum(c))
