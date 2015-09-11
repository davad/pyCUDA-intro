// Add in parallel

// Multiple blocks, one thread each
#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int N)  {
  int id = blockIdx.x*blockDim.x + threadIdx.x;

  if( id < N ){
    c[id] = a[id] + b[id];
  }
}

int main() {
  // Vector size
  int N = 100000;

  // Host vectors
  int *a, *b;
  int *c; // output vector

  // Device vectors
  int *d_a, *d_b;
  int *d_c;  // device copies

  // Size in bytes of each vector
  size_t size = N*sizeof(int);

  // Allocate host memory
  a = (int*)malloc(size);
  b = (int*)malloc(size);
  c = (int*)malloc(size);

  // Allocate device memory
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  // Initialize host vectors
  for( int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = -(i-1);
  }

  // Copy host input vectors to device
  cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
  cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

  // Number of thread per block
  int threadCount = 128;

  // Number of blocks per grid
  int blockCount = (int)ceil((float)N/threadCount);

  // Launch add() on GPU
  add<<<blockCount,threadCount>>>(d_a, d_b, d_c, N);

  // Copy result to host
  cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost);

  // Results should sum up to N
  int sum = 0;
  for (int i = 0; i < N; i++) {
    if (i < 5) {
      printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    sum += c[i];
  }
  printf("...\n");

  printf("Should be %d\nResults: %d\n", N,sum);

  // Cleanup host
  free(a);
  free(b);
  free(c);

  // Cleanup device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
