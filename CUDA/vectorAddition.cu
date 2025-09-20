#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define SIZE 10000000
#define BLOCK_SIZE 256
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void new_vec(float** v) {
  *v = (float*)malloc(sizeof(float) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    (*v)[i] = (float)rand() / RAND_MAX;
  }
}

// Kernel to add two vectors and store in third
__global__ void vec_add_gpu(float* a, float* b, float* c) {
  int idx = blockIdx.x * blockDim.x + blockIdx.x;
  if (idx < SIZE) {
    c[idx] = a[idx] + b[idx];
  }
}

// Same as abv but with 3D Grid
//
// Performance of the Kernel depends on the no of
// operations that happen in a single thread.
//
// For addition of two vectors, using 3D Kernel
// won't improve the performance as compared to 1D,
// since the no. of operations (+, *, =) increases
// significantly in 3D as compared to 1D per thread.
__global__ void vec_add_gpu_3D(float* a,
                               float* b,
                               float* c,
                               int nx,
                               int ny,
                               int nz) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  // 3 adds, 3 multiplies, 3 stores in the thread;

  if (i < nx && j < ny && k < nz) {
    int idx = i + j * nx + k * nx * ny;
    if (idx < nx * ny * nz) {
      c[idx] = a[idx] + b[idx];
    }
  }
}

void vec_add_cpu(float* a, float* b, float* c) {
  for (int i = 0; i < SIZE; i++) {
    c[i] = a[i] + b[i];
  }
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  float *h_a = 0, *h_b = 0, *h_c = 0;
  float *d_a = 0, *d_b = 0, *d_c = 0;
  size_t size = sizeof(float) * SIZE;

  srand(time(NULL));

  new_vec(&h_a);
  new_vec(&h_b);
  h_c = (float*)malloc(size);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // For 1D Block
  int num_blocks = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

  int nx = 100, ny = 100, nz = 1000;  // SIZE = 100 * 100 * 1000
  dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
  // clang-format off
  dim3 num_blocks_3d(
    (nx + block_size_3d.x - 1) / block_size_3d.x,
    (ny + block_size_3d.y - 1) / block_size_3d.y,
    (nz + block_size_3d.z - 1) / block_size_3d.z
  );
  // clang-format on

  // Warm up is necessary for good benchmarking
  // The overhead for launching first kernel can add
  // many ms of latency
  printf("Warm up initialized\n");
  for (int i = 0; i < 20; i++) {
    vec_add_cpu(h_a, h_b, h_c);
    vec_add_gpu<<<num_blocks, BLOCK_SIZE>>>(h_a, h_b, h_c);
    cudaDeviceSynchronize();
    vec_add_gpu_3D<<<num_blocks_3d, block_size_3d>>>(h_a, h_b, h_c, nx, ny, nz);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU vec addition:\n");
  double cpu_time = 0;
  for (int i = 0; i < 20; i++) {
    double start_time = get_time();
    vec_add_cpu(h_a, h_b, h_c);
    double end_time = get_time();
    cpu_time += end_time - start_time;
  }
  cpu_time = cpu_time / 20.0;
  printf("Avg Time: %f ms\n", cpu_time * 1000);

  printf("Benchmarking GPU-1D vec addition:\n");
  double gpu_time = 0;
  for (int i = 0; i < 20; i++) {
    cudaMemset(d_c, 0, size);
    double start_time = get_time();
    vec_add_gpu<<<num_blocks, BLOCK_SIZE>>>(h_a, h_b, h_c);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_time += end_time - start_time;
  }
  gpu_time = gpu_time / 20.0;
  printf("Avg Time: %f ms\n", gpu_time * 1000);

  printf("Benchmarking GPU-3D vec addition:\n");
  double gpu_time_3d = 0;
  for (int i = 0; i < 20; i++) {
    cudaMemset(d_c, 0, size);
    double start_time = get_time();
    vec_add_gpu_3D<<<num_blocks_3d, block_size_3d>>>(h_a, h_b, h_c, nx, ny, nz);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_time_3d += end_time - start_time;
  }
  gpu_time_3d = gpu_time_3d / 20.0;
  printf("Avg Time: %f ms\n", gpu_time_3d * 1000);

  printf("GPU-1D %5ldx Faster than CPU\n", (long)(cpu_time / gpu_time));
  printf("GPU-3D %5ldx Faster than CPU\n", (long)(cpu_time / gpu_time_3d));
  printf("GPU-3D %5ldx Faster than GPU-1D\n", (long)(gpu_time / gpu_time_3d));

  // Warm up initialized
  // Benchmarking CPU vec addition:
  // Avg Time: 9.547453 ms
  // Benchmarking GPU-1D vec addition:
  // Avg Time: 0.001845 ms
  // Benchmarking GPU-3D vec addition:
  // Avg Time: 0.001806 ms
  // GPU-1D  5175x Faster than CPU
  // GPU-3D  5286x Faster than CPU
  // GPU-3D     1x Faster than GPU-1D

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
