#include <cuda_runtime_api.h>
#include <cstdio>
#include <iostream>

using namespace std;

__global__ void whoami() {
  // offset of block from block 0,0,0
  // clang-format off
  int block_id = blockIdx.x +
                 blockIdx.y * gridDim.x +
                 blockIdx.z * gridDim.y * gridDim.x;

  // offset of First Thread in the block
  int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;

  // offset of thread within the block
  int thread_offset = threadIdx.x +
                      threadIdx.y * blockDim.x +
                      threadIdx.z * blockDim.y * blockDim.x;

  int id = block_offset + thread_offset;

  printf(
    "%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
    id,
    blockIdx.x, blockIdx.y, blockIdx.z, block_id,
    threadIdx.x, threadIdx.y, threadIdx.z, thread_offset
  );
  // clang-format on
}

int main() {
  // Grid   consists of Blocks  ([123]Dims)
  // Each Thread within a Block within the Grid can access global mem (VRAM)
  // Grids handle batch processing, where each block is a batch element
  //
  // Blocks consists of Threads ([123]Dims)
  // Each block has shared memory (visible to all threads in the block)
  const int grid_x = 2, grid_y = 3, grid_z = 4;
  const int block_x = 4, block_y = 4, block_z = 4;
  // Warps & Weft
  // Each warp is inside of a Block and parallelizes 32 Threds
  // Instructions are issued to Warps, that then tell Threads
  // No way of getting around using warps
  // Warp Scheduler makes warps run

  int blocks_per_grid = grid_x * grid_y * grid_z;
  int threads_per_block = block_x * block_y * block_z;

  cout << "Blocks / Grid  : " << blocks_per_grid << endl;
  cout << "Threads / Block: " << threads_per_block << endl;
  cout << "Total Threads  : " << blocks_per_grid * threads_per_block << endl;

  dim3 blocksPerGrid(grid_x, grid_y, grid_z);
  dim3 threadsPerBlock(block_x, block_y, block_z);

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  // Waits to make sure all threads are synced and completed
  cudaDeviceSynchronize();
}
