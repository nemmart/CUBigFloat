#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ __forceinline__ uint32_t warp_ballot(bool expr) {
  thread_block          block=this_thread_block();
  thread_block_tile<32> warp=tiled_partition<32>(block);

  return warp.ballot(expr);
}

__device__ __forceinline__ uint32_t shfl(uint32_t x, int32_t src) {
  thread_block          block=this_thread_block();
  thread_block_tile<32> warp=tiled_partition<32>(block);

  return warp.shfl(x, src);
}

__device__ __forceinline__ uint32_t shfl_down(uint32_t x, int32_t count) {
  thread_block          block=this_thread_block();
  thread_block_tile<32> warp=tiled_partition<32>(block);

  return warp.shfl_down(x, count);
}

__device__ __forceinline__ uint32_t shfl_up(uint32_t x, int32_t count) {
  thread_block          block=this_thread_block();
  thread_block_tile<32> warp=tiled_partition<32>(block);

  return warp.shfl_up(x, count);
}
