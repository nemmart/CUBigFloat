#ifdef __CUDA_ARCH__
#warning using shared memory __shfl
#endif

__shared__ uint32_t shuffle_data[128];

__device__ __forceinline__ warp_ballot(bool bit) {
  __ballot(bit);
}

__device__ __forceinline__ uint32_t shfl(uint32_t x, int32_t src) {
  shuffle_data[threadIdx.x]=x;
  return shuffle_data[(threadIdx.x & ~0x1F) + (src & 0x1F)];
}

__device__ __forceinline__ uint32_t shfl_down(uint32_t x, int32_t count) {
  shuffle_data[threadIdx.x]=x;
  if((threadIdx.x & 0x1F) + (count & 0x1F)<=31)
    x=shuffle_data[threadIdx.x + (count & 0x1F)];
  return x;
}

__device__ __forceinline__ uint32_t shfl_up(uint32_t x, int32_t count) {
  shuffle_data[threadIdx.x]=x;
  if((threadIdx.x & 0x1F)>=(count & 0x1F))
    x=shuffle_data[threadIdx.x - (count & 0x1F)];
  return x;
}

