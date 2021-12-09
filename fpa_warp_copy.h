using namespace gpu_fpa::warp_math;

template<uint32_t size>
__device__ __forceinline__ void fpa_warp_copy_a(fpa_arguments arguments, uint32_t a_exp) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t registers[size];
  Limbs    value(registers);

  warp_int_load<size>(value, arguments.a_limbs + warp*size*32);
  warp_int_store<size>(arguments.r_limbs + warp*size*32, value);
  if(warp_thread==0)
    arguments.r_exps[warp]=a_exp;
}

template<uint32_t size>
__device__ __forceinline__ void fpa_warp_copy_b(fpa_arguments arguments, uint32_t b_exp) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t registers[size];
  Limbs    value(registers);

  warp_int_load<size>(value, arguments.b_limbs + warp*size*32);
  warp_int_store<size>(arguments.r_limbs + warp*size*32, value);
  if(warp_thread==0)
    arguments.r_exps[warp]=b_exp;
}

