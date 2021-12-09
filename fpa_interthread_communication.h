#pragma once

#ifndef __CUDA_ARCH__
  #define warp_ballot __ballot
  #define shfl __shfl
  #define shfl_down __shfl_down
  #define shfl_up __shfl_up
#endif

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__<200
    #include "fpa_ic_memory.h"
  #endif

  #if __CUDACC_VER_MAJOR__>=9
    #include "fpa_ic_cooperative.h"
  #else
    #include "fpa_ic_warp_sync.h"
  #endif
#endif
