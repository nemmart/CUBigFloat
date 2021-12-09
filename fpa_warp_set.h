__device__ __forceinline__ uint32_t fpa_warp_set_ui_internal(uint32_t *r, uint32_t size, uint64_t ui) {
  uint32_t warp_thread=threadIdx.x & 0x1F;
  uint32_t registers[1];
  Limbs    x(registers);
  uint32_t shift;

  if(ui==0)
    return FPA_ZERO;

  shift=__clzll(ui);
  ui=ui<<shift;

  if(size==1) {
    if(warp_thread==31)
      x.limbs[0]=ui>>32;
    else if(warp_thread==30)
      x.limbs[0]=ui;
    else
      x.limbs[0]=0;
    warp_int_store<1>(r, x);
  }
  else {
    x.limbs[0]=0;
    for(int32_t chunk=0;chunk<size;chunk++) {
      if(chunk==size-2 && warp_thread==31)
        x.limbs[0]=ui;
      if(chunk==size-1 && warp_thread==31)
        x.limbs[0]=ui>>32;
      warp_int_store<1>(r + chunk*32, x);
    }
  }

  return (FPA_BIAS-shift+64)*2;
}

__device__ __forceinline__ uint32_t fpa_warp_set_si_internal(uint32_t *r, uint32_t size, int64_t si) {
  if(si>0)
    return fpa_warp_set_ui_internal(r, size, si);
  else if(si<0)
    return fpa_warp_set_ui_internal(r, size, -si) ^ 0x01;
  else
    return FPA_ZERO;
}

__device__ void fpa_warp_set_ui(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint64_t ui;
  uint32_t r_exp;

  if(warp>arguments.count)
    return;

  ui=((uint64_t *)arguments.values)[warp];
  r_exp=fpa_warp_set_ui_internal(arguments.r_limbs + warp*(arguments.precision>>5), arguments.precision>>10, ui);

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

__device__ void fpa_warp_set_si(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  int64_t  si;
  uint32_t r_exp;

  if(warp>arguments.count)
    return;

  si=((int64_t *)arguments.values)[warp];
  r_exp=fpa_warp_set_si_internal(arguments.r_limbs + warp*(arguments.precision>>5), arguments.precision>>10, si);

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

__device__ void fpa_warp_set_float(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  float    value, normalized;
  uint32_t r_exp;
  int32_t  exp;
  int64_t  si;

  if(warp>arguments.count)
    return;

  value=((float *)arguments.values)[warp];

  if(value==0)
    r_exp=FPA_ZERO;
  else if(isnan(value))
    r_exp=FPA_NAN;
  else if(isinf(value)==-1)
    r_exp=FPA_NEG_INF;
  else if(isinf(value)==1)
    r_exp=FPA_POS_INF;
  else {
    normalized=frexp(value, &exp);
    si=(int64_t)ldexp(normalized, 24);
    r_exp=fpa_warp_set_si_internal(arguments.r_limbs + warp*(arguments.precision>>5), arguments.precision>>10, si) + (exp-24)*2;
  }

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

__device__ void fpa_warp_set_double(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  double   value, normalized;
  uint32_t r_exp;
  int32_t  exp;
  int64_t  si;

  if(warp>arguments.count)
    return;

  value=((double *)arguments.values)[warp];
  if(value==0)
    r_exp=FPA_ZERO;
  else if(isnan(value))
    r_exp=FPA_NAN;
  else if(isinf(value)==-1)
    r_exp=FPA_NEG_INF;
  else if(isinf(value)==1)
    r_exp=FPA_POS_INF;
  else {
    normalized=frexp(value, &exp);
    si=(int64_t)ldexp(normalized, 53);
    r_exp=fpa_warp_set_si_internal(arguments.r_limbs + warp*(arguments.precision>>5), arguments.precision>>10, si) + (exp-53)*2;
  }

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

__device__ void fpa_warp_set(fpa_arguments arguments) {
  uint32_t            warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t            registers[8];
  Limbs               source(registers);
  uint32_t            r_exp, round;
  uint32_t            result_size=arguments.precision>>10, source_size=arguments.set_source_precision>>10;
//  __shared__ uint32_t transpose[GEOMETRY / 32 * 256];

  if(warp>arguments.count)
    return;

  r_exp=arguments.a_exps[warp];
  if(r_exp<10) {
    if(warp_thread==0)
      arguments.r_exps[warp]=r_exp;
    return;
  }

  #pragma unroll
  for(int32_t limb=0;limb<8;limb++) {
    if(limb<8-source_size)
      source.limbs[limb]=0;
    else {
      Limbs current(registers+limb);

      warp_int_load<1>(current, arguments.a_limbs + (warp+1)*source_size*32 + (limb-8)*32);
    }
  }

  if(source_size>result_size) {
    uint32_t lsw, roundWord, stickyWord, value;
    int32_t base=(warp_thread+1)*source_size;

    #pragma unroll
    for(int32_t limb=0;limb<7;limb++) {
      if(limb==7-result_size) {
        lsw=shfl(source.limbs[limb+1], 0);
        roundWord=shfl(source.limbs[limb], 31);
      }
    }

    // ok to here
    value=(base-1<result_size*32) ? source.limbs[7] : 0;
    #pragma unroll
    for(int32_t limb=6;limb>=0;limb--)
      value=(base-(8-limb)<result_size*32) ? value | source.limbs[limb] : value;
    stickyWord=warp_ballot(value!=0);

    round=fpa_round(arguments.mode, r_exp & 0x01, lsw, roundWord, stickyWord);

    #pragma unroll
    for(int32_t limb=0;limb<8;limb++)
      if(limb<8-source_size)
        source.limbs[limb]=0xFFFFFFFF;
    warp_int_fast_round<8>(source, round);
  }

/*
  shared=transpose + (threadIdx.x>>5)*256;
  #pragma unroll
  for(int32_t limb=0;limb<8;limb++) {
    shared[threadIdx.x &
*/
  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}