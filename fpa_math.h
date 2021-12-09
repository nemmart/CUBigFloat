__device__ __forceinline__ uint32_t APPROX32(uint32_t d) {
  uint64_t a=d;

  // if(d==0x80000000)
  //   return 0xFFFFFFFF;
  // return ceil(2^64 / d) - 2^32

  if(d==0x80000000)
    return 0xFFFFFFFF;

  a=-a / d;
  return ((uint32_t)a)+2;
}

__device__ __forceinline__ uint32_t DIV32(uint32_t hi, uint32_t lo, uint32_t d, uint32_t approx) {
  PTXInliner inliner;
  uint32_t   q, add, ylo, yhi, c, zero=0;

  // q=MIN(0xFFFFFFFF, HI(approx * hi) + hi + ((lo<d) ? 1 : 2));

  add=(lo<d) ? 1 : 2;
  inliner.MULHI(q, hi, approx);
  q=q+add;                             // the only case where this can carry out is if hi and approx are both 0xFFFFFFFF and add=2
  q=q+hi;                              // but in this case, q will end up being 0xFFFFFFFF, which is what we want
  q=(q>=hi) ? q : 0xFFFFFFFF;          // if q+hi carried out, set q to 0xFFFFFFFF

  inliner.MULLO(ylo, q, d);
  inliner.MULHI(yhi, q, d);

  // correction step:
  //
  // if(yhi:ylo>hi:lo) {
  //   q--;
  //   yhi:ylo-=d;
  // }
  // if(yhi:ylo>hi:lo)
  //   q--;
  // return q;

  inliner.SUB_CC(lo, lo, ylo);
  inliner.SUBC_CC(hi, hi, yhi);
  inliner.SUBC(c, zero, zero);
  q=q+c;

  inliner.ADD_CC(lo, lo, d);
  inliner.ADDC_CC(hi, hi, zero);
  inliner.ADDC(q, q, c);

  return q;
}

__device__ __forceinline__ uint32_t SQRT32(uint32_t x) {
  PTXInliner inliner;
  uint32_t   s=0, t, r, neg=~x;
  int32_t    check;

  s=(x>=0x40000000) ? 0x8000 : 0;
  #pragma unroll
  for(int32_t bit=0x4000;bit>=0x1;bit=bit/2) {
    t=s | bit;
    inliner.MADLO(r, t, t, neg);   // neg=-(x+1)
    check=r;
    s=(check<0) ? t : s;
  }
  return s;
}

__device__ __forceinline__ uint32_t SQRT64(uint32_t &hi, uint32_t &lo) {
  PTXInliner inliner;
  uint32_t   s=SQRT32(hi), shifted=s<<16;
  uint32_t   rhi, rlo;
  int32_t    sign;
  uint32_t   half=0xFFFF0000, ones=0xFFFFFFFF;

  // has the requirement that hi>=0x40000000

  inliner.MADLO_CC(rlo, shifted, shifted, half);
  inliner.MADHIC(rhi, shifted, shifted, ones);

  inliner.SUB_CC(rlo, lo, rlo);
  inliner.SUBC(rhi, hi, rhi);

  if(rhi>=0x1FFFD)
    s=shifted+0xFFFF;
  else {
    #if __CUDA_ARCH__<350
      rlo=rlo>>17;
      rhi=(rhi<<15) | rlo;
    #else
      rhi=__funnelshift_lc(rlo, rhi, 15);
    #endif
    s=shifted+rhi/s;
  }

  inliner.MULLO(rlo, s, s);
  inliner.MULHI(rhi, s, s);
  inliner.SUB_CC(rlo, lo, rlo);
  inliner.SUBC(rhi, hi, rhi);
  sign=rhi;
  s=s + ((sign<0) ? 0xFFFFFFFF : 0);

  inliner.MULLO(rlo, s, s);
  inliner.MULHI(rhi, s, s);
  inliner.SUB_CC(lo, lo, rlo);
  inliner.SUBC(hi, hi, rhi);
  sign=hi;
  s=s + ((sign<0) ? 0xFFFFFFFF : 0);

  return s;
}

__device__ __forceinline__ uint32_t SQRTQ32(uint32_t hi, uint32_t lo, uint32_t d, uint32_t approx) {
  PTXInliner inliner;
  uint32_t   zero=0, one=1;

  inliner.ADD_CC(lo, lo, one);
  inliner.ADDC(hi, hi, zero);

  if(hi>=2)
    return 0xFFFFFFFF;

  #if __CUDA_ARCH__<350
    hi=(hi<<31) + (lo>>1);
    lo=lo<<31;
  #else
    hi=__funnelshift_lc(lo, hi, 31);
    lo=lo<<31;
  #endif

  return DIV32(hi, lo, d, approx);
}
