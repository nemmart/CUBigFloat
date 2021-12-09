__device__ __forceinline__ void make_long(uint64_t &x, uint32_t hi, uint32_t lo) {
  uint64_t local;

  asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(local) : "r"(lo), "r"(hi));
  x=local;
}

__device__ __forceinline__ void make_hilo(uint32_t &hi, uint32_t &lo, uint64_t x) {
  uint32_t lHi, lLo;

  asm volatile ("mov.b64 {%0,%1},%2;" : "=r"(lLo), "=r"(lHi) : "l"(x));
  hi=lHi;
  lo=lLo;
}

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
  uint32_t   q, add, ylo, yhi;
  uint64_t   x, y;

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

  make_long(x, hi, lo);
  make_long(y, yhi, ylo);

  if(y>x) {
    q--;
    y-=d;
  }
  if(y>x)
    q--;
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
  uint64_t   s, t, x;

  make_long(x, hi, lo);
  s=(hi>=0x40000000) ? 0x80000000 : 0;
  #pragma unroll
  for(int32_t bit=0x40000000;bit>=0x1;bit=bit/2) {
    t=s | bit;
    s=(x>=t*t) ? t : s;
  }
  x-=s*s;
  make_hilo(hi, lo, x);
  return s;
}

__device__ __forceinline__ uint32_t SQRTQ32(uint32_t hi, uint32_t lo, uint32_t d, uint32_t approx) {
  PTXInliner inliner;
  uint64_t   x;

  make_long(x, hi, lo);
  x++;
  make_hilo(hi, lo, x);

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
