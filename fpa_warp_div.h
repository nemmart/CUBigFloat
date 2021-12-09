__constant__ uint8_t div_action_table[81] = {
   /*               b=+value       -value         zero          -1/inf,        +1/inf         -inf,           +inf,          nan,        snan  */
   /* a=+value */   FPA_DIV,       FPA_DIV,       FPA_NAN,      FPA_NEG_INF,   FPA_POS_INF,   FPA_NEG_TINY,   FPA_POS_TINY,  FPA_NAN,    FPA_SNAN,
   /*   -value */   FPA_DIV,       FPA_DIV,       FPA_NAN,      FPA_POS_INF,   FPA_NEG_INF,   FPA_POS_TINY,   FPA_NEG_TINY,  FPA_NAN,    FPA_SNAN,
   /*     zero */   FPA_ZERO,      FPA_ZERO,      FPA_NAN,      FPA_ZERO,      FPA_ZERO,      FPA_ZERO,       FPA_ZERO,      FPA_NAN,    FPA_SNAN,
   /*   -1/inf */   FPA_NEG_TINY,  FPA_POS_TINY,  FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_POS_TINY,   FPA_NEG_TINY,  FPA_NAN,    FPA_SNAN,
   /*   +1/inf */   FPA_POS_TINY,  FPA_NEG_TINY,  FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NEG_TINY,   FPA_POS_TINY,  FPA_NAN,    FPA_SNAN,
   /*     -inf */   FPA_POS_INF,   FPA_NEG_INF,   FPA_NAN,      FPA_POS_INF,   FPA_NEG_INF,   FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     +inf */   FPA_NEG_INF,   FPA_POS_INF,   FPA_NAN,      FPA_NEG_INF,   FPA_POS_INF,   FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*      nan */   FPA_NAN,       FPA_NAN,       FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     snan */   FPA_SNAN,      FPA_SNAN,      FPA_SNAN,     FPA_SNAN,      FPA_SNAN,      FPA_SNAN,       FPA_SNAN,      FPA_SNAN,   FPA_SNAN,
};

__device__ __forceinline__ void fpa_warp_div_1(Context &context, Limbs q, Limbs num, Limbs denom) {
  PTXInliner inliner;
  uint32_t   warp_thread=threadIdx.x & 0x1F;
  uint32_t   d, approx, est, hi, lo, carryOut=0, ignore;
  uint32_t   registers[2];
  Limbs      phi(registers), plo(registers+1);
  uint32_t   zero=0, ones=0xFFFFFFFF;

  if(warp_int_compare<1>(num, denom)>=0) {
    warp_int_sub<1, false>(context, num, denom);
    warp_int_fast_propagate_sub<1>(context, num);
    carryOut=1;
  }

  d=shfl(denom.limbs[0], 31);
  approx=APPROX32(d);

  #pragma unroll
  for(int32_t index=31;index>=0;index--) {
    lo=shfl(num.limbs[0], 30);
    hi=shfl(num.limbs[0], 31);
    est=DIV32(hi, lo, d, approx);

    while(true) {
      inliner.MULLO(plo.limbs[0], est, denom.limbs[0]);
      phi.limbs[0]=shfl_down(plo.limbs[0], 1);
      phi.limbs[0]=(warp_thread==31) ? 0 : phi.limbs[0];

      plo.limbs[0]=(warp_thread==0) ? plo.limbs[0] : 0;
      inliner.ADD_CC(ignore, plo.limbs[0], ones);
      inliner.MADHIC_CC(phi.limbs[0], est, denom.limbs[0], phi.limbs[0]);
      inliner.ADDC(context.carry, zero, zero);

      warp_int_fast_propagate_add<1>(context, phi);
      if(warp_int_compare<1>(num, phi)>=0)
        break;
      est--;
    }

    inliner.SUB_CC(num.limbs[0], num.limbs[0], phi.limbs[0]);
    inliner.SUBC(context.carry, zero, zero);
    warp_int_fast_propagate_sub<1>(context, num);

    num.limbs[0]=shfl_up(num.limbs[0], 1);
    num.limbs[0]=(warp_thread==0) ? 0-plo.limbs[0] : num.limbs[0];

    q.limbs[0]=(warp_thread==index) ? est : q.limbs[0];
  }
  context.carry=carryOut;
}

template<uint32_t size>
__device__ __forceinline__ void fpa_warp_div_n(Context &context, Limbs q, Limbs num, Limbs denom) {
  PTXInliner inliner;
  uint32_t   warp_thread=threadIdx.x & 0x1F;
  uint32_t   carryOut=0, temp;
  uint32_t   registers[2*size+3];
  Limbs      d(registers), approx(registers+1), est(registers+2), p(registers+3), plo(registers+3), phi(registers+3+size);
  int32_t    count;
  uint32_t   zero=0, one=1;

  if(warp_int_compare<size>(num, denom)>=0) {
    warp_int_sub<size, false>(context, num, denom);
    warp_int_fast_propagate_sub<size>(context, num);
    carryOut=1;
  }

  SPREAD_N<size, true>(d, denom, 31);
  APPROX_N<size>(approx, d);

  warp_int_clear_all<size>(q);

  #pragma nounroll
  for(int32_t index=31;index>=0;index--) {
    DIV_N<size, false>(est, num, d, approx);
    THREAD_PRODUCT_N<size, true, false>(p, est, denom);
    COMPACT_N<size, false>(q, est, index);

    warp_int_sub<size, false>(context, num, phi);
    warp_int_fast_propagate_sub<size>(context, num);
    count=shfl(num.limbs[0], 31);

    #pragma unroll
    for(int32_t limb=0;limb<size;limb++) {
      num.limbs[limb]=shfl_up(num.limbs[limb], 1);
      num.limbs[limb]=(warp_thread==0) ? 0 : num.limbs[limb];
    }

    warp_int_sub<size, false>(context, num, plo);
    temp=warp_int_fast_propagate_sub<size>(context, num);
    if(temp==1)
      count--;

    while(count<0) {
      // est--    note, this can not go negative
      if(warp_thread==index) {
        PTXChain chain1(size);
        chain1.SUB(q.limbs[0], q.limbs[0], one);
        #pragma unroll
        for(int32_t limb=1;limb<size;limb++)
          chain1.SUB(q.limbs[limb], q.limbs[limb], zero);
        chain1.end();
      }

      warp_int_add<size, false>(context, num, denom);
      temp=warp_int_fast_propagate_add<size>(context, num);
      if(temp==1)
        count++;
    }
  }
  context.carry=carryOut;
}

template<uint32_t size>
__device__ void fpa_warp_div(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t a_exp, b_exp, action, sign, roundWord, stickyWord, lsw, round;
  Context  context;
  uint32_t registers[3*size];
  Limbs    num(registers), denom(registers+size), q(registers+2*size);
  int32_t  r_exp;

  if(warp>=arguments.count)
    return;

  a_exp=arguments.a_exps[warp];
  b_exp=arguments.b_exps[warp];
  action=div_action_table[fpa_classify2(a_exp, b_exp)];

  if(action!=FPA_DIV) {
    arguments.r_exps[warp]=action;
    return;
  }

  r_exp=(a_exp>>1)-(b_exp>>1);  // ranges from 0x7FFFFFFF-10 to 10-0x7FFFFFFF=0x8000000B
  sign=(a_exp ^ b_exp) & 0x01;

  warp_int_load<size>(num, arguments.a_limbs + warp*size*32);
  warp_int_load<size>(denom, arguments.b_limbs + warp*size*32);

  if(size==1)
    fpa_warp_div_1(context, q, num, denom);
  else
    fpa_warp_div_n<size>(context, q, num, denom);

  lsw=shfl(q.limbs[0], 0);
  if(context.carry!=0) {
    stickyWord=warp_int_or_all<size>(num);
    warp_int_shift_right_bits<size>(q, 1, 1);
    roundWord=lsw<<31;
    lsw=lsw>>1;
    round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
    warp_int_fast_round<size>(q, round);
    r_exp++;
  }
  else {
    roundWord=0;
    if(warp_int_fast_double<size>(num))
      roundWord=0x80000000;
    if(warp_int_compare<size>(num, denom)>=0)
      roundWord=0x80000000;
    stickyWord=warp_int_or_all<size>(num);

    round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
    if(round && warp_int_and_all<size>(q)==0xFFFFFFFF) {
      // critical case
      warp_int_clear_all<size>(q);
      if(warp_thread==31)
        q.limbs[size-1]=0x80000000;
      r_exp++;
    }
    else
      warp_int_fast_round<size>(q, round);
  }

  // using signed comparison
  if(r_exp<10-FPA_BIAS)
    r_exp=FPA_POS_TINY-sign;
  else if(r_exp>=0x7FFFFFFF-FPA_BIAS)
    r_exp=FPA_POS_INF-sign;
  else {
    r_exp=r_exp + r_exp + FPA_BIAS + FPA_BIAS + sign;
    warp_int_store<size>(arguments.r_limbs + warp*size*32, q);
  }

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

/*
template<uint32_t size>
__device__ void fpa_warp_debug(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5;
  uint32_t registers[12];
  Limbs    num(registers), denom(registers+4), q(registers+8), d(registers+9), approx(registers+10), temp(registers+11);

  if(warp>=arguments.count)
    return;

  warp_int_load<4>(num, arguments.a_limbs + warp * 4 * 32);
  warp_int_load<4>(denom, arguments.b_limbs + warp * 4 * 32);

  SPREAD_N<4, true>(d, denom, 31);
  APPROX_N<4>(approx, d);

  SPREAD_N<4, true>(temp, num, 30);
  temp.limbs[0]=shfl_down(temp.limbs[0], 4);
  SPREAD_N<4, false>(temp, num, 31);

  DIV_N<4, true>(q, temp, d, approx);

  COMPACT_N<4, true>(denom, q, 31);

  warp_int_store<4>(arguments.r_limbs + warp * 4 *32, denom);
  arguments.r_exps[warp]=2*FPA_BIAS;
}
*/
