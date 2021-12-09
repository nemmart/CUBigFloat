__constant__ uint8_t sqrt_action_table[9] = {
   /* x=+value */   FPA_SQRT,
   /*   -value */   FPA_NAN,
   /*     zero */   FPA_ZERO,
   /*   -1/inf */   FPA_NAN,
   /*   +1/inf */   FPA_POS_TINY,
   /*     -inf */   FPA_NAN,
   /*     +inf */   FPA_POS_INF,
   /*      nan */   FPA_NAN,
   /*     snan */   FPA_SNAN,
};

__device__ __forceinline__ void fpa_warp_sqrt_1(Context &context, Limbs s, Limbs x, uint32_t shiftIn=0) {
  PTXInliner inliner;
  uint32_t   warp_thread=threadIdx.x & 0x1F;
  uint32_t   hi, lo, divisor, approx, q, add;
  uint32_t   registers[1];
  Limbs      p(registers);
  int32_t    top;
  uint32_t   zero=0;

  hi=shfl(x.limbs[0], 31);
  lo=shfl(x.limbs[0], 30);

  divisor=SQRT64(hi, lo);   // returns the remainder
  top=hi;
  if(warp_thread==30)
    x.limbs[0]=lo;
  x.limbs[0]=shfl_up(x.limbs[0], 1);
  x.limbs[0]=(warp_thread==0) ? shiftIn : x.limbs[0];

  approx=APPROX32(divisor);

  s.limbs[0]=(warp_thread==31) ? divisor+divisor : 0;   // there is a 'silent' 1 at the top of s

  #pragma unroll 1
  for(int32_t index=30;index>=0;index--) {
    lo=shfl(x.limbs[0], 31);
    q=SQRTQ32(top, lo, divisor, approx);

    s.limbs[0]=(warp_thread==index) ? q : s.limbs[0];

    inliner.MULHI(p.limbs[0], q, s.limbs[0]);
    warp_int_sub<1, false>(context, x, p);
    warp_int_fast_propagate_sub<1>(context, x);

    top=shfl(x.limbs[0], 31)-q;    // we subtract q because of the silent 1 in s

    x.limbs[0]=shfl_up(x.limbs[0], 1);
    x.limbs[0]=(warp_thread==0) ? 0 : x.limbs[0];

    inliner.MULLO(p.limbs[0], q, s.limbs[0]);
    warp_int_sub<1, false>(context, x, p);
    if(warp_int_fast_propagate_sub<1>(context, x)==1)
      top--;

    while(top<0) {
      top++;
      q--;

      add=(warp_thread==index) ? q : 0;
      inliner.ADD_CC(x.limbs[0], x.limbs[0], add);
      inliner.ADDC(context.carry, zero, zero);

      warp_int_add<1, false>(context, x, s);

      s.limbs[0]=(warp_thread==index) ? q : s.limbs[0];

      // push carry to next thread
      add=shfl_up(context.carry, 1);
      add=(warp_thread==0) ? 0 : add;
      context.carry=(warp_thread==31) ? context.carry : 0;

      // integrate carry
      inliner.ADD_CC(x.limbs[0], x.limbs[0], add);
      inliner.ADDC(context.carry, context.carry, zero);
      if(warp_int_fast_propagate_add<1>(context, x)==1)
        top++;
    }

    s.limbs[0]=(warp_thread==index+1) ? s.limbs[0] + (q>>31) : s.limbs[0];
    s.limbs[0]=(warp_thread==index) ? q+q : s.limbs[0];
  }

  warp_int_shift_right_bits<1>(s, 1, 1);   // restore the silent 1
  context.carry=top;
}

template<uint32_t size>
__device__ __forceinline__ void fpa_warp_sqrt_n(Context &context, Limbs s, Limbs x, uint32_t shiftIn=0) {
  PTXInliner inliner;
  uint32_t   warp_thread=threadIdx.x & 0x1F;
  uint32_t   add;
  uint32_t   registers[2*size+4];
  Limbs      divisor(registers), approx(registers+1), q(registers+2), temp(registers+3), p(registers+4), plo(registers+4), phi(registers+4+size);
  int32_t    top;
  uint32_t   zero=0, one=1;

  // spread hi/lo into temp
  SPREAD_N<size, true>(temp, x, 30);
  temp.limbs[0]=shfl_down(temp.limbs[0], size);
  SPREAD_N<size, false>(temp, x, 31);

  SQRT_N<size>(context, divisor, temp);     // returns the remainder high bit in context.carry and low bits in temp
  top=context.carry;

  COMPACT_N<size, true>(s, divisor, 31);    // push the 2*divisor into s, sans the silent high bit
  PTXChain chain1(size);
  #pragma unroll
  for(int32_t limb=0;limb<size;limb++)
    chain1.ADD(s.limbs[limb], s.limbs[limb], s.limbs[limb]);
  chain1.end();

  warp_int_shift_left_words<size>(x, size);
  x.limbs[size-1]=(warp_thread==0) ? shiftIn : x.limbs[size-1];   // handle shiftIn

  COMPACT_N<size, false>(x, temp, 31);      // push the remainder, sans high bit, back into x

  APPROX_N<size>(approx, divisor);

  context.carry=0;
  #pragma nounroll
  for(int32_t index=30;index>=0;index--) {
    SQRTQ_N<size, false>(q, top, x, divisor, approx);

    COMPACT_N<size, false>(s, q, index);
    THREAD_PRODUCT_N<size, true, false>(p, q, s);

    warp_int_sub<size, false>(context, x, phi);
    warp_int_fast_propagate_sub<size>(context, x);

    top=shfl(x.limbs[0], 31)-shfl(q.limbs[0], 32-size);    // we subtract q because of the silent 1 in s

    warp_int_shift_left_words<size>(x, size);
    warp_int_sub<size, false>(context, x, plo);
    if(warp_int_fast_propagate_sub<size>(context, x)==1)
      top--;

    while(top<0) {
      top++;

      warp_int_add<size, false>(context, x, s);

      if(warp_thread==index) {
        // q--
        PTXChain chain2(size);
        chain2.SUB(s.limbs[0], s.limbs[0], one);
        #pragma unroll
        for(int32_t limb=1;limb<size;limb++)
          chain2.SUB(s.limbs[limb], s.limbs[limb], zero);
        chain2.end();

        // add q
        PTXChain chain3(size+1);
        #pragma unroll
        for(int32_t limb=0;limb<size;limb++)
          chain3.ADD(x.limbs[limb], x.limbs[limb], s.limbs[limb]);
        chain3.ADD(context.carry, context.carry, zero);
        chain3.end();
      }

      // push carry to next thread
      add=shfl_up(context.carry, 1);
      add=(warp_thread==0) ? 0 : add;
      context.carry=(warp_thread==31) ? context.carry : 0;

      // integrate carry
      PTXChain chain4(size+1);
      chain4.ADD(x.limbs[0], x.limbs[0], add);
      #pragma unroll
      for(int32_t limb=1;limb<size;limb++)
         chain4.ADD(x.limbs[limb], x.limbs[limb], zero);
      chain4.ADD(context.carry, context.carry, zero);

      // resolve carries
      if(warp_int_fast_propagate_add<size>(context, x)==1)
        top++;
    }

    if(warp_thread==index) {
      PTXChain chain5(size+1);
      #pragma unroll
      for(int32_t limb=0;limb<size;limb++)
        chain5.ADD(s.limbs[limb], s.limbs[limb], s.limbs[limb]);
      chain5.ADD(add, zero, zero);
    }
    add=shfl(add, index);
    s.limbs[0]=(warp_thread==index+1) ? s.limbs[0] + add : s.limbs[0];
  }

  warp_int_shift_right_bits<size>(s, 1, 1);   // restore the silent 1
  context.carry=top;
}

template<uint32_t size>
__device__ void fpa_warp_sqrt(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t r_exp, action, roundWord, stickyWord, lsw, round;
  Context  context;
  uint32_t registers[2*size];
  Limbs    x(registers), s(registers+size);
  int32_t  shifted=0;

  if(warp>=arguments.count)
    return;

  r_exp=arguments.a_exps[warp];
  action=sqrt_action_table[fpa_classify1(r_exp)];
  if(action!=FPA_SQRT) {
    arguments.r_exps[warp]=action;
    return;
  }

  warp_int_load<size>(x, arguments.a_limbs + warp*size*32);

  r_exp=(r_exp>>1) - FPA_BIAS;

  if((r_exp & 0x01)!=0) {
    lsw=shfl(x.limbs[0], 0);
    shifted=lsw & 0x01;
    warp_int_shift_right_bits<size>(x, 1);
    r_exp++;
  }
  if(size==1)
    fpa_warp_sqrt_1(context, s, x, shifted<<31);
  else
    fpa_warp_sqrt_n<size>(context, s, x, shifted<<31);

  if(context.carry>0)
    roundWord=1;
  else
    roundWord=(warp_int_compare<size>(x, s)>=1-shifted);
  stickyWord=context.carry | warp_int_or_all<size>(x);

  lsw=shfl(s.limbs[0], 0);
  round=fpa_round(arguments.mode, 0, lsw, roundWord<<31, stickyWord);

  if(warp_int_and_all<size>(s)==0xFFFFFFFF && round) {
    warp_int_clear_all<size>(s);
    if(warp_thread==31)
      s.limbs[size-1]=0x80000000;
    r_exp++;
  }
  else
    warp_int_fast_round<size>(s, round);

  warp_int_store<size>(arguments.r_limbs + warp*size*32, s);

  // note, this is just r_exp, not r_exp + r_exp
  r_exp=r_exp + FPA_BIAS + FPA_BIAS;

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

