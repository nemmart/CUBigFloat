__constant__ uint8_t mul_action_table[81] = {
   /*               b=+value       -value         zero          -1/inf,        +1/inf         -inf,           +inf,          nan,        snan  */
   /*   +value */   FPA_MUL,       FPA_MUL,       FPA_ZERO,     FPA_NEG_TINY,  FPA_POS_TINY,  FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*   -value */   FPA_MUL,       FPA_MUL,       FPA_ZERO,     FPA_POS_TINY,  FPA_NEG_TINY,  FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*   a=zero */   FPA_ZERO,      FPA_ZERO,      FPA_ZERO,     FPA_ZERO,      FPA_ZERO,      FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*   -1/inf */   FPA_NEG_TINY,  FPA_POS_TINY,  FPA_ZERO,     FPA_POS_TINY,  FPA_NEG_TINY,  FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*   +1/inf */   FPA_POS_TINY,  FPA_NEG_TINY,  FPA_ZERO,     FPA_NEG_TINY,  FPA_POS_TINY,  FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     -inf */   FPA_NEG_INF,   FPA_POS_INF,   FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*     +inf */   FPA_POS_INF,   FPA_NEG_INF,   FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*      nan */   FPA_NAN,       FPA_NAN,       FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     snan */   FPA_SNAN,      FPA_SNAN,      FPA_SNAN,     FPA_SNAN,      FPA_SNAN,      FPA_SNAN,       FPA_SNAN,      FPA_SNAN,   FPA_SNAN,
};

template<uint32_t size>
__device__ __forceinline__ void fpa_warp_mul_block(Context &context, Limbs product, Limbs a, Limbs b, uint32_t thread) {
  uint32_t a_value;
  uint32_t zero=0;

  #pragma unroll
  for(int32_t i=0;i<size;i++) {
    a_value=shfl(a.limbs[i], thread);
    PTXChain chain1(size+2);
    #pragma unroll
    for(int32_t j=0;j<size;j++)
      chain1.MADLO(product.limbs[i+j], a_value, b.limbs[j], product.limbs[i+j]);
    chain1.ADD(product.limbs[i+size], context.carry, zero);
    chain1.ADD(context.carry, zero, zero);
    chain1.end();

    PTXChain chain2(size+1);
    #pragma unroll
    for(int32_t j=0;j<size-1;j++)
      chain2.MADHI(product.limbs[i+j+1], a_value, b.limbs[j], product.limbs[i+j+1]);
    chain2.MADHI(product.limbs[i+size], a_value, b.limbs[size-1], product.limbs[i+size]);
    chain2.ADD(context.carry, context.carry, zero);
    chain2.end();
  }
  context.carry=0;
}

template<uint32_t size>
__device__ void fpa_warp_mul(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t a_exp, b_exp, action, r_exp, sign, roundWord, stickyWord=0, lsw, msw, round;
  Context  context;
  uint32_t registers[4*size];
  Limbs    product(registers), a(registers+2*size), b(registers+3*size);
  uint32_t zero=0;

  if(warp>=arguments.count)
    return;

  a_exp=arguments.a_exps[warp];
  b_exp=arguments.b_exps[warp];
  action=mul_action_table[fpa_classify2(a_exp, b_exp)];

  r_exp=(a_exp>>1) + (b_exp>>1);
  sign=(a_exp^b_exp) & 0x01;
  if(action==FPA_MUL) {
    warp_int_clear_all<size>(product);
    warp_int_load<size>(a, arguments.a_limbs + warp*size*32);
    warp_int_load<size>(b, arguments.b_limbs + warp*size*32);

    #pragma nounroll
    for(int32_t thread=0;thread<32;thread++) {
      fpa_warp_mul_block<size>(context, product, a, b, thread);

      // process least significant <size> words
      #pragma unroll
      for(int32_t limb=0;limb<size-1;limb++)
        stickyWord=stickyWord | product.limbs[limb];
      if(thread<31)
        stickyWord=stickyWord | product.limbs[size-1];
      else
        roundWord=product.limbs[size-1];

      // pull from thread+1
      #pragma unroll
      for(int32_t limb=0;limb<size;limb++) {
        uint32_t x=shfl_down(product.limbs[limb], 1);
        product.limbs[limb]=(warp_thread==31) ? 0 : x;
      }

      // p.lo = p.hi + p.lo(thread+1)
      PTXChain chain1(size+1);
      #pragma unroll
      for(int32_t limb=0;limb<size;limb++)
        chain1.ADD(product.limbs[limb], product.limbs[limb], product.limbs[limb+size]);
      chain1.ADD(context.carry, zero, zero);
      chain1.end();
    }
    warp_int_fast_propagate_add<size>(context, product);    // should be necessary only at the end -- compiler bug?

    stickyWord=shfl(stickyWord, 0);
    roundWord=shfl(roundWord, 0);
    lsw=shfl(product.limbs[0], 0);
    msw=shfl(product.limbs[size-1], 31);

    if(msw==0x7FFFFFFF && warp_int_and_words<size>(product, 32*size-1)==0xFFFFFFFF) {
      round=fpa_round(arguments.mode, sign, 0x01, roundWord, stickyWord);
      if(round) {
        warp_int_clear_all<size>(product);
        if(warp_thread==31)
          product.limbs[size-1]=0x80000000;
      }
      else {
        warp_int_shift_left_bits<size>(product, 1, roundWord>>31);
        roundWord=roundWord<<1;
        round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
        warp_int_fast_round<size>(product, round);
        r_exp--;
      }
    }
    else {
      if(msw<0x80000000) {
        warp_int_shift_left_bits<size>(product, 1, roundWord>>31);
        roundWord=roundWord<<1;
        r_exp--;
      }
      round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
      warp_int_fast_round<size>(product, round);
    }

    // using unsigned comparisons
    if(r_exp<FPA_BIAS+5)
      r_exp=FPA_POS_TINY - sign;
    else if(r_exp>0x7FFFFFFFu + FPA_BIAS)
      r_exp=FPA_POS_INF - sign;
    else {
      r_exp=r_exp + r_exp - 2*FPA_BIAS + sign;
      warp_int_store<size>(arguments.r_limbs + warp*size*32, product);
    }
  }
  else
    r_exp=action;

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}
