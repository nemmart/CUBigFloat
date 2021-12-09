using namespace gpu_fpa::warp_math;

__constant__ uint8_t neg_action_table[9] = {
   /* a=+value */   FPA_NEG,
   /*   -value */   FPA_NEG,
   /*     zero */   FPA_ZERO,
   /*   -1/inf */   FPA_POS_TINY,
   /*   +1/inf */   FPA_NEG_TINY,
   /*     -inf */   FPA_POS_INF,
   /*     +inf */   FPA_NEG_INF,
   /*      nan */   FPA_NAN,
   /*     snan */   FPA_SNAN,
};

__constant__ uint8_t add_action_table[81] = {
   /*               b=+value       -value         zero          -1/inf,        +1/inf         -inf,           +inf,          nan,        snan  */
   /*   +value */   FPA_ADD,       FPA_SUB,       FPA_COPY_A,   FPA_COPY_A,    FPA_COPY_A,    FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*   -value */   FPA_SUB,       FPA_ADD,       FPA_COPY_A,   FPA_COPY_A,    FPA_COPY_A,    FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*   a=zero */   FPA_COPY_B,    FPA_COPY_B,    FPA_ZERO,     FPA_NEG_TINY,  FPA_POS_TINY,  FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*   -1/inf */   FPA_COPY_B,    FPA_COPY_B,    FPA_NEG_TINY, FPA_NEG_TINY,  FPA_NAN,       FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*   +1/inf */   FPA_COPY_B,    FPA_COPY_B,    FPA_POS_TINY, FPA_NAN,       FPA_POS_TINY,  FPA_NEG_INF,    FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*     -inf */   FPA_NEG_INF,   FPA_NEG_INF,   FPA_NEG_INF,  FPA_NEG_INF,   FPA_NEG_INF,   FPA_NEG_INF,    FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     +inf */   FPA_POS_INF,   FPA_POS_INF,   FPA_POS_INF,  FPA_POS_INF,   FPA_POS_INF,   FPA_NAN,        FPA_POS_INF,   FPA_NAN,    FPA_SNAN,
   /*      nan */   FPA_NAN,       FPA_NAN,       FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     snan */   FPA_SNAN,      FPA_SNAN,      FPA_SNAN,     FPA_SNAN,      FPA_SNAN,      FPA_SNAN,       FPA_SNAN,      FPA_SNAN,   FPA_SNAN,
};

__constant__ uint8_t sub_action_table[81] = {
   /*               b=+value       -value         zero          -1/inf,        +1/inf         -inf,           +inf,          nan,        snan  */
   /*   +value */   FPA_SUB,       FPA_ADD,       FPA_COPY_A,   FPA_COPY_A,    FPA_COPY_A,    FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*   -value */   FPA_ADD,       FPA_SUB,       FPA_COPY_A,   FPA_COPY_A,    FPA_COPY_A,    FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*   a=zero */   FPA_NEG_B,     FPA_NEG_B,     FPA_ZERO,     FPA_POS_TINY,  FPA_NEG_TINY,  FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*   -1/inf */   FPA_NEG_B,     FPA_NEG_B,     FPA_NEG_TINY, FPA_NAN,       FPA_NEG_INF,   FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*   +1/inf */   FPA_NEG_B,     FPA_NEG_B,     FPA_POS_TINY, FPA_POS_TINY,  FPA_NAN,       FPA_POS_INF,    FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*     -inf */   FPA_NEG_INF,   FPA_NEG_INF,   FPA_NEG_INF,  FPA_NEG_INF,   FPA_NEG_INF,   FPA_NAN,        FPA_NEG_INF,   FPA_NAN,    FPA_SNAN,
   /*     +inf */   FPA_POS_INF,   FPA_POS_INF,   FPA_POS_INF,  FPA_POS_INF,   FPA_POS_INF,   FPA_POS_INF,    FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*      nan */   FPA_NAN,       FPA_NAN,       FPA_NAN,      FPA_NAN,       FPA_NAN,       FPA_NAN,        FPA_NAN,       FPA_NAN,    FPA_SNAN,
   /*     snan */   FPA_SNAN,      FPA_SNAN,      FPA_SNAN,     FPA_SNAN,      FPA_SNAN,      FPA_SNAN,       FPA_SNAN,      FPA_SNAN,   FPA_SNAN,
};

template<uint32_t size>
__device__ void fpa_warp_add_case(fpa_arguments arguments, uint32_t a_exp, uint32_t b_exp) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  Context  context;
  uint32_t registers[size*2], r_exp, sign=a_exp & 0x01, carry, lsw, roundWord, stickyWord, round;
  Limbs    acc(registers), add(registers+size);
  int32_t  shift;

  // signs must be the same
  if(a_exp>=b_exp) {
    warp_int_load<size>(acc, arguments.a_limbs + warp*size*32);
    warp_int_load<size>(add, arguments.b_limbs + warp*size*32);
    shift=a_exp-b_exp>>1;
    r_exp=a_exp;
  }
  else {
    warp_int_load<size>(acc, arguments.b_limbs + warp*size*32);
    warp_int_load<size>(add, arguments.a_limbs + warp*size*32);
    shift=b_exp-a_exp>>1;
    r_exp=b_exp;
  }

  if(shift<32) {
    roundWord=shfl(add.limbs[0], 0)<<32-shift;
    roundWord=(shift==0) ? 0 : roundWord;
    stickyWord=0;
    warp_int_shift_right_bits<size>(add, shift);
  }
  else if(shift<=1024*size) {
    stickyWord=0;
    if((shift & 0x1F)!=0) {
      uint32_t bits=shift & 0x1F;

      stickyWord=shfl(add.limbs[0], 0)<<32-bits;
      warp_int_shift_right_bits<size>(add, bits);
    }

    shift=shift>>5;
    if(stickyWord==0)
      stickyWord=warp_int_or_words<size>(add, shift-1);

    warp_int_shift_right_words<size>(add, shift-1);
    roundWord=shfl(add.limbs[0], 0);

    warp_int_shift_right_1_word<size>(add);
  }
  else {
    roundWord=0;
    stickyWord=1;
    warp_int_clear_all<size>(add);
  }

  warp_int_add<size, false>(context, acc, add);
  carry=warp_int_fast_propagate_add<size>(context, acc);
  lsw=shfl(acc.limbs[0], 0);

  if(carry==1) {
    stickyWord=stickyWord | roundWord;
    roundWord=lsw<<31;

    warp_int_shift_right_bits<size>(acc, 1, 1);
    round=fpa_round(arguments.mode, sign, lsw>>1, roundWord, stickyWord);
    warp_int_fast_round<size>(acc, round);
    r_exp=r_exp+2;
  }
  else {
    round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
    if(carry==0xFFFFFFFF && round) {
      warp_int_clear_all<size>(acc);
      if(warp_thread==31)
        acc.limbs[size-1]=0x80000000;
      r_exp=r_exp+2;
    }
    else
      warp_int_fast_round<size>(acc, round);
  }

  if(r_exp<10)
    r_exp=FPA_POS_INF - sign;
  else
    warp_int_store<size>(arguments.r_limbs + warp*size*32, acc);

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

template<uint32_t size>
__device__ void fpa_warp_sub_diff_case(fpa_arguments arguments, uint32_t a_exp, uint32_t b_exp) {
  PTXInliner inliner;
  uint32_t   warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  Context    context;
  uint32_t   registers[size*2], r_exp, sign, msw, lsw, roundWord, stickyWord, round, bit_shift;
  Limbs      acc(registers), sub(registers+size);
  int32_t    shift=(a_exp>>1)-(b_exp>>1);
  uint32_t   zero=0;

  if(shift>=0) {
    warp_int_load<size>(acc, arguments.a_limbs + warp*size*32);
    warp_int_load<size>(sub, arguments.b_limbs + warp*size*32);
    r_exp=a_exp;
  }
  else {
    warp_int_load<size>(acc, arguments.b_limbs + warp*size*32);
    warp_int_load<size>(sub, arguments.a_limbs + warp*size*32);
    shift=-shift;
    r_exp=b_exp;
  }

  sign=r_exp & 0x01;
  if(shift<32) {
    roundWord=shfl(sub.limbs[0], 0)<<32-shift;
    stickyWord=0;
    warp_int_shift_right_bits<size>(sub, shift);
  }
  else if(shift<=1024*size) {
    stickyWord=0;
    if((shift & 0x1F)!=0) {
      uint32_t bits=shift & 0x1F;

      stickyWord=shfl(sub.limbs[0], 0)<<32-bits;
      warp_int_shift_right_bits<size>(sub, bits);
    }

    shift=shift>>5;
    if(stickyWord==0)
      stickyWord=warp_int_or_words<size>(sub, shift-1);

    warp_int_shift_right_words<size>(sub, shift-1);
    roundWord=shfl(sub.limbs[0], 0);

    warp_int_shift_right_1_word<size>(sub);
  }
  else {
    roundWord=0;
    stickyWord=1;
    warp_int_clear_all<size>(sub);
  }

  inliner.SUB_CC(stickyWord, zero, stickyWord);
  inliner.SUBC_CC(roundWord, zero, roundWord);
  warp_int_sub<size, true>(context, acc, sub);
  warp_int_fast_propagate_sub<size>(context, acc);
  msw=shfl(acc.limbs[size-1], 31);
  lsw=shfl(acc.limbs[0], 0);

  bit_shift=0;
  if(msw==0x7FFFFFFF && warp_int_and_words<size>(acc, 32*size-1)==0xFFFFFFFF) {
    // critical case
    if(fpa_round(arguments.mode, sign, 0x01, roundWord, stickyWord)) {
      warp_int_clear_all<size>(acc);
      if(warp_thread==31)
        acc.limbs[size-1]=0x80000000;
    }
    else {
      warp_int_shift_left_bits<size>(acc, 1, roundWord>>31);
      round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
      warp_int_fast_round<size>(acc, round);
      bit_shift=2;
    }
  }
  else if(msw>=0x40000000) {
    if(msw<0x80000000) {
      lsw=roundWord>>31;
      warp_int_shift_left_bits<size>(acc, 1, lsw);
      roundWord=roundWord<<1;
      bit_shift=2;
    }
    round=fpa_round(arguments.mode, sign, lsw, roundWord, stickyWord);
    warp_int_fast_round<size>(acc, round);
  }
  else {
    // shift in the roundWord
    warp_int_shift_left_bits<size>(acc, 1, roundWord>>31);

    shift=warp_int_clz_words<size>(acc);
    warp_int_shift_left_words<size>(acc, shift);

    bit_shift=__clz(shfl(acc.limbs[size-1], 31));
    if(bit_shift!=0)
      warp_int_shift_left_bits<size>(acc, bit_shift);

    bit_shift=bit_shift + 32*shift + 1;
    bit_shift=bit_shift + bit_shift;
  }

  if(r_exp<bit_shift+10)
    r_exp=FPA_POS_TINY - sign;
  else {
    r_exp=r_exp - bit_shift;
    warp_int_store<size>(arguments.r_limbs + warp*size*32, acc);
  }

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

template<uint32_t size>
__device__ void fpa_warp_sub_same_case(fpa_arguments arguments, uint32_t a_exp, uint32_t b_exp) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  Context  context;
  uint32_t registers[size*2], r_exp, borrow;
  Limbs    acc(registers), sub(registers+size);
  int32_t  word_shift, bit_shift;

  warp_int_load<size>(acc, arguments.a_limbs + warp*size*32);
  warp_int_load<size>(sub, arguments.b_limbs + warp*size*32);

  warp_int_sub<size, false>(context, acc, sub);
  borrow=warp_int_fast_propagate_sub<size>(context, acc);
  if(borrow==0xFFFFFFFF) {
    // zero case
    r_exp=FPA_ZERO;
  }
  else {
    if(borrow==1) {
      warp_int_fast_negate<size>(acc);
      r_exp=b_exp;
    }
    else
      r_exp=a_exp;

    word_shift=warp_int_clz_words<size>(acc);
    warp_int_shift_left_words<size>(acc, word_shift);

    bit_shift=__clz(shfl(acc.limbs[size-1], 31));
    if(bit_shift!=0)
      warp_int_shift_left_bits<size>(acc, bit_shift);

    bit_shift=bit_shift + 32*word_shift;

    bit_shift=bit_shift + bit_shift;
    if(r_exp<bit_shift+10)
      r_exp=FPA_POS_TINY - (r_exp & 0x01);
    else {
      r_exp=r_exp - bit_shift;
      warp_int_store<size>(arguments.r_limbs + warp*size*32, acc);
    }
  }

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

__device__ void fpa_warp_neg(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t a_exp, r_exp, action, size=arguments.precision>>10;
  uint32_t registers[1];
  Limbs    limbs(registers);

  if(warp>=arguments.count)
    return;

  a_exp=arguments.a_exps[warp];
  action=neg_action_table[fpa_classify1(a_exp)];

  if(action==FPA_NEG) {
    r_exp=a_exp ^ 0x01;
    for(int32_t index=0;index<size;index++) {
      warp_int_load<1>(limbs, arguments.a_limbs + warp*size*32 + index*32);
      warp_int_store<1>(arguments.r_limbs + warp*size*32 + index*32, limbs);
    }
  }
  else
    r_exp=action;

  if(warp_thread==0)
    arguments.r_exps[warp]=r_exp;
}

template<uint32_t size>
__device__ void fpa_warp_add(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t a_exp, b_exp, action;

  if(warp>=arguments.count)
    return;

  a_exp=arguments.a_exps[warp];
  b_exp=arguments.b_exps[warp];
  action=add_action_table[fpa_classify2(a_exp, b_exp)];

  switch(action) {
    case FPA_ADD:
      fpa_warp_add_case<size>(arguments, a_exp, b_exp);
      break;
    case FPA_SUB:
      if((a_exp ^ b_exp)<2)
        fpa_warp_sub_same_case<size>(arguments, a_exp, b_exp);
      else
        fpa_warp_sub_diff_case<size>(arguments, a_exp, b_exp);
      break;
    case FPA_COPY_A:
      fpa_warp_copy_a<size>(arguments, a_exp);
      break;
    case FPA_COPY_B:
      fpa_warp_copy_b<size>(arguments, b_exp);
      break;
    default:
      if(warp_thread==0)
        arguments.r_exps[warp]=action;
      break;
  }
}

template<uint32_t size>
__device__ void fpa_warp_sub(fpa_arguments arguments) {
  uint32_t warp=(blockIdx.x * blockDim.x + threadIdx.x)>>5, warp_thread=threadIdx.x & 0x1F;
  uint32_t a_exp, b_exp, action;

  if(warp>=arguments.count)
    return;

  a_exp=arguments.a_exps[warp];
  b_exp=arguments.b_exps[warp];
  action=sub_action_table[fpa_classify2(a_exp, b_exp)];

  switch(action) {
    case FPA_ADD:
      b_exp=b_exp ^ 0x01;
      fpa_warp_add_case<size>(arguments, a_exp, b_exp);
      break;
    case FPA_SUB:
      b_exp=b_exp ^ 0x01;
      if((a_exp ^ b_exp)<2)
        fpa_warp_sub_same_case<size>(arguments, a_exp, b_exp);
      else
        fpa_warp_sub_diff_case<size>(arguments, a_exp, b_exp);
      break;
    case FPA_COPY_A:
      fpa_warp_copy_a<size>(arguments, a_exp);
      break;
    case FPA_COPY_B:
      fpa_warp_copy_b<size>(arguments, b_exp);
      break;
    default:
      if(warp_thread==0)
        arguments.r_exps[warp]=action;
      break;
  }
}

