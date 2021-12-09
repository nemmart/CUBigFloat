__constant__ int8_t sgn_action_table[9] = {
   /* a=+value */   +1,
   /*   -value */   -1,
   /*     zero */    0,
   /*   -1/inf */   -1,
   /*   +1/inf */   +1,
   /*     -inf */   -1,
   /*     +inf */   +1,
   /*      nan */   FPA_CMP_ERR,
   /*     snan */   FPA_CMP_ERR,
};

__constant__ int8_t cmp_action_table[81] = {
   /*               b=+value       -value         zero           -1/inf,        +1/inf         -inf,          +inf,          nan,           snan  */
   /*   +value */   FPA_CMP,       +1,            +1,            +1,            +1,            +1,            -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*   -value */   -1,            FPA_CMP,       -1,            -1,            -1,            +1,            -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*   a=zero */   -1,            +1,             0,            +1,            -1,            +1,            -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*   -1/inf */   -1,            +1,            -1,            FPA_CMP_ERR,   -1,            +1,            -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*   +1/inf */   -1,            +1,            +1,            +1,            FPA_CMP_ERR,   +1,            -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*     -inf */   -1,            -1,            -1,            -1,            -1,            FPA_CMP_ERR,   -1,            FPA_CMP_ERR,   FPA_CMP_ERR,
   /*     +inf */   +1,            +1,            +1,            +1,            +1,            +1,            FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,
   /*      nan */   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,
   /*     snan */   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,   FPA_CMP_ERR,
};

__device__ void fpa_threaded_sgn(fpa_arguments arguments) {
  uint32_t thread=blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a_exp, r;
  int32_t  action;

  if(thread>=arguments.count)
    return;

  a_exp=arguments.a_exps[thread];
  action=sgn_action_table[fpa_classify1(a_exp)];

  if(action!=FPA_CMP_ERR)
    r=action;
  else
    r=0;
  arguments.r_exps[thread]=r;
}

__device__ void fpa_threaded_cmp(fpa_arguments arguments) {
  uint32_t thread=blockIdx.x * blockDim.x + threadIdx.x, warp_thread=thread & 0x1F;
  uint32_t size=arguments.precision>>10, instance=thread & ~0x1F;
  uint32_t a_exp, b_exp, help;
  int32_t  action=0, r, comparison, sgn;
  uint32_t registers[18];
  Limbs    a(registers), b(registers+8), load_a(registers+16), load_b(registers+17);

  if(thread<arguments.count) {
    a_exp=arguments.a_exps[thread];
    b_exp=arguments.b_exps[thread];
    action=cmp_action_table[fpa_classify2(a_exp, b_exp)];
  }

  if(action==FPA_CMP) {
    sgn=(a_exp & 0x01) ? -1 : 1;
    if(a_exp>b_exp)
      action=sgn;
    if(a_exp<b_exp)
      action=-sgn;
  }

  if(action==FPA_CMP_ERR)
    r=0;
  else if(action!=FPA_CMP)
    r=action;

  // threads that need to compare limb data, use the whole warp
  help=warp_ballot(action==FPA_CMP);
  for(int32_t bit=0;bit<32;bit++) {
    if(((help>>bit) & 0x01)!=0) {
      #pragma unroll
      for(int32_t limb=0;limb<8;limb++) {
        if(limb<size) {
          warp_int_load<1>(load_a, arguments.a_limbs + instance*size*32 + limb*32);
          warp_int_load<1>(load_b, arguments.b_limbs + instance*size*32 + limb*32);
        }
        a.limbs[limb]=(limb<size) ? load_a.limbs[0] : 0;
        b.limbs[limb]=(limb<size) ? load_b.limbs[0] : 0;
      }
      comparison=warp_int_compare<8>(a, b);
      if(warp_thread==bit)
        r=sgn*comparison;
    }
    instance++;
  }

  if(thread<arguments.count)
    arguments.r_exps[thread]=r;
}
