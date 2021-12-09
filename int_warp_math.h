#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include "ptx/ptx.h"

using namespace xmp;

namespace gpu_fpa { namespace warp_math {
  template<uint32_t size>
  __device__ __forceinline__ uint32_t divide(uint32_t numerator) {
    PTXInliner inliner;

    if(size==1)
      return numerator;
    else {
      #if __CUDA_ARCH__>=500 && __CUDA_ARCH__<600
        uint32_t est=0xFFFF/size + ((0xFFFF % size==size-1) ? 2 : 1);
        uint32_t q, zero=0;

        inliner.XMADLL(q, est, numerator, zero);
        return q>>16;
      #else
        uint32_t est=0xFFFFFFFF/size + ((0xFFFFFFFF % size==size-1) ? 2 : 1);

        return __umulhi(est, numerator);
      #endif
    }
  }

  __device__ void swap(uint32_t &a, uint32_t &b) {
    uint32_t swap;

    swap=a;
    a=b;
    b=swap;
  }

  class Context {
    public:
    uint32_t carry;

    __device__ __forceinline__ Context() {
      carry=0;
    }
  };

  class Limbs {
    public:
    uint32_t *limbs;

    __device__ __forceinline__ Limbs(uint32_t *registers) {
      limbs=registers;
    }
  };

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_load(Limbs x, uint32_t *data) {
    int32_t warp_thread=threadIdx.x & 0x1f;

    data+=warp_thread;
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      x.limbs[index]=data[32*index];
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_store(uint32_t *data, Limbs x) {
    int32_t warp_thread=threadIdx.x & 0x1f;

    data+=warp_thread;
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      data[32*index]=x.limbs[index];
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_load_scrambled(Limbs x, uint32_t *data, uint32_t chunk) {
    int     warp_thread=threadIdx.x & 0x1F;
    int32_t word=chunk*32 + warp_thread;

    x.limbs[0]=data[word/size + word%size*32];
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_clear_all(Limbs x) {
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      x.limbs[index]=0;
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_get_word(Limbs x, int32_t word) {
    uint32_t value, select=word%size;

    value=x.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      value=(index!=select) ? value : x.limbs[index];

    return shfl(value, word/size);
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_right_1_word(Limbs x) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t value;

    value=shfl_down(x.limbs[0], 1);
    #pragma unroll
    for(int32_t index=0;index<size-1;index++)
      x.limbs[index]=x.limbs[index+1];
    x.limbs[size-1]=(warp_thread==31) ? 0 : value;
  }

/*
  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_right_words(Limbs x, int32_t words) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t q=divide<size>(words), r=words-q*size;
    uint32_t temp[size];

    if(r==0 && q!=0) {
      #pragma unroll
      for(int word=0;word<size;word++) {
        x.limbs[word]=shfl_down(x.limbs[word], q);
        x.limbs[word]=(31-warp_thread<q) ? 0 : x.limbs[word];
      }
    }

    #pragma unroll
    for(int offset=1;offset<size;offset++) {
      if(r==offset) {
        #pragma unroll
        for(int word=0;word<size;word++)
          temp[word]=x.limbs[(word+offset)%size];
        #pragma unroll
        for(int word=0;word<size;word++) {
          if(word>=offset) {
            x.limbs[word]=shfl_down(temp[word], q+1);
            x.limbs[word]=(31-warp_thread<=q) ? 0 : x.limbs[word];
          }
          else {
            x.limbs[word]=shfl_down(temp[word], q);
            x.limbs[word]=(31-warp_thread<q) ? 0 : x.limbs[word];
          }
        }
      }
    }
  }
*/

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_right_words(Limbs x, int32_t amount) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t skip=amount/size, remaining=amount-skip*size;
    uint32_t value;

    // just not very smart

    if(size==1 || skip>0) {
      #pragma unroll
      for(int32_t index=0;index<size;index++) {
        value=shfl_down(x.limbs[index], skip);
        x.limbs[index]=(warp_thread+skip>=32) ? 0 : value;
      }
    }

    if(size!=1 && remaining>0) {
      #pragma unroll
      for(int32_t index=0;index<size-1;index++) {
        if(index<remaining) {
          value=shfl_down(x.limbs[0], 1);
          #pragma unroll
          for(int32_t limb=0;limb<size-1;limb++)
            x.limbs[limb]=x.limbs[limb+1];
          x.limbs[size-1]=(warp_thread==31) ? 0 : value;
        }
      }
    }
  }

/*
  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_left_words(Limbs x, int32_t words) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t q=divide<size>(words), r=words-q*size;
    uint32_t temp[size];

    if(r==0 && q!=0) {
      #pragma unroll
      for(int word=0;word<size;word++) {
        x.limbs[word]=shfl_up(x.limbs[word], q);
        x.limbs[word]=(warp_thread<q) ? 0 : x.limbs[word];
      }
    }

    #pragma unroll
    for(int offset=1;offset<size;offset++) {
      if(r==offset) {
        #pragma unroll
        for(int word=0;word<size;word++)
          temp[(word+offset)%size]=x.limbs[word];
        #pragma unroll
        for(int word=0;word<size;word++) {
          if(word<offset) {
            x.limbs[word]=shfl_up(temp[word], q+1);
            x.limbs[word]=(warp_thread<=q) ? 0 : x.limbs[word];
          }
          else {
            x.limbs[word]=shfl_up(temp[word], q);
            x.limbs[word]=(warp_thread<q) ? 0 : x.limbs[word];
          }
        }
      }
    }
  }
*/

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_left_words(Limbs x, int32_t amount) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t skip=amount/size, remaining=amount-skip*size;
    uint32_t value;

    // just not very smart

    if(size==1 || skip>0) {
      #pragma unroll
      for(int32_t index=0;index<size;index++) {
        value=shfl_up(x.limbs[index], skip);
        x.limbs[index]=(warp_thread<skip) ? 0 : value;
      }
    }

    if(size!=1 && remaining>0) {
      #pragma unroll
      for(int32_t index=0;index<size-1;index++) {
        if(index<remaining) {
          value=shfl_up(x.limbs[size-1], 1);
          #pragma unroll
          for(int32_t limb=size-1;limb>0;limb--)
            x.limbs[limb]=x.limbs[limb-1];
          x.limbs[0]=(warp_thread==0) ? 0 : value;
        }
      }
    }
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_right_bits(Limbs x, int32_t bits, uint32_t in=0) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t value;

    value=shfl_down(x.limbs[0], 1);
    value=(warp_thread==31) ? in : value;

    #pragma unroll
    for(int32_t index=0;index<size-1;index++)
      x.limbs[index]=(x.limbs[index]>>bits) | (x.limbs[index+1]<<32-bits);
    x.limbs[size-1]=(x.limbs[size-1]>>bits) | (value<<32-bits);
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_shift_left_bits(Limbs x, int32_t bits, uint32_t in=0) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t value;

    value=shfl_up(x.limbs[size-1], 1);
    value=(warp_thread==0) ? in : value>>32-bits;

    #pragma unroll
    for(int32_t index=size-1;index>0;index--)
      x.limbs[index]=(x.limbs[index]<<bits) | (x.limbs[index-1]>>32-bits);
    x.limbs[0]=(x.limbs[0]<<bits) | value;
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_clz_words(Limbs x) {
    int32_t top=size-1, leading, clz, ballot;

    #pragma unroll
    for(int32_t index=size-1;index>=0;index--)
      if(top==index && x.limbs[index]==0)
        top--;
    leading=size-1-top;

    ballot=warp_ballot(top>=0);
    if(ballot==0)
      return size*32;
    clz=__clz(ballot);
    return clz*size + shfl(leading, 31-clz);
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_add_1(Context &context, Limbs acc, uint32_t add) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t zero=0;

    if(warp_thread!=0)
      add=0;

    PTXChain chain1(size+1);
    chain1.ADD(acc.limbs[0], acc.limbs[0], add);
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      chain1.ADD(acc.limbs[index], acc.limbs[index], zero);
    chain1.ADD(context.carry, context.carry, zero);
    chain1.end();
  }

  template<uint32_t size, bool carryIn>
  __device__ __forceinline__ void warp_int_add(Context &context, Limbs acc, Limbs add) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    uint32_t   zero=0;

    if(carryIn) {
      if(warp_thread==0)
        inliner.ADDC_CC(acc.limbs[0], acc.limbs[0], add.limbs[0]);
      else
        inliner.ADD_CC(acc.limbs[0], acc.limbs[0], add.limbs[0]);
    }
    else
      inliner.ADD_CC(acc.limbs[0], acc.limbs[0], add.limbs[0]);

    PTXChain chain1(size+1, true, false);
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      chain1.ADD(acc.limbs[index], acc.limbs[index], add.limbs[index]);
    chain1.ADD(context.carry, context.carry, zero);
    chain1.end();
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_sub_1(Context &context, Limbs acc, uint32_t sub) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t zero=0;

    if(warp_thread!=0)
      sub=0;

    PTXChain chain1(size+1);
    chain1.SUB(acc.limbs[0], acc.limbs[0], sub);
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      chain1.SUB(acc.limbs[index], acc.limbs[index], zero);
    chain1.SUB(context.carry, context.carry, zero);
    chain1.end();
  }

  template<uint32_t size, bool carryIn>
  __device__ __forceinline__ void warp_int_sub(Context &context, Limbs acc, Limbs sub) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    uint32_t   zero=0;

    if(carryIn) {
      if(warp_thread==0)
        inliner.SUBC_CC(acc.limbs[0], acc.limbs[0], sub.limbs[0]);
      else
        inliner.SUB_CC(acc.limbs[0], acc.limbs[0], sub.limbs[0]);
    }
    else
      inliner.SUB_CC(acc.limbs[0], acc.limbs[0], sub.limbs[0]);

    PTXChain chain1(size+1, true, false);
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      chain1.SUB(acc.limbs[index], acc.limbs[index], sub.limbs[index]);
    chain1.SUB(context.carry, context.carry, zero);
    chain1.end();
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_resolve_carry(Context &context, Limbs acc) {
    PTXInliner inliner;
    uint32_t   x, carry;
    uint32_t   g, p;
    uint32_t   zero=0;

    x=acc.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x & acc.limbs[index];

    g=warp_ballot(context.carry==1);
    p=warp_ballot(x==0xFFFFFFFF);

    inliner.ADD(x, g, p);
    inliner.ADD_CC(x, x, g);
    inliner.ADDC(carry, carry, zero);
    return carry;
  }


  /****************************************************************
   * returns 1 if carries out
   * returns -1 if all bits are one
   * returns 0 otherwise
   ****************************************************************/
  template<uint32_t size>
  __device__ __forceinline__ int32_t warp_int_fast_propagate_add(Context &context, Limbs acc) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F, lane=1<<warp_thread;
    uint32_t   g, p, x, ignore, carry;
    uint32_t   zero=0, ones=0xFFFFFFFF;

    x=acc.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x & acc.limbs[index];

    g=warp_ballot(context.carry==1);
    p=warp_ballot(x==0xFFFFFFFF);

    inliner.ADD_CC(x, g, g);
    inliner.ADDC(carry, zero, zero);
    inliner.ADD_CC(g, x, p);
    inliner.ADDC(carry, carry, zero);

    g=(g^p)&lane;
    PTXChain chain1(size+1);
    chain1.ADD(ignore, g, ones);
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      chain1.ADD(acc.limbs[index], acc.limbs[index], zero);
    chain1.end();

    context.carry=0;
    return carry-(p==0xFFFFFFFF);
  }

  /****************************************************************
   * returns 1 if borrows out
   * returns -1 if all bits are zero
   * returns 0 otherwise
   ****************************************************************/
  template<uint32_t size>
  __device__ __forceinline__ int32_t warp_int_fast_propagate_sub(Context &context, Limbs acc) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F, lane=1<<warp_thread;
    uint32_t   g, p, x, ignore, carry;
    uint32_t   zero=0;

    x=acc.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x | acc.limbs[index];

    g=warp_ballot(context.carry==0xFFFFFFFF);
    p=warp_ballot(x==0);

    inliner.ADD_CC(x, g, g);
    inliner.ADDC(carry, zero, zero);
    inliner.ADD_CC(g, x, p);
    inliner.ADDC(carry, carry, zero);

    g=(g^p)&lane;
    PTXChain chain1(size+1);
    chain1.SUB(ignore, zero, g);
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      chain1.SUB(acc.limbs[index], acc.limbs[index], zero);
    chain1.end();

    context.carry=0;
    return carry-(p==0xFFFFFFFF);
  }

  /****************************************************************
   * returns 1 if all bits are zero
   * returns 0 otherwise
   ****************************************************************/
  template<uint32_t size>
  __device__ __forceinline__ int32_t warp_int_fast_negate(Limbs acc) {
    uint32_t warp_thread=threadIdx.x & 0x1F, lane=1<<warp_thread;
    uint32_t p, x;
    uint32_t zero=0, ones=0xFFFFFFFF;

    x=acc.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x | acc.limbs[index];

    p=warp_ballot(x==0);
    p=(p+1^p)&lane;

    PTXChain chain1(size+1);
    chain1.ADD(x, p, ones);
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      chain1.SUB(acc.limbs[index], zero, acc.limbs[index]);
    chain1.end();

    return p==0xFFFFFFFF;
  }

  template<uint32_t size>
  __device__ __forceinline__ void warp_int_fast_round(Limbs acc, uint32_t round) {
    uint32_t warp_thread=threadIdx.x & 0x1F, lane=1<<warp_thread;
    uint32_t p, x;
    uint32_t zero=0, ones=0xFFFFFFFF;

    if(round>0) {
      x=acc.limbs[0];
      #pragma unroll
      for(int32_t index=1;index<size;index++)
        x=x & acc.limbs[index];

      p=warp_ballot(x==0xFFFFFFFF);
      p=(p+round^p)&lane;

      PTXChain chain1(size+1);
      chain1.ADD(x, p, ones);
      #pragma unroll
      for(int32_t index=0;index<size;index++)
        chain1.ADD(acc.limbs[index], acc.limbs[index], zero);
      chain1.end();
    }
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_fast_double(Limbs limbs) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t carry, add;
    uint32_t zero=0;

    PTXChain chain1(size+1);
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      chain1.ADD(limbs.limbs[index], limbs.limbs[index], limbs.limbs[index]);
    chain1.ADD(carry, zero, zero);
    chain1.end();

    add=shfl_up(carry, 1);
    limbs.limbs[0]=(warp_thread==0) ? limbs.limbs[0] : limbs.limbs[0] + add;
    return shfl(carry, 31);
  }

  template<uint32_t size, bool signed_carry>
  __device__ __forceinline__ uint32_t warp_int_resolve(Context &context, Limbs acc, uint32_t round=0) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F, lane=1<<warp_thread;
    uint32_t   g, p, x, ignore;
    uint32_t   zero=0, most=0x7FFFFFFF, ones=0xFFFFFFFF;

    if(!signed_carry) {
      x=shfl_up(context.carry, 1);
      x=(warp_thread==0) ? round : x;
      context.carry = (warp_thread==31) ? context.carry : 0;

      PTXChain chain1(size+1);
      #pragma unroll
      for(int32_t index=0;index<size;index++)
        chain1.ADD(acc.limbs[index], acc.limbs[index], (index==0) ? x : zero);
      chain1.ADD(context.carry, context.carry, zero);
      chain1.end();
    }
    if(signed_carry) {
      context.carry=context.carry + ((warp_thread==31) ? 0xFFFFFFFF : 0x80000000);

      PTXChain chain2(size+1);
      chain2.ADD(acc.limbs[0], acc.limbs[0], most);
      for(int32_t index=1;index<size;index++)
        chain2.ADD(acc.limbs[index], acc.limbs[index], ones);
      chain2.ADD(context.carry, context.carry, zero);
      chain2.end();

      x=shfl_up(context.carry, 1);
      x=(warp_thread==0) ? 0x80000001+round : x;
      context.carry = (warp_thread==31) ? context.carry : 0;

      PTXChain chain3(size+1);
      #pragma unroll
      for(int32_t index=0;index<size;index++)
        chain3.ADD(acc.limbs[index], acc.limbs[index], (index==0) ? x : zero);
      chain3.ADD(context.carry, context.carry, zero);
      chain3.end();
    }

    x=acc.limbs[0];
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x & acc.limbs[index];

    g=warp_ballot(context.carry==1);
    p=warp_ballot(x==0xFFFFFFFF);

    inliner.ADD_CC(x, g, g);
    inliner.ADDC(context.carry, context.carry, zero);
    inliner.ADD_CC(g, x, p);
    inliner.ADDC(context.carry, context.carry, zero);

    g=(g^p)&lane;

    PTXChain chain4(size+1);
    chain4.ADD(ignore, g, ones);
    #pragma unroll
    for(int32_t index=0;index<size;index++)
      chain4.ADD(acc.limbs[index], acc.limbs[index], zero);
    chain4.end();

    context.carry=(warp_thread<31) ? 0 : context.carry;
    return shfl(context.carry, 31);
  }

  template<uint32_t size>
  __device__ __forceinline__ bool warp_int_negative(Context &context, Limbs acc) {
    uint32_t x=acc.limbs[0];

    // assume that context.carry is either 0 of negative

    #pragma unroll
    for(int32_t index=1;index<size;index++)
      x=x | acc.limbs[index];

    return warp_ballot(context.carry>=0x80000000)>warp_ballot(x!=0);
  }

  template<uint32_t size>
  __device__ __forceinline__ int32_t warp_int_compare(Limbs a, Limbs b) {
    uint32_t all, current;
    uint32_t b1, b2;
    uint32_t zero=0;

    if(size==1) {
      b1=warp_ballot(a.limbs[0]>b.limbs[0]);
      b2=warp_ballot(a.limbs[0]<b.limbs[0]);
    }
    else {
      PTXChain chain1(size+1);
      chain1.SUB(all, a.limbs[0], b.limbs[0]);
      #pragma unroll
      for(int32_t index=1;index<size;index++) {
        chain1.SUB(current, a.limbs[index], b.limbs[index]);
        all=all | current;
      }
      chain1.SUB(current, zero, zero);
      b2=warp_ballot(current==0xFFFFFFFF);
      b1=warp_ballot(all!=0) ^ b2;
    }
    return (b1>b2)-(b1<b2);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_or_all(Limbs a) {
    uint32_t value=a.limbs[0];

    #pragma unroll
    for(int32_t index=1;index<size;index++)
      value=value | a.limbs[index];
    return warp_ballot(value!=0);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_and_all(Limbs a) {
    uint32_t value=a.limbs[0];

    #pragma unroll
    for(int32_t index=1;index<size;index++)
      value=value & a.limbs[index];
    return warp_ballot(value==0xFFFFFFFF);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_or_words(Limbs a, int32_t words) {
    int32_t  warp_thread=threadIdx.x & 0x1F, base=warp_thread*size;
    uint32_t value;

    value=(words>base) ? a.limbs[0] : 0;
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      value=(words>base+index) ? value | a.limbs[index] : value;
    return warp_ballot(value!=0);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_and_words(Limbs a, int32_t words) {
    uint32_t warp_thread=threadIdx.x & 0x1F, base=warp_thread*size;
    uint32_t value;

    value=(words>base) ? a.limbs[0] : 0xFFFFFFFF;
    #pragma unroll
    for(int32_t index=1;index<size;index++)
      value=(words>base+index) ? value & a.limbs[index] : value;
    return warp_ballot(value==0xFFFFFFFF);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_or_bits(Limbs a, uint32_t bits) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t value, mask=(1<<(bits & 0x1F))-1;

    if(bits>=warp_thread*32*size+32)
      value=a.limbs[0];
    else if(bits>warp_thread*32*size)
      value=a.limbs[0] & mask;
    #pragma unroll
    for(int32_t index=1;index<size;index++) {
      if(bits>=warp_thread*32*size+index*32+32)
        value=value | a.limbs[index];
      else if(bits>warp_thread*32*size+index*32)
        value=value | (a.limbs[index] & mask);
    }
    return warp_ballot(value!=0);
  }

  template<uint32_t size>
  __device__ __forceinline__ uint32_t warp_int_and_bits(Limbs a, uint32_t bits) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t value=0xFFFFFFFF, mask=-(1<<(bits & 0x1F));

    if(bits>=warp_thread*32*size+32)
      value=a.limbs[0];
    else if(bits>warp_thread*32*size)
      value=a.limbs[0] | mask;
    #pragma unroll
    for(int32_t index=1;index<size;index++) {
      if(bits>=warp_thread*32*size+index*32+32)
        value=value & a.limbs[index];
      else if(bits>warp_thread*32*size+index*32)
        value=value & (a.limbs[index] | mask);
    }
    return warp_ballot(value==0xFFFFFFFF);
  }

  template<uint32_t N, bool zero>
  __device__ __forceinline__ void SPREAD_N(Limbs r, Limbs compact, uint32_t source_thread) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t shuffle;

    if(zero) {
      shuffle=shfl(compact.limbs[0], source_thread);
      r.limbs[0]=(warp_thread==32-N) ? shuffle : 0;
    }
    else {
      shuffle=shfl(compact.limbs[0], source_thread);
      r.limbs[0]=(warp_thread==32-N) ? shuffle : r.limbs[0];
    }

    #pragma unroll
    for(int32_t index=1;index<N;index++) {
      shuffle=shfl(compact.limbs[index], source_thread);
      r.limbs[0]=(warp_thread==32-N+index) ? shuffle : r.limbs[0];
    }
  }

  template<uint32_t N, bool zero>
  __device__ __forceinline__ void COMPACT_N(Limbs r, Limbs spread, uint32_t destination_thread) {
    uint32_t warp_thread=threadIdx.x & 0x1F;
    uint32_t shuffle;

    if(zero) {
      #pragma unroll
      for(int32_t index=0;index<N;index++) {
        shuffle=shfl(spread.limbs[0], 32-N+index);
        r.limbs[index]=(warp_thread==destination_thread) ? shuffle : 0;
      }
    }
    else {
      #pragma unroll
      for(int32_t index=0;index<N;index++) {
        shuffle=shfl(spread.limbs[0], 32-N+index);
        r.limbs[index]=(warp_thread==destination_thread) ? shuffle : r.limbs[index];
      }
    }
  }

  // restricted to N<=16
  template<uint32_t N, bool spread, bool add>
  __device__ __forceinline__ void WARP_PRODUCT_N(Limbs r, Limbs a, Limbs b) {
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    PTXInliner inliner;
    uint32_t   k, r0=0, r1=0;
    Context    context;
    uint32_t   zero=0;

    if(add)
      r0=r.limbs[0];

    context.carry=0;
    for(int32_t index=0;index<N;index++) {
      if(spread)
        k=shfl(a.limbs[0], 32-N+index);
      else
        k=shfl(a.limbs[index], 31);
      inliner.MADLO_CC(r0, k, b.limbs[0], r0);
      inliner.MADHIC(r1, k, b.limbs[0], context.carry);
      r0=shfl(r0, warp_thread+1);
      inliner.ADD_CC(r0, r0, r1);
      inliner.ADDC(context.carry, zero, zero);
    }

    r.limbs[0]=r0;
    warp_int_fast_propagate_add<1>(context, r);
  }

/*
  // restricted to N<16
  template<uint32_t N, bool spread, bool add>
  __device__ __forceinline__ void WARP_PRODUCT_N(Limbs r, Limbs a, Limbs b) {
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    PTXInliner inliner;
    uint32_t   k;
    Context    context;
    uint32_t   zero=0;

    if(spread)
      k=shfl(a.limbs[0], 32-N);
    else
      k=shfl(a.limbs[0], 31);
    if(add) {
      inliner.MADLO_CC(r.limbs[0], k, b.limbs[0], r.limbs[0]);
      r.limbs[0]=shfl(r.limbs[0], warp_thread+1);
      context.carry=shfl(context.carry, warp_thread+1);
      inliner.MADHIC_CC(r.limbs[0], k, b.limbs[0], r.limbs[0]);
      inliner.ADDC(context.carry, context.carry, zero);
    }
    else {
      inliner.MULLO(r.limbs[0], k, b.limbs[0]);
      r.limbs[0]=shfl(r.limbs[0], warp_thread+1);
      inliner.MADHI_CC(r.limbs[0], k, b.limbs[0], r.limbs[0]);
      inliner.ADDC(context.carry, zero, zero);
    }

    #pragma unroll
    for(int32_t index=1;index<N;index++) {
      if(spread)
        k=shfl(a.limbs[0], 32-N+index);
      else
        k=shfl(a.limbs[index], 31);
      inliner.MADLO_CC(r.limbs[0], k, b.limbs[0], r.limbs[0]);
      r.limbs[0]=shfl(r.limbs[0], warp_thread+1);
      context.carry=shfl(context.carry, warp_thread+1);
      inliner.MADHIC_CC(r.limbs[0], k, b.limbs[0], r.limbs[0]);
      inliner.ADDC(context.carry, context.carry, zero);
    }

    if(N>1) {
      context.carry=shfl_up(context.carry, 1);
      inliner.ADD_CC(r.limbs[0], r.limbs[0], context.carry);
      inliner.ADDC(context.carry, zero, zero);
    }
    warp_int_fast_propagate_add<1>(context, r);
  }
*/

  template<uint32_t N, bool spread, bool addLow>
  __device__ __forceinline__ void THREAD_PRODUCT_N(Limbs r, Limbs a, Limbs b, int32_t source_thread=-1) {
    uint32_t value, carry;
    uint32_t zero=0;

    if(!addLow) {
      #pragma unroll
      for(int32_t limb=0;limb<N;limb++)
        r.limbs[limb]=0;
    }

    #pragma unroll
    for(int32_t i=0;i<N;i++) {
      if(spread)
        value=shfl(a.limbs[0], 32-N+i);
      else {
        if(source_thread!=-1)
          value=shfl(a.limbs[i], source_thread);
        else
          value=a.limbs[i];
      }

      PTXChain chain1(N+1);
      #pragma unroll
      for(int32_t j=0;j<N;j++)
        chain1.MADLO(r.limbs[i+j], value, b.limbs[j], r.limbs[i+j]);
      chain1.ADD(carry, zero, zero);
      chain1.end();

      PTXChain chain2(N);
      #pragma unroll
      for(int32_t j=0;j<N-1;j++)
        chain2.MADHI(r.limbs[i+j+1], value, b.limbs[j], r.limbs[i+j+1]);
      chain2.MADHI(r.limbs[i+N], value, b.limbs[N-1], carry);
      chain2.end();
    }
  }

  // restricted to N<32
  template<uint32_t N>
  __device__ __forceinline__ void APPROX_N(Limbs approx, Limbs denom) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    uint32_t   d, approx32, hi, lo, est;
    Context    context;
    uint32_t   registers[2];
    Limbs      num(registers), temp(registers+1);
    uint32_t   zero=0, one=1;

    d=shfl(denom.limbs[0], 31);
    if(d==0x80000000 && warp_ballot(denom.limbs[0]==0)==0x7FFFFFFF) {
      approx.limbs[0]=(warp_thread>=32-N) ? 0xFFFFFFFF : 0;
      return;
    }
    approx32=APPROX32(d);

    warp_int_clear_all<1>(num);
    warp_int_sub<1, false>(context, num, denom);
    warp_int_fast_propagate_sub<1>(context, num);

    #pragma unroll
    for(int32_t index=31;index>=32-N;index--) {
      lo=shfl(num.limbs[0], 30);
      hi=shfl(num.limbs[0], 31);
      est=DIV32(hi, lo, d, approx32);

      while(true) {
        inliner.MULLO(lo, est, denom.limbs[0]);
        lo=shfl(lo, warp_thread+1);
        inliner.MADHI_CC(temp.limbs[0], est, denom.limbs[0], lo);
        inliner.ADDC(context.carry, zero, zero);
        warp_int_fast_propagate_add<1>(context, temp);
        if(warp_int_compare<1>(temp, num)<=0)
          break;
        est--;
      }
      warp_int_sub<1, false>(context, num, temp);
      warp_int_fast_propagate_sub<1>(context, num);

      num.limbs[0]=shfl_up(num.limbs[0], 1);
      if(index==31)
        approx.limbs[0]=(warp_thread==index) ? est : 0;
      else
        approx.limbs[0]=(warp_thread==index) ? est : approx.limbs[0];
    }

    if(warp_thread==32-N) {
      inliner.ADD_CC(approx.limbs[0], approx.limbs[0], one);
      inliner.ADDC(context.carry, zero, zero);
    }
    warp_int_fast_propagate_add<1>(context, approx);
  }

  template<uint32_t N, bool spread>
  __device__ __forceinline__ void DIV_N(Limbs q, Limbs num, Limbs denom, Limbs approx) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    uint32_t   value, carry;
    Context    context;
    uint32_t   registers[3];
    Limbs      temp(registers+0), hi(registers+1), lo(registers+2);
    uint32_t   zero=0;

    if(spread) {
      hi.limbs[0]=num.limbs[0];  // hi=hilo
      lo.limbs[0]=shfl_up(num.limbs[0], N);
    }
    else {
      // get hi and lo
      SPREAD_N<N, true>(hi, num, 31);
      SPREAD_N<N, true>(lo, num, 30);
    }

    // value=(denom>lo) ? 1 : 2;
    value=(warp_int_compare<1>(denom, lo)==1) ? 1 : 2;

    // q = hi * approx + value (value is shifted into high area)
    q.limbs[0]=(warp_thread==0) ? value : 0;
    WARP_PRODUCT_N<N, true, true>(q, hi, approx);

    if(spread)
      q.limbs[0]=(warp_thread>=32-N) ? q.limbs[0] : 0;  // truncate q, so that lo doesn't impact result

    // q = q + hi
    warp_int_add<1, false>(context, q, hi);
    carry=warp_int_fast_propagate_add<1>(context, q);

    // if we carried out, set q to all ones
    q.limbs[0]=(carry==1) ? 0xFFFFFFFF : q.limbs[0];
    q.limbs[0]=(warp_thread<32-N) ? 0 : q.limbs[0];

    // compute temp=q * denom
    WARP_PRODUCT_N<N, true, false>(temp, q, denom);

    if(!spread) {
      // move lo into the right positions
      lo.limbs[0]=shfl_down(lo.limbs[0], N);

      // merge lo into hi
      hi.limbs[0] = (warp_thread>=32-N) ? hi.limbs[0] : lo.limbs[0];
    }

    // move denom into the right positions
    lo.limbs[0]=shfl(denom.limbs[0], warp_thread+N);

    // if q*d>hilo then q--
    warp_int_sub<1, false>(context, hi, temp);
    if(warp_int_fast_propagate_sub<1>(context, hi)!=1)
      return;

    value=(warp_thread==32-N) ? 1 : 0;

    // if hilo + d carries out then, we're good
    warp_int_add<1, false>(context, hi, lo);
    carry=warp_int_fast_propagate_add<1>(context, hi);
    if(carry!=1)
      value=value+value;

    // q=q-value
    inliner.SUB_CC(q.limbs[0], q.limbs[0], value);
    inliner.SUBC(context.carry, zero, zero);
    warp_int_fast_propagate_sub<1>(context, q);
  }

  template<uint32_t N>
  __device__ __forceinline__ void SQRT_N(Context &context, Limbs s, Limbs x) {
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
    x.limbs[0]=(warp_thread==0) ? 0 : x.limbs[0];

    approx=APPROX32(divisor);

    s.limbs[0]=(warp_thread==31) ? divisor+divisor : 0;   // there is a 'silent' 1 at the top of s

    #pragma nounroll
    for(int32_t index=30;index>=32-N;index--) {
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

  template<uint32_t N, bool spread>
  __device__ __forceinline__ void SQRTQ_N(Limbs q, uint32_t hi, Limbs lo, Limbs d, Limbs approx) {
    PTXInliner inliner;
    uint32_t   warp_thread=threadIdx.x & 0x1F;
    Context    context;
    uint32_t   registers[1];
    Limbs      temp(registers);
    uint32_t   add;
    uint32_t   zero=0;

    if(spread)
      temp.limbs[0]=lo.limbs[0];
    else
      SPREAD_N<N, true>(temp, lo, 31);

    warp_int_shift_right_bits<1>(temp, 1, hi);

    add=(warp_thread==32-N-1) ? 0x80000000 : 0;

    inliner.ADD_CC(temp.limbs[0], temp.limbs[0], add);
    inliner.ADDC(context.carry, zero, zero);
    if(warp_int_fast_propagate_add<1>(context, temp)==1) {
      q.limbs[0]=(warp_thread>=32-N) ? 0xFFFFFFFF : 0;
      return;
    }

    DIV_N<N, true>(q, temp, d, approx);
  }


}}