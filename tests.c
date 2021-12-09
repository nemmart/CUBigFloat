#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpfr.h>
#include "tests.h"

#define FP_BIAS 0x3FFFFFF3
#define MODE    fpa_round_ties_to_even

bool state_initialized=false;
gmp_randstate_t state;

static float FLOAT32(uint32_t u) {
  union {
    float f;
    uint32_t u;
  } q;

  q.u=u;
  return q.f;
}

static double DOUBLE64(uint64_t u) {
  union {
    double   d;
    uint64_t u;
  } q;

  q.u=u;
  return q.d;
}

static uint32_t from_nibble(uint32_t nibble) {
  if(nibble>='0' && nibble<='9')
    return nibble-'0';
  else if(nibble>='a' && nibble<='f')
    return nibble-'a'+10;
  else if(nibble>='A' && nibble<='F')
    return nibble-'A'+10;
  else {
    fprintf(stderr, "Invalid nibble: %c\n", nibble);
    exit(1);
  }
}

static uint32_t to_nibble(uint32_t nibble) {
  if(nibble<10)
    return nibble+'0';
  else
    return nibble-10+'a';
}

static mpfr_rnd_t translate(fpa_rounding_mode rounding) {
  switch(rounding) {
    case fpa_round_truncate:
      return MPFR_RNDZ;

    case fpa_round_up:
      return MPFR_RNDU;

    case fpa_round_down:
      return MPFR_RNDD;

    case fpa_round_ties_to_even:
      return MPFR_RNDN;

    case fpa_round_ties_away_from_zero:
      fprintf(stderr, "no equivalent for fpa_round_ties_away_from_zero\n");
      exit(1);
  }
}

const char *tests_name(tests_t *tests) {
  switch(tests->test_type) {
    case tests_set_ui:
      return "set_ui";
    case tests_set_si:
      return "set_si";
    case tests_set_float:
      return "set_float";
    case tests_set_double:
      return "set_double";
    case tests_set:
      return "set";
    case tests_cmp:
      return "cmp";
    case tests_sgn:
      return "sgn";
    case tests_neg:
      return "neg";
    case tests_add:
      return "add";
    case tests_sub:
      return "sub";
    case tests_mul:
      return "mul";
    case tests_div:
      return "div";
    case tests_sqrt:
      return "sqrt";
  }
}

uint32_t tests_random_word() {
  if(!state_initialized) {
    gmp_randinit_default(state);
    state_initialized=true;
  }
  return gmp_urandomb_ui(state, 32);
}

void tests_get_mpfr(mpfr_t value, uint32_t exponent, uint32_t *mantissa, uint32_t precision) {
  mpz_t   mpz;
  int32_t shift;

  if(exponent==2) {
    mpfr_set_ui(value, 0, MPFR_RNDZ);
    return;
  }

  mpz_init(mpz);
  mpz_import(mpz, precision/32, -1, 4, 0, 0, mantissa);
  mpfr_set_z(value, mpz, MPFR_RNDZ);
  if((exponent & 0x01)!=0)
    mpfr_neg(value, value, MPFR_RNDZ);
  shift=(exponent>>1)-FP_BIAS-precision;
  if(shift>0)
    mpfr_mul_2exp(value, value, shift, MPFR_RNDZ);
  else if(shift<0)
    mpfr_div_2exp(value, value, -shift, MPFR_RNDZ);
  mpz_clear(mpz);
}

void tests_set_mpfr(uint32_t *exponent, uint32_t *mantissa, uint32_t precision, mpfr_t value) {
  mpfr_exp_t exp;
  mpfr_t     temp;
  mpz_t      mpz;
  size_t     words;

  if(mpfr_sgn(value)==0) {
    *exponent=2;
    return;
  }

  mpfr_init2(temp, precision);
  mpz_init(mpz);
  mpfr_frexp(&exp, temp, value, MPFR_RNDZ);
  *exponent=2*(FP_BIAS + (int32_t)exp) + (mpfr_sgn(value)==-1);
  mpfr_mul_2exp(temp, temp, precision, MPFR_RNDZ);
  mpfr_get_z(mpz, temp, MPFR_RNDZ);
  mpz_export(mantissa, &words, -1, 4, 0, 0, mpz);
  for(;words<precision/32;words++)
    mantissa[words]=0;

  mpfr_clear(temp);
  mpz_clear(mpz);
}

/*
void tests_random_mpfr(mpfr_t x, uint32_t bits) {
  if(!state_initialized) {
    gmp_randinit_default(state);
    state_initialized=true;
  }

  mpfr_set_prec(x, bits);
  mpfr_urandomb(x, state);
}
*/

void tests_random_mpfr(mpfr_t x, uint32_t bits) {
  int32_t  words=bits/32, index;
  uint32_t data[words];
  uint32_t values[5]={0, 0x00000001, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF};

  for(index=0;index<words;index++)
    data[index]=values[tests_random_word()%5];
  tests_get_mpfr(x, 2*FP_BIAS, data, bits);
}

tests_t *tests_alloc(tests_type_t test_type, uint32_t count, uint32_t precision, uint32_t set_source_precision, fpa_rounding_mode rounding) {
  tests_t *tests=(tests_t *)malloc(sizeof(struct tests_s));

  tests->test_type=test_type;
  tests->count=count;
  tests->precision=precision;
  tests->set_source_precision=set_source_precision;
  tests->rounding=rounding;

  tests->current=0;

  tests->gold_exps=NULL;
  tests->gold_limbs=NULL;
  tests->r_exps=NULL;
  tests->r_limbs=NULL;
  tests->a_exps=NULL;
  tests->a_limbs=NULL;
  tests->b_exps=NULL;
  tests->b_limbs=NULL;
  tests->values=NULL;
}

void tests_free(tests_t *tests) {
  if(tests->gold_exps!=NULL)
    free(tests->gold_exps);
  if(tests->gold_limbs!=NULL)
    free(tests->gold_limbs);
  if(tests->r_exps!=NULL)
    free(tests->r_exps);
  if(tests->r_limbs!=NULL)
    free(tests->r_limbs);
  if(tests->a_exps!=NULL)
    free(tests->a_exps);
  if(tests->a_limbs!=NULL)
    free(tests->a_limbs);
  if(tests->b_exps!=NULL)
    free(tests->b_exps);
  if(tests->b_limbs!=NULL)
    free(tests->b_limbs);
  free(tests);
}

void tests_add_mpfr_ui64(tests_t *tests, mpfr_t gold, uint64_t value) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->values=(uint64_t *)malloc(8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  ((uint64_t *)tests->values)[current]=value;
  tests->current++;
}

void tests_add_mpfr_float(tests_t *tests, mpfr_t gold, float value) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->values=(float *)malloc(4*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  ((float *)tests->values)[current]=value;
  tests->current++;
}

void tests_add_mpfr_double(tests_t *tests, mpfr_t gold, double value) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->values=(double *)malloc(8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  ((double *)tests->values)[current]=value;
  tests->current++;
}

void tests_add_compare_mpfr(tests_t *tests, int32_t gold, mpfr_t a) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(precision/8*count);
  }
  tests->gold_exps[current]=(uint32_t)gold;
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[current*precision/32]), precision, a);
  tests->current++;
}

void tests_add_compare_mpfr2(tests_t *tests, int32_t gold, mpfr_t a, mpfr_t b) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(precision/8*count);
    tests->b_exps=(uint32_t *)malloc(4*count);
    tests->b_limbs=(uint32_t *)malloc(precision/8*count);
  }
  tests->gold_exps[current]=(uint32_t)gold;
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[current*precision/32]), precision, a);
  tests_set_mpfr(&(tests->b_exps[current]), &(tests->b_limbs[current*precision/32]), precision, b);
  tests->current++;
}

void tests_add_mpfr2(tests_t *tests, mpfr_t gold, mpfr_t a) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(precision/8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[current*precision/32]), precision, a);
  tests->current++;
}

void tests_add_source_mpfr2(tests_t *tests, mpfr_t gold, mpfr_t a) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t source_precision=tests->set_source_precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(source_precision/8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[current*source_precision/32]), source_precision, a);
  tests->current++;
}

void tests_add_mpfr2_short_long(tests_t *tests, mpfr_t gold, mpfr_t a) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(2*precision/8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[2*current*precision/32]), precision*2, a);
  tests->current++;
}

void tests_add_mpfr3(tests_t *tests, mpfr_t gold, mpfr_t a, mpfr_t b) {
  uint32_t count=tests->count;
  uint32_t precision=tests->precision;
  uint32_t current=tests->current;

  if(current==tests->count) {
    fprintf(stderr, "too many tests\n");
    exit(1);
  }
  if(current==0) {
    tests->gold_exps=(uint32_t *)malloc(4*count);
    tests->gold_limbs=(uint32_t *)malloc(precision/8*count);
    tests->r_exps=(uint32_t *)malloc(4*count);
    tests->r_limbs=(uint32_t *)malloc(precision/8*count);
    tests->a_exps=(uint32_t *)malloc(4*count);
    tests->a_limbs=(uint32_t *)malloc(precision/8*count);
    tests->b_exps=(uint32_t *)malloc(4*count);
    tests->b_limbs=(uint32_t *)malloc(precision/8*count);
  }
  tests_set_mpfr(&(tests->gold_exps[current]), &(tests->gold_limbs[current*precision/32]), precision, gold);
  tests_set_mpfr(&(tests->a_exps[current]), &(tests->a_limbs[current*precision/32]), precision, a);
  tests_set_mpfr(&(tests->b_exps[current]), &(tests->b_limbs[current*precision/32]), precision, b);
  tests->current++;
}

void tests_compare_results(tests_t *tests) {
  uint32_t count=tests->count;
  uint32_t words=tests->precision/32;
  bool     mismatch;
  int32_t  instance, word, show;

  for(instance=0;instance<count;instance++) {
    mismatch=tests->gold_exps[instance]!=tests->r_exps[instance];
    if(tests->gold_limbs!=NULL && tests->gold_exps[instance]!=2)
      for(word=0;word<words;word++)
        mismatch=mismatch | tests->gold_limbs[instance*words + word]!=tests->r_limbs[instance*words + word];
    if(mismatch) {
      if(tests->gold_limbs==NULL)
        printf("Mismatch on test case %d:   correct=%d  computed=%d\n", instance, tests->gold_exps[instance], tests->r_exps[instance]);
      else
        printf("Mismatch on test case %d\n", instance);
      if(tests->a_limbs!=NULL) {
        printf("        A=");
        if(tests->a_exps[instance]%2==0)
          printf("+0.");
        else
          printf("-0.");
        for(show=words-1;show>=0;show--)
          printf("%08X", tests->a_limbs[instance*words + show]);
        printf(" e%d\n", tests->a_exps[instance]/2 - FP_BIAS);
      }
      if(tests->b_limbs!=NULL) {
        printf("        B=");
        if(tests->b_exps[instance]%2==0)
          printf("+0.");
        else
          printf("-0.");
        for(show=words-1;show>=0;show--)
          printf("%08X", tests->b_limbs[instance*words + show]);
        printf(" e%d\n", tests->b_exps[instance]/2 - FP_BIAS);
      }
      if(tests->gold_limbs!=NULL) {
        printf("Correct:  ");
        if(tests->gold_exps[instance]%2==0)
          printf("+0.");
        else
          printf("-0.");
        for(show=words-1;show>=0;show--)
          if(tests->gold_limbs[instance*words + show]==tests->r_limbs[instance*words + show])
            printf("%08X", tests->gold_limbs[instance*words + show]);
          else
            printf("\033[1m%08X\033[0m", tests->gold_limbs[instance*words + show]);
        printf(" e%d\n", tests->gold_exps[instance]/2 - FP_BIAS);
      }
      if(tests->r_limbs!=NULL) {
        printf("Computed: ");
        if(tests->r_exps[instance]%2==0)
          printf("+0.");
        else
          printf("-0.");
        for(show=words-1;show>=0;show--)
          printf("%08X", tests->r_limbs[instance*words + show]);
        printf(" e%d\n", tests->r_exps[instance]/2 - FP_BIAS);
      }
      exit(1);
    }
  }
}

tests_t *tests_generate_set_ui(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_set_ui, count, precision, 0, MODE);
  int32_t  index, shift;
  uint64_t x;
  mpfr_t   r;

  mpfr_init2(r, precision);
  for(index=0;index<count;index++) {
    x=tests_random_word();
    x=(x<<32)+tests_random_word();

    mpfr_set_ui(r, x, MPFR_RNDZ);
    tests_add_mpfr_ui64(tests, r, x);
  }
  mpfr_clear(r);
  return tests;
}

tests_t *tests_generate_set_si(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_set_si, count, precision, 0, MODE);
  int32_t  index, shift;
  int64_t  x;
  mpfr_t   r;

  mpfr_init2(r, precision);
  for(index=0;index<count;index++) {
    x=tests_random_word();
    x=(x<<32)+tests_random_word();

    mpfr_set_si(r, x, MPFR_RNDZ);
    tests_add_mpfr_ui64(tests, r, x);
  }
  mpfr_clear(r);
  return tests;
}

tests_t *tests_generate_set_float(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_set_float, count, precision, 0, MODE);
  float    denormal=FLOAT32(1);
  float    simple[10]={1.0, 0.1, 1234e10, 1234e-10, -1.0, -0.1, -1234e10, -1234e-10, denormal, -denormal};
  double   powers[9]={1e-20, 1e-10, 1, 1e10, 1e20};
  int32_t  index, shift;
  float    x;
  mpfr_t   r;

  mpfr_init2(r, precision);
  for(index=0;index<count;index++) {
    if(index<10)
      x=simple[index];
    else {
      x=tests_random_word();
      x=x*powers[tests_random_word()%5];
      if(tests_random_word()%2==0)
        x=-x;
    }

    mpfr_set_flt(r, x, MPFR_RNDZ);
    tests_add_mpfr_float(tests, r, x);
  }
  mpfr_clear(r);
  return tests;
}

tests_t *tests_generate_set_double(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_set_double, count, precision, 0, MODE);
  double   denormal=DOUBLE64(1);
  double   simple[10]={1.0, 0.1, 1234e10, 1234e-10, -1.0, -0.1, -1234e200, -1234e-200, denormal, -denormal};
  double   powers[12]={1e-50, 1e-40, 1e-30, 1e-20, 1e-10, 0, 1, 1e10, 1e20, 1e30, 1e40, 1e50};
  int32_t  index, shift;
  double   x;
  mpfr_t   r;

  mpfr_init2(r, precision);
  for(index=0;index<count;index++) {
    if(index<10)
      x=simple[index];
    else {
      uint64_t ui;

      ui=tests_random_word();
      ui=(ui<<32) + tests_random_word();
      x=((double)ui)*powers[tests_random_word()%12];
      if(tests_random_word()%2==0)
        x=-x;
    }

    mpfr_set_d(r, x, MPFR_RNDZ);
    tests_add_mpfr_double(tests, r, x);
  }
  mpfr_clear(r);
  return tests;
}

tests_t *tests_generate_set(uint32_t count, uint32_t precision, uint32_t source_precision) {
  tests_t *tests=tests_alloc(tests_set, count, precision, source_precision, MODE);
  int32_t  index, shift;
  mpfr_t   r, x;

  mpfr_init2(r, precision);
  mpfr_init2(x, source_precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, source_precision);

    shift=tests_random_word()%(source_precision*5/8)-tests_random_word()%(source_precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);

    mpfr_set(r, x, translate(tests->rounding));
    tests_add_source_mpfr2(tests, r, x);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  return tests;
}

/*
tests_t *tests_generate_set_shrink(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_set, count, precision, MODE);
  int32_t  index, shift;
  mpfr_t   r, x;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision*2);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision*2);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);

    mpfr_set(r, x, translate(tests->rounding));
    tests_add_mpfr2_short_long(tests, r, x);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  return tests;
}
*/

tests_t *tests_generate_sgn(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_sgn, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   x, y;
  int32_t  r;

  mpfr_init2(x, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);

    r=mpfr_sgn(x);
    tests_add_compare_mpfr(tests, r, x);
  }
  mpfr_clear(x);
  return tests;
}

tests_t *tests_generate_cmp(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_cmp, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   x, y;
  int32_t  r;

  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);
    tests_random_mpfr(y, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(y, y, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(y, y, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);
    if(tests_random_word()%2==0)
      mpfr_neg(y, y, MPFR_RNDZ);

    r=mpfr_cmp(x, y);
    tests_add_compare_mpfr2(tests, r, x, y);
  }
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}


tests_t *tests_generate_neg(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_neg, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   r, x;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);

    mpfr_neg(r, x, translate(tests->rounding));
    tests_add_mpfr2(tests, r, x);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  return tests;
}

tests_t *tests_generate_add(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_add, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   r, x, y;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);
    tests_random_mpfr(y, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(y, y, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(y, y, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);
    if(tests_random_word()%2==0)
      mpfr_neg(y, y, MPFR_RNDZ);

    mpfr_add(r, x, y, translate(tests->rounding));
    tests_add_mpfr3(tests, r, x, y);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}

tests_t *tests_generate_sub(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_sub, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   r, x, y;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);
    tests_random_mpfr(y, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(y, y, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(y, y, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);
    if(tests_random_word()%2==0)
      mpfr_neg(y, y, MPFR_RNDZ);

    mpfr_sub(r, x, y, translate(tests->rounding));
    tests_add_mpfr3(tests, r, x, y);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}

tests_t *tests_generate_mul(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_mul, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   r, x, y;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  for(index=0;index<count;index++) {
    tests_random_mpfr(x, precision);
    tests_random_mpfr(y, precision);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(y, y, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(y, y, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);
    if(tests_random_word()%2==0)
      mpfr_neg(y, y, MPFR_RNDZ);

    mpfr_mul(r, x, y, translate(tests->rounding));
    tests_add_mpfr3(tests, r, x, y);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}

tests_t *tests_generate_div(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_div, count, precision, 0, MODE);
  int32_t  index, shift;
  mpfr_t   r, x, y;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  for(index=0;index<count;index++) {
//    do {
      tests_random_mpfr(x, precision);
      tests_random_mpfr(y, precision);
//      if(mpfr_cmp_d(x, 0.5)<0)
//        mpfr_add_d(x, x, 0.5, MPFR_RNDZ);
//      if(mpfr_cmp_d(y, 0.5)<0)
//        mpfr_add_d(y, y, 0.5, MPFR_RNDZ);
//    } while(mpfr_cmp(x, y)>=0);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(y, y, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(y, y, -shift, MPFR_RNDZ);

    if(tests_random_word()%2==0)
      mpfr_neg(x, x, MPFR_RNDZ);
    if(tests_random_word()%2==0)
      mpfr_neg(y, y, MPFR_RNDZ);

    mpfr_div(r, x, y, translate(tests->rounding));
    tests_add_mpfr3(tests, r, x, y);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}

tests_t *tests_generate_sqrt(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(tests_sqrt, count, precision, 0, MODE);
  int32_t  index, shift, random;
  mpfr_t   r, s, x;

  mpfr_init2(r, precision);
  mpfr_init2(s, precision/2);
  mpfr_init2(x, precision);
  for(index=0;index<count;index++) {
    random=tests_random_word()%4;
    switch(random) {
      case 0: case 1: case 2:
        tests_random_mpfr(s, precision/2);
        while(mpfr_cmp_d(s, 0.75)<0)
          mpfr_add_d(s, s, 0.25, MPFR_RNDZ);
        mpfr_set(x, s, MPFR_RNDZ);
        mpfr_mul(x, x, x, MPFR_RNDZ);
        mpfr_mul_2exp(x, x, precision, MPFR_RNDZ);
        mpfr_add_si(x, x, random-1, MPFR_RNDZ);
        mpfr_div_2exp(x, x, precision, MPFR_RNDZ);
        mpfr_sqrt(r, x, translate(tests->rounding));
        break;
      case 3:
        tests_random_mpfr(x, precision);
        break;
    }

    shift=tests_random_word()%(precision*5/8)-tests_random_word()%(precision*5/8);
    if(shift>0)
      mpfr_mul_2exp(x, x, shift, MPFR_RNDZ);
    else if(shift<0)
      mpfr_div_2exp(x, x, -shift, MPFR_RNDZ);

    mpfr_sqrt(r, x, translate(tests->rounding));
    tests_add_mpfr2(tests, r, x);
  }
  mpfr_clear(r);
  mpfr_clear(s);
  mpfr_clear(x);
  return tests;
}

/*
tests_t *tests_generate_debug(uint32_t count, uint32_t precision) {
  tests_t *tests=tests_alloc(count, precision);
  int32_t  index, shift;
  mpfr_t   r, x, y, z;

  mpfr_init2(r, precision);
  mpfr_init2(x, precision);
  mpfr_init2(y, precision);
  mpfr_init2(z, precision);
  for(index=0;index<count;index++) {
    do {
      tests_random_mpfr(z, 256);
      if(mpfr_cmp_d(z, 0.5)<0)
        mpfr_add_d(z, z, 0.5, MPFR_RNDZ);
      mpfr_set(x, z, MPFR_RNDZ);

      tests_random_mpfr(z, 128);
      if(mpfr_cmp_d(z, 0.5)<0)
        mpfr_add_d(z, z, 0.5, MPFR_RNDZ);
      mpfr_set(y, z, MPFR_RNDZ);
    } while(mpfr_cmp(x, y)>=0);


    mpfr_set_prec(z, 128);
    mpfr_div(z, x, y, MPFR_RNDZ);
    mpfr_set(r, z, MPFR_RNDZ);
    tests_add_mpfr3(tests, r, x, y);
  }
  mpfr_clear(r);
  mpfr_clear(x);
  mpfr_clear(y);
  return tests;
}
*/

float *tests_run() {
}
