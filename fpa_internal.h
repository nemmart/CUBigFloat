#define FPA_BIAS       0x3FFFFFF3

#define FPA_ROUND_TRUNCATE            0   // TRUNC
#define FPA_ROUND_UP                  1   // UP
#define FPA_ROUND_DOWN                2   // DOWN
#define FPA_ROUND_TIES_TO_EVEN        3   // TTE
#define FPA_ROUND_TIES_AWAY_FROM_ZERO 4   // TAFZ

#define FPA_POS_VALUE    0
#define FPA_NEG_VALUE    1
#define FPA_ZERO         2
#define FPA_NEG_TINY     3
#define FPA_POS_TINY     4
#define FPA_NEG_INF      5
#define FPA_POS_INF      6
#define FPA_NAN          7
#define FPA_SNAN         8

#define FPA_ACTION       16
#define FPA_MUL          16
#define FPA_DIV          17
#define FPA_SQRT         18

#define FPA_NEG          19
#define FPA_ADD          20
#define FPA_SUB          21
#define FPA_COPY_A       22
#define FPA_COPY_B       23
#define FPA_NEG_B        24

#define FPA_CMP          25
#define FPA_CMP_ERR      26

typedef struct {
  uint32_t           count;
  uint32_t           precision;
  uint32_t           set_source_precision;
  fpa_rounding_mode  mode;
  uint32_t          *r_exps;
  uint32_t          *r_limbs;
  uint32_t          *a_exps;
  uint32_t          *a_limbs;
  uint32_t          *b_exps;
  uint32_t          *b_limbs;
  void              *values;
} fpa_arguments;

typedef enum {
  fpa_add_operation, fpa_sub_operation,
} fpa_operation_t;

__device__ __forceinline__ uint32_t fpa_classify1(uint32_t x_exp) {
  return (x_exp<=FPA_SNAN) ? x_exp : x_exp & 0x01;
}

__device__ __forceinline__ uint32_t fpa_classify2(uint32_t a_exp, uint32_t b_exp) {
  a_exp=(a_exp<=FPA_SNAN) ? a_exp : a_exp & 0x01;
  b_exp=(b_exp<=FPA_SNAN) ? b_exp : b_exp & 0x01;
  return a_exp * 9 + b_exp;
}

__device__ __forceinline__ uint32_t fpa_round(fpa_rounding_mode mode, uint32_t signWord, uint32_t evenWord, uint32_t roundWord, uint32_t stickyWords) {
  switch(mode) {
    case fpa_round_truncate:
      return 0;

    case fpa_round_up:
      if(signWord!=0)
        return 0;
      return (roundWord | stickyWords)!=0;

    case fpa_round_down:
      if(signWord==0)
        return 0;
      return (roundWord | stickyWords)!=0;

    case fpa_round_ties_away_from_zero:
      return roundWord>=0x80000000u;

    case fpa_round_ties_to_even:
      if(roundWord<0x80000000u)
        return 0;
      if(roundWord>0x80000000u)
        return 1;
      return stickyWords!=0 | (evenWord & 0x01);
  }
  return 0;
}