#ifndef FPA_H
#define FPA_H

#include <stdint.h>

#define GPU_FPA

typedef enum {
  fpa_round_truncate, fpa_round_up, fpa_round_down, fpa_round_ties_to_even, fpa_round_ties_away_from_zero
} fpa_rounding_mode;

typedef enum {
  fpa_success,
  fpa_math_error,
  fpa_invalid_precisions,
  fpa_invalid_counts,
} fpa_result;

#ifdef GPU
  struct fpa_s {
    uint32_t count;
    uint32_t precision;
    uint32_t *gpu_exps;
    uint32_t *gpu_limbs;
  };
#endif

#ifdef CPU
  struct fpa_s {
    uint32_t count;
    uint32_t precision;
    mpfr_t   *mpfr;
  };
#endif

typedef struct fpa_s fpa_t[1];
typedef struct fpa_s *fpa_ptr;

#ifdef __cplusplus
extern "C" {
#endif

fpa_result fpa_init(fpa_t fpa, uint32_t count, uint32_t precision);
fpa_result fpa_clear(fpa_t fpa);

fpa_result fpa_copy_array_to_device(fpa_t fpa, uint32_t *exps, uint32_t *limbs);
fpa_result fpa_copy_array_to_host(fpa_t fpa, uint32_t *exps, uint32_t *limbs);

fpa_result fpa_set_ui(fpa_t r, uint64_t *ui);
fpa_result fpa_set_si(fpa_t r, int64_t *si);
fpa_result fpa_set_float(fpa_t r, float *floats);
fpa_result fpa_set_double(fpa_t r, double *doubles);
fpa_result fpa_set(fpa_t r, fpa_t x, fpa_rounding_mode mode);

fpa_result fpa_sgn(int32_t *r, fpa_t x);
fpa_result fpa_cmp(int32_t *r, fpa_t a, fpa_t b);

fpa_result fpa_neg(fpa_t r, fpa_t a);
fpa_result fpa_add(fpa_t r, fpa_t a, fpa_t b, fpa_rounding_mode mode);
fpa_result fpa_sub(fpa_t r, fpa_t a, fpa_t b, fpa_rounding_mode mode);
fpa_result fpa_mul(fpa_t r, fpa_t a, fpa_t b, fpa_rounding_mode mode);
fpa_result fpa_div(fpa_t r, fpa_t a, fpa_t b, fpa_rounding_mode mode);
fpa_result fpa_sqrt(fpa_t r, fpa_t x, fpa_rounding_mode mode);

#ifdef __cplusplus
}
#endif

#endif
