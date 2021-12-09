#include <stdio.h>
#include <stdint.h>
#include <mpfr.h>
#include "fpa.h"

typedef enum {
  tests_set_ui, tests_set_si, tests_set_float, tests_set_double, tests_set, tests_cmp, tests_sgn, tests_neg, tests_add, tests_sub, tests_mul, tests_div, tests_sqrt
} tests_type_t;

typedef struct tests_s {
  tests_type_t      test_type;
  uint32_t          count;
  uint32_t          precision;
  uint32_t          set_source_precision;
  fpa_rounding_mode rounding;

  uint32_t          current;

  uint32_t         *gold_exps;  // overloaded: exps or values for cmp and sgn
  uint32_t         *gold_limbs;
  uint32_t         *r_exps;     // overloaded: exps or values for cmp and sgn
  uint32_t         *r_limbs;
  uint32_t         *a_exps;
  uint32_t         *a_limbs;
  uint32_t         *b_exps;
  uint32_t         *b_limbs;

  void             *values;     // used for set
} tests_t;

void tests_get_mpfr(mpfr_t value, uint32_t exponent, uint32_t *mantissa, uint32_t precision);
void tests_set_mpfr(uint32_t *exponent, uint32_t *mantissa, uint32_t precision, mpfr_t value);


tests_t *tests_alloc(tests_type_t test_type, uint32_t count, uint32_t precision, uint32_t set_source_precision, fpa_rounding_mode rounding);
void tests_free(tests_t *tests);
void tests_compare_results(tests_t *tests);

const char *tests_name(tests_t *tests);

tests_t *tests_generate_set_ui(uint32_t count, uint32_t precision);
tests_t *tests_generate_set_si(uint32_t count, uint32_t precision);
tests_t *tests_generate_set_float(uint32_t count, uint32_t precision);
tests_t *tests_generate_set_double(uint32_t count, uint32_t precision);
tests_t *tests_generate_set(uint32_t count, uint32_t precision, uint32_t source_precision);
tests_t *tests_generate_sgn(uint32_t count, uint32_t precision);
tests_t *tests_generate_cmp(uint32_t count, uint32_t precision);
tests_t *tests_generate_neg(uint32_t count, uint32_t precision);
tests_t *tests_generate_add(uint32_t count, uint32_t precision);
tests_t *tests_generate_sub(uint32_t count, uint32_t precision);
tests_t *tests_generate_mul(uint32_t count, uint32_t precision);
tests_t *tests_generate_div(uint32_t count, uint32_t precision);
tests_t *tests_generate_sqrt(uint32_t count, uint32_t precision);

tests_t *tests_generate_debug(uint32_t count, uint32_t precision);
