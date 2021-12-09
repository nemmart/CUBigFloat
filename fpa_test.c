#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "fpa.h"
#include "tests.h"

#define $GPU(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d\n", __FILE__, __LINE__); exit(1); }

void test_set_ui(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r;

  printf("set_ui tests at %d bits\n", precision);

  tests=tests_generate_set_ui(count, precision);
  fpa_init(r, count, precision);

  fpa_set_ui(r, (uint64_t *)tests->values);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set_ui tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
}

void test_set_si(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r;

  printf("set_si tests at %d bits\n", precision);

  tests=tests_generate_set_si(count, precision);
  fpa_init(r, count, precision);

  fpa_set_si(r, (uint64_t *)tests->values);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set_si tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
}

void test_set_float(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r;

  printf("set_float tests at %d bits\n", precision);

  tests=tests_generate_set_float(count, precision);
  fpa_init(r, count, precision);

  fpa_set_float(r, (float *)tests->values);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set_float tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
}

void test_set_double(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r;

  printf("set_double tests at %d bits\n", precision);

  tests=tests_generate_set_double(count, precision);
  fpa_init(r, count, precision);

  fpa_set_double(r, (double *)tests->values);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set_double tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
}

/*
void test_set_extend(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a;

  printf("set (extend) tests at %d bits\n", precision);

  tests=tests_generate_set_extend(count, precision);
  fpa_init(r, count, precision*2);
  fpa_init(a, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_set(r, a, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set (extend) tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
}

void test_set_shrink(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a;

  printf("set (shrink) tests at %d bits\n", precision);

  tests=tests_generate_set_shrink(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision*2);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_set(r, a, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("set (shrink) tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
}
*/

void test_sgn(uint32_t count, uint32_t precision) {
  tests_t  *tests;
  fpa_t     a;
  uint32_t *gpuR;

  printf("sgn tests at %d bits\n", precision);

  tests=tests_generate_sgn(count, precision);
  fpa_init(a, count, precision);

  $GPU(cudaMalloc((void **)&gpuR, count*4));
  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_sgn(gpuR, a);

  $GPU(cudaMemcpy(tests->r_exps, gpuR, count*4, cudaMemcpyDeviceToHost));

  tests_compare_results(tests);

  printf("sgn tests passed\n\n");
  tests_free(tests);
  fpa_clear(a);
  $GPU(cudaFree(gpuR));
}

void test_cmp(uint32_t count, uint32_t precision) {
  tests_t  *tests;
  fpa_t     a, b;
  uint32_t *gpuR;

  printf("cmp tests at %d bits\n", precision);

  tests=tests_generate_cmp(count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  $GPU(cudaMalloc((void **)&gpuR, count*4));
  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_cmp(gpuR, a, b);

  $GPU(cudaMemcpy(tests->r_exps, gpuR, count*4, cudaMemcpyDeviceToHost));

  tests_compare_results(tests);

  printf("cmp tests passed\n\n");
  tests_free(tests);
  fpa_clear(a);
  fpa_clear(b);
  $GPU(cudaFree(gpuR));
}

void test_neg(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a;

  printf("neg tests at %d bits\n", precision);

  tests=tests_generate_neg(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_neg(r, a);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("neg tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
}

void test_add(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a, b;

  printf("add tests at %d bits\n", precision);

  tests=tests_generate_add(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_add(r, a, b, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("add tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
  fpa_clear(b);
}

void test_sub(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a, b;

  printf("sub tests at %d bits\n", precision);

  tests=tests_generate_sub(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_sub(r, a, b, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("sub tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
  fpa_clear(b);
}

void test_mul(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a, b;

  printf("mul tests at %d bits\n", precision);

  tests=tests_generate_mul(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_mul(r, a, b, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("mul tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
  fpa_clear(b);
}

void test_div(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a, b;

  printf("div tests at %d bits\n", precision);

  tests=tests_generate_div(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_div(r, a, b, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("div tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
  fpa_clear(b);
}

void test_sqrt(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a;

  printf("sqrt tests at %d bits\n", precision);

  tests=tests_generate_sqrt(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_sqrt(r, a, tests->rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("sqrt tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
}

/*
void test_debug(uint32_t count, uint32_t precision) {
  tests_t *tests;
  fpa_t    r, a, b;

  printf("debug tests at %d bits\n", precision);

  tests=tests_generate_debug(count, precision);
  fpa_init(r, count, precision);
  fpa_init(a, count, precision);
  fpa_init(b, count, precision);

  fpa_copy_array_to_device(a, tests->a_exps, tests->a_limbs);
  fpa_copy_array_to_device(b, tests->b_exps, tests->b_limbs);
  fpa_debug(r, a, b, tests->fpa_rounding);
  fpa_copy_array_to_host(r, tests->r_exps, tests->r_limbs);

  tests_compare_results(tests);

  printf("debug tests passed\n\n");
  tests_free(tests);
  fpa_clear(r);
  fpa_clear(a);
  fpa_clear(b);
}
*/

int main() {
/*
  test_set_ui(1000, 1024);
  test_set_ui(1000, 5120);
  test_set_si(1000, 1024);
  test_set_si(1000, 5120);
  test_set_float(1000, 1024);
  test_set_float(1000, 5120);
  test_set_double(1000, 1024);
  test_set_double(1000, 5120);

//  test_set_extend(1000, 1024);
//  test_set_extend(1000, 5120);
//  test_set_shrink(1000, 1024);
//  test_set_shrink(1000, 5120);

  test_sgn(10000, 1024);
  test_sgn(10000, 5120);
  test_cmp(1000000, 1024);
  test_cmp(100000, 5120);

  test_neg(100000, 1024);
  test_neg(10000, 5120);
  test_add(1000000, 1024);
  test_add(100000, 5120);
  test_sub(1000000, 1024);
  test_sub(100000, 5120);
*/
  test_mul(1000000, 2048);
  test_mul(100000, 5120);
  test_div(1000000, 1024);
  test_div(100000, 2048);

  test_sqrt(1000000, 1024);
  test_sqrt(100000, 5120);

//  test_debug(1000000, 4096);

}
