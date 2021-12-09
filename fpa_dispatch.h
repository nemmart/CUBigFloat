#define ROUNDUP(x, den) (x+den-1)/den
#define GEOMETRY 128

#ifdef __cplusplus
extern "C" {
#endif

fpa_result fpa_set_ui(fpa_t r, uint64_t *values) {
  fpa_arguments  arguments;
  uint64_t      *gpu_values;

  $GPU(cudaMalloc(&gpu_values, 8*r->count));
  $GPU(cudaMemcpy(gpu_values, values, 8*r->count, cudaMemcpyHostToDevice));

  arguments.count=r->count;
  arguments.precision=r->precision;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.values=gpu_values;

  fpa_kernel_warp_set_ui<<<ROUNDUP(r->count*32, GEOMETRY), GEOMETRY>>>(arguments);

  $GPU(cudaFree(gpu_values));

  return fpa_success;
}

fpa_result fpa_set_si(fpa_t r, int64_t *values) {
  fpa_arguments  arguments;
  int64_t       *gpu_values;

  $GPU(cudaMalloc(&gpu_values, 8*r->count));
  $GPU(cudaMemcpy(gpu_values, values, 8*r->count, cudaMemcpyHostToDevice));

  arguments.count=r->count;
  arguments.precision=r->precision;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.values=gpu_values;

  fpa_kernel_warp_set_si<<<ROUNDUP(r->count*32, GEOMETRY), GEOMETRY>>>(arguments);

  $GPU(cudaFree(gpu_values));

  return fpa_success;
}

fpa_result fpa_set_float(fpa_t r, float *values) {
  fpa_arguments  arguments;
  float         *gpu_values;

  $GPU(cudaMalloc(&gpu_values, 4*r->count));
  $GPU(cudaMemcpy(gpu_values, values, 4*r->count, cudaMemcpyHostToDevice));

  arguments.count=r->count;
  arguments.precision=r->precision;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.values=gpu_values;

  fpa_kernel_warp_set_float<<<ROUNDUP(r->count*32, GEOMETRY), GEOMETRY>>>(arguments);

  $GPU(cudaFree(gpu_values));

  return fpa_success;
}

fpa_result fpa_set_double(fpa_t r, double *values) {
  fpa_arguments  arguments;
  float         *gpu_values;

  $GPU(cudaMalloc(&gpu_values, 8*r->count));
  $GPU(cudaMemcpy(gpu_values, values, 8*r->count, cudaMemcpyHostToDevice));

  arguments.count=r->count;
  arguments.precision=r->precision;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.values=gpu_values;

  fpa_kernel_warp_set_double<<<ROUNDUP(r->count*32, GEOMETRY), GEOMETRY>>>(arguments);

  $GPU(cudaFree(gpu_values));

  return fpa_success;
}

fpa_result fpa_set(fpa_t r, fpa_t x, fpa_rounding_mode mode) {
  fpa_arguments  arguments;
  uint32_t       count=r->count;

  if(count!=x->count)
    return fpa_invalid_counts;

  arguments.count=count;
  arguments.precision=r->precision;
  arguments.set_source_precision=x->precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=x->gpu_exps;
  arguments.a_limbs=x->gpu_limbs;

  fpa_kernel_warp_set<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);

  return fpa_success;
}

fpa_result fpa_sgn(int32_t *r, fpa_t a) {
  fpa_arguments  arguments;

//  $GPU(cudaMalloc(&gpuR, 4*a->count));

  arguments.count=a->count;
  arguments.precision=a->precision;
  arguments.r_exps=(uint32_t *)r;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;

  fpa_kernel_threaded_sgn<<<ROUNDUP(a->count, GEOMETRY), GEOMETRY>>>(arguments);

//  $GPU(cudaMemcpy(r, gpuR, 4*a->count, cudaMemcpyDeviceToHost));
//  $GPU(cudaFree(gpuR));

  return fpa_success;
}

fpa_result fpa_cmp(int32_t *r, fpa_t a, fpa_t b) {
  fpa_arguments arguments;
  uint32_t      count=a->count, precision=a->precision;

  if(count!=b->count)
    return fpa_invalid_counts;
  if(precision!=b->precision)
    return fpa_invalid_precisions;

//  $GPU(cudaMalloc(&gpuR, 4*count));

  arguments.count=count;
  arguments.precision=precision;
  arguments.r_exps=(uint32_t *)r;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  fpa_kernel_threaded_cmp<<<ROUNDUP(count, GEOMETRY), GEOMETRY>>>(arguments);

//  $GPU(cudaMemcpy(r, gpuR, 4*count, cudaMemcpyDeviceToHost));
//  $GPU(cudaFree(gpuR));

  return fpa_success;
}

fpa_result fpa_neg(fpa_ptr r, fpa_ptr a) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count)
    return fpa_invalid_counts;
  if(precision!=a->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;

  fpa_kernel_warp_neg<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
  return fpa_success;
}

fpa_result fpa_add(fpa_ptr r, fpa_ptr a, fpa_ptr b, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count || count!=b->count)
    return fpa_invalid_counts;
  if(precision!=a->precision || precision!=b->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  switch(r->precision) {
    case 1024:
      fpa_kernel_warp_add_1<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 2048:
      fpa_kernel_warp_add_2<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 3072:
      fpa_kernel_warp_add_3<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 4096:
      fpa_kernel_warp_add_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 5120:
      fpa_kernel_warp_add_5<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 6144:
      fpa_kernel_warp_add_6<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 7168:
      fpa_kernel_warp_add_7<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 8192:
      fpa_kernel_warp_add_8<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}

fpa_result fpa_sub(fpa_ptr r, fpa_ptr a, fpa_ptr b, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count || count!=b->count)
    return fpa_invalid_counts;
  if(precision!=a->precision || precision!=b->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  switch(r->precision) {
    case 1024:
      fpa_kernel_warp_sub_1<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 2048:
      fpa_kernel_warp_sub_2<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 3072:
      fpa_kernel_warp_sub_3<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 4096:
      fpa_kernel_warp_sub_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 5120:
      fpa_kernel_warp_sub_5<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 6144:
      fpa_kernel_warp_sub_6<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 7168:
      fpa_kernel_warp_sub_7<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 8192:
      fpa_kernel_warp_sub_8<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}

fpa_result fpa_mul(fpa_ptr r, fpa_ptr a, fpa_ptr b, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count || count!=b->count)
    return fpa_invalid_counts;
  if(precision!=a->precision || precision!=b->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  switch(precision) {
    case 1024:
      fpa_kernel_warp_mul_1<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 2048:
      fpa_kernel_warp_mul_2<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 3072:
      fpa_kernel_warp_mul_3<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 4096:
      fpa_kernel_warp_mul_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 5120:
      fpa_kernel_warp_mul_5<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 6144:
      fpa_kernel_warp_mul_6<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 7168:
      fpa_kernel_warp_mul_7<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 8192:
      fpa_kernel_warp_mul_8<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}

fpa_result fpa_div(fpa_ptr r, fpa_ptr a, fpa_ptr b, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count || count!=b->count)
    return fpa_invalid_counts;
  if(precision!=a->precision || precision!=b->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  switch(precision) {
    case 1024:
      fpa_kernel_warp_div_1<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 2048:
      fpa_kernel_warp_div_2<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 3072:
      fpa_kernel_warp_div_3<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 4096:
      fpa_kernel_warp_div_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 5120:
      fpa_kernel_warp_div_5<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 6144:
      fpa_kernel_warp_div_6<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 7168:
      fpa_kernel_warp_div_7<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 8192:
      fpa_kernel_warp_div_8<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}

fpa_result fpa_sqrt(fpa_ptr r, fpa_ptr a, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count)
    return fpa_invalid_counts;
  if(precision!=a->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;

  switch(precision) {
    case 1024:
      fpa_kernel_warp_sqrt_1<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 2048:
      fpa_kernel_warp_sqrt_2<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 3072:
      fpa_kernel_warp_sqrt_3<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 4096:
      fpa_kernel_warp_sqrt_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 5120:
      fpa_kernel_warp_sqrt_5<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 6144:
      fpa_kernel_warp_sqrt_6<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 7168:
      fpa_kernel_warp_sqrt_7<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    case 8192:
      fpa_kernel_warp_sqrt_8<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}

/*
fpa_result fpa_debug(fpa_ptr r, fpa_ptr a, fpa_ptr b, fpa_rounding_mode mode) {
  fpa_arguments arguments;
  uint32_t      count=r->count, precision=r->precision;

  if(count!=a->count || count!=b->count)
    return fpa_invalid_counts;
  if(precision!=a->precision || precision!=b->precision)
    return fpa_invalid_precisions;

  arguments.count=count;
  arguments.precision=precision;
  arguments.mode=mode;
  arguments.r_exps=r->gpu_exps;
  arguments.r_limbs=r->gpu_limbs;
  arguments.a_exps=a->gpu_exps;
  arguments.a_limbs=a->gpu_limbs;
  arguments.b_exps=b->gpu_exps;
  arguments.b_limbs=b->gpu_limbs;

  switch(precision) {
    case 4096:
      fpa_kernel_warp_debug_4<<<ROUNDUP(count*32, GEOMETRY), GEOMETRY>>>(arguments);
      break;
    default:
      return fpa_invalid_precisions;
  }
  return fpa_success;
}
*/

#ifdef __cplusplus
}
#endif
