fpa_result fpa_init(fpa_ptr fpa, uint32_t count, uint32_t precision) {
  if(precision<1024 || precision>8192 || precision%1024!=0)
    return fpa_invalid_precisions;
  fpa->count=count;
  fpa->precision=precision;
  $GPU(cudaMalloc(&(fpa->gpu_exps), count*4));
  $GPU(cudaMalloc(&(fpa->gpu_limbs), count*precision/8));
  return fpa_success;
}

fpa_result fpa_clear(fpa_ptr fpa) {
  $GPU(cudaFree(fpa->gpu_exps));
  $GPU(cudaFree(fpa->gpu_limbs));
  return fpa_success;
}

fpa_result fpa_copy_array_to_device(fpa_ptr fpa, uint32_t *exps, uint32_t *limbs) {
  uint32_t *transpose=(uint32_t *)malloc(fpa->count*fpa->precision/8);
  uint32_t  instances=fpa->count, words=fpa->precision/1024;

  for(int32_t instance=0;instance<instances;instance++)
    for(int32_t word=0;word<words;word++)
      for(int32_t thread=0;thread<32;thread++)
        transpose[instance*words*32 + word*32+thread]=limbs[instance*words*32 + thread*words+word];

  $GPU(cudaMemcpy(fpa->gpu_exps, exps, fpa->count*4, cudaMemcpyHostToDevice));
  $GPU(cudaMemcpy(fpa->gpu_limbs, transpose, fpa->count*fpa->precision/8, cudaMemcpyHostToDevice));

  free(transpose);
  return fpa_success;
}

fpa_result fpa_copy_array_to_host(fpa_ptr fpa, uint32_t *exps, uint32_t *limbs) {
  uint32_t *transpose=(uint32_t *)malloc(fpa->count*fpa->precision/8);
  uint32_t  instances=fpa->count, words=fpa->precision/1024;

  $GPU(cudaMemcpy(exps, fpa->gpu_exps, fpa->count*4, cudaMemcpyDeviceToHost));
  $GPU(cudaMemcpy(transpose, fpa->gpu_limbs, fpa->count*fpa->precision/8, cudaMemcpyDeviceToHost));

  for(int32_t instance=0;instance<instances;instance++)
    for(int32_t word=0;word<words;word++)
      for(int32_t thread=0;thread<32;thread++)
        limbs[instance*words*32 + thread*words+word]=transpose[instance*words*32 + word*32+thread];

  free(transpose);
  return fpa_success;
}