#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuda.h>
#include "ptx/ptx.h"

#include "fpa.h"
#include "fpa_internal.h"

using namespace xmp;

#define GEOMETRY 128

// GPU side
#include "fpa_interthread_communication.h"
#include "fpa_math.h"
#include "int_warp_math.h"

// warp per instance
#include "fpa_warp_copy.h"
#include "fpa_warp_set.h"
#include "fpa_warp_add_sub.h"
#include "fpa_warp_mul.h"
#include "fpa_warp_div.h"
#include "fpa_warp_sqrt.h"

// thread per instance
#include "fpa_threaded.h"

// dispatch kernels
#include "fpa_kernels.h"

// CPU side
#define $GPU(call) if((call)!=0) { printf("\nCall \"" #call "\" failed from %s, line %d\n", __FILE__, __LINE__); exit(1); }

extern "C" {
  #include "fpa_management.h"
  #include "fpa_dispatch.h"
}
