### cuBigFloat -- high precision floating point arithmetic in CUDA

The following library provides high precision floating arithmetic in CUDA.  The library was developed
as part of my Ph.D. research, and unfortunately due to other commitments is no longer maintained.

### Library Documentation

Chapter 4 of my [dissertation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=2252&context=dissertations_2)
provides an overview of the library, some documentation on the APIs and algorithms and some performance data.

### Compiling and Running

Requirements:
* Ubuntu Linux machine with a recent GPU card (Volta+)
* Install GMP
* Install MPFR

```
/usr/local/cuda/bin/nvcc -DGPU -arch=sm_70 
./test
```

### Questions

Please feel free to email me at nemmart@yrrid.com, I'll respond as time permits.

