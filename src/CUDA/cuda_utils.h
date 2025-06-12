#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda.h>
#include <stdio.h>  // Asegúrate de incluir esto para printf

#define HANDLE_RESULT(expr) {cudaError_t _asdf__err; if ((_asdf__err = expr) != cudaSuccess) { printf("cuda call failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_asdf__err)); exit(1);}}

__device__ int cuda_str_eq(const char *s1, const char *s2);

__device__ int cuda_atoi(const char *str);

__device__ int cuda_strlen(const char *str);

__device__ int cuda_sprintf_int(char* str, int n);

__device__ unsigned int jenkins_hash(int j, const char *str);

#endif
