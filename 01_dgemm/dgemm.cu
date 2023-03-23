/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../common/common.h"

#define BLOCK_WIDTH 32
#define TAILLE 4096

void init(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0;

  srand(2023);

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      A[i * size + j] = rand();
      B[i * size + j] = rand();
      C[i * size + j] = 0.0;
    }
  }
}

void mult(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0, k = 0;

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      double sum = 0.;
      for(k = 0; k < size; k++)
      {
        sum += A[i * size + k] * B[k* size + j];
      }
      C[i * size + j] = sum;
    }
  }
}

__global__
void MulMatrixKernel(double* A, double* B, double* C, int N)
{
  int col    = threadIdx.x + blockDim.x * blockIdx.x;
  int line  = threadIdx.y + blockDim.y * blockIdx.y;

  if((col < N) && (line < N))
  {
    double val = 0.0f;
    for(int k = 0; k < N; k++)
    {
      val += A[line * N + k] * B[k * N + col];
    }
    C[line * N + col] = val;
  }
}

int main(int argc, char** argv){
  int N;

  double *A;
  double *B;
  double *C;

  double t0 = 0., t1 = 0., duration = 0.;

  N = (argc < 2)?1024:atoi(argv[1]);
  fprintf(stdout, "N = %d\n", N);

  // Memory allocation
  A = (double*) malloc(sizeof(double) * N * N);
  B = (double*) malloc(sizeof(double) * N * N);
  C = (double*) malloc(sizeof(double) * N * N);

  // Value initialization
  init(A, B, C, N);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(double) * N * N);
  cudaMalloc(&d_B, sizeof(double) * N * N);
  cudaMalloc(&d_C, sizeof(double) * N * N);

  cudaMemcpy(d_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(double) * N * N, cudaMemcpyHostToDevice);

  int nbBlocks = N / BLOCK_WIDTH;
  if(N % BLOCK_WIDTH) nbBlocks++;
  dim3 gridSize(nbBlocks, nbBlocks);
  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);

  t0 = get_elapsedtime();

  MulMatrixKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  t1 = get_elapsedtime();

  cudaMemcpy(C, d_C, sizeof(double) * N * N, cudaMemcpyDeviceToHost);

  duration = (t1 - t0);

  uint64_t nb_op = N * N * N;
  fprintf(stdout, "cuda:time: %lfs\n", duration);
  fprintf(stdout, "cuda:mflops: %.2f\n", (nb_op / duration)*1E-6);

  // Compute multiplication
  t0 = get_elapsedtime();
  mult(A, B, C, N);
  t1 = get_elapsedtime();

  // Pretty print
  duration = (t1 - t0);
  fprintf(stdout, "seq:time: %lfs\n", duration);
  fprintf(stdout, "seq:mflops: %.2f\n", (nb_op / duration)*1E-6);

  free(A);
  free(B);
  free(C);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
