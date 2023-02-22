/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "../common/common.h"

#define ITER 1000
#define SIZE 4*1000000

__global__ void kernel(int* y, int* a, int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	y[i] = a[i] * b[i];
}

__host__ void reduction(uint64_t resultat, int * a)
{

	int i;
  resultat = (uint64_t)0;
	for(i=0; i< SIZE; i++)
  {
    resultat += a[i];
  }
}

int main()
{

	int* a; int* b; int* c;

	int j;
	int i;
  uint64_t res1, res2;
  double t0 = 0., t1 = 0., duration = 0.;
  double t2 = 0., t3 = 0., duration2 = 0.;

	cudaMallocManaged(&a, SIZE*sizeof(int));
	cudaMallocManaged(&b, SIZE*sizeof(int));
	cudaMallocManaged(&c, SIZE*sizeof(int));

	for(i=0; i<SIZE; i++)
  {
    a[i] = i;
    b[i] = 2;
    c[i] = 0;
  }

  // Prefetch the data to the GPU
  int device = -1;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(a, SIZE*sizeof(float), device, NULL);
  cudaMemPrefetchAsync(b, SIZE*sizeof(float), device, NULL);

  int blockSize = 256;
  int numBlocks = (SIZE + blockSize - 1) / blockSize;


  t0 = get_elapsedtime();
	for(j=0; j<ITER; j++)
	{
  	cudaMemPrefetchAsync(c, SIZE*sizeof(float), device, NULL);
		kernel<<<numBlocks, blockSize>>>(c, a, b);
		cudaDeviceSynchronize();
		reduction(res1, c);
	}
  t1 = get_elapsedtime();

  t2 = get_elapsedtime();
	for(j=0; j<ITER; j++)
	{
		kernel<<<numBlocks, blockSize>>>(c, a, b);
		cudaDeviceSynchronize();
	}

 	for(j=0; j<ITER; j++)
	{
		reduction(res2, c);
	}
  t3 = get_elapsedtime();

  duration = (t1 - t0);
  duration2 = (t3 - t2);
	printf("time ping-pong = %lf | time grouped = %lf\n", duration, duration2);

	return 0;
}
