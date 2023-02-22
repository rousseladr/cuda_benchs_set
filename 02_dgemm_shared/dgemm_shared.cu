/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_WIDTH 32
#define TILE_WIDTH 32
#define TAILLE 4096


__global__
	void MulMatrixKernel(float* A, float* B, float* C, int N)
	{
		int col		= threadIdx.x + blockDim.x * blockIdx.x;
		int ligne	= threadIdx.y + blockDim.y * blockIdx.y;

		if((col < N) && (ligne < N))
    {
			float val = 0.0f;
			for(int k = 0; k < N; k++)
      {
				val += A[ligne * N + k] * B[k * N + col];
			}
			C[ligne * N + col] = val;
		}
	}

__global__
	void MulMatrixShare(float* A, float* B, float* C, int N){
		__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
		__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

		int ligne	= blockIdx.y * BLOCK_WIDTH + threadIdx.y;
		int col	  = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

		float value = 0.0f;

		for(int id_tile = 0; id_tile < gridDim.x; id_tile++)
    {
      int i = id_tile * TILE_WIDTH + threadIdx.x;
      int j = id_tile * TILE_WIDTH + threadIdx.y;

			s_A[threadIdx.y][threadIdx.x] = A[ligne * N + j]; // charger un élément de A [un élément par thread]
			s_B[threadIdx.y][threadIdx.x] = B[i * N + col]; // charger un élément de B [un élément par thread]

      // Attente que tous les threads ont bien chargé dans la mémoire partagée leurs deux indices
			__syncthreads();

			for(int k =0; k < TILE_WIDTH; k++)
      {
				value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
      }

      // S'assurer que tous les threads ont bien fini le calcul du préliminaire du tile courant avant de commencer la prochaine étape du calcul de cette tile
			__syncthreads();
		}

    // Enregistrer la valeur accumulée dans C (mémoire globale)
		C[ligne * N + col] = value;
}

int main(int argc, char** argv)
{
	int N = (argc >= 2)?(atoi(argv[1])):TAILLE;
	int nbBlocks = N / BLOCK_WIDTH;
	//if(N % BLOCK_WIDTH) nbBlocks++;
	if(N % BLOCK_WIDTH) N += (N % BLOCK_WIDTH);
	dim3 gridSize(nbBlocks, nbBlocks);
	dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);

	float *A, *B, *C;
	float *d_A, *d_B, *d_C;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	A = (float*) malloc(sizeof(float) * N * N);
	B = (float*) malloc(sizeof(float) * N * N);
	C = (float*) malloc(sizeof(float) * N * N);

	cudaMalloc(&d_A, sizeof(float) * N * N);
	cudaMalloc(&d_B, sizeof(float) * N * N);
	cudaMalloc(&d_C, sizeof(float) * N * N);

	srand(2019);

	for(int i = 0; i < N * N; i++)
  {
		A[i] = rand();
		B[i] = rand();
		C[i] = 0.0f;
	}

	cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(float) * N * N, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	//MulMatrixKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
	MulMatrixShare<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
	cudaEventRecord(stop);

	cudaMemcpy(C, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Matrice %dx%d\n\tTemps: %f s\n", N, N, milliseconds/1000);
	//printf("%f", milliseconds/1000);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(A);
	free(B);
	free(C);

	return 0;
}
