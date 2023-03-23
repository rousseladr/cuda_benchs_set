/*
 Copyright 2023 Adrien Roussel <adrien.roussel@protonmail.com>
 SPDX-License-Identifier: CECILL-C
*/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>

#include <cuda.h>

#include "../common/common.h"

#define ITER 1000

__global__ void kernel(int* y, int* a, int* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	y[i] = a[i] * b[i];
}

__host__ void reduction(uint64_t resultat, int * a, uint64_t N)
{

	int i;
  resultat = (uint64_t)0;
	for(i=0; i< N; i++)
  {
    resultat += a[i];
  }
}

int main(int argc, char *argv[])
{

  bool verbose = false;
  int cpu = -1;
  int s, j;
  uint64_t size_in_mbytes = 100;
  double size_in_kbytes = size_in_mbytes*1000;
  double size_in_bytes = size_in_kbytes*1000;
  uint64_t N = (size_in_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

  int opt;
  while ((opt = getopt(argc, argv, "vhs:")) != -1)
  {
    switch (opt)
    {
      case 's':
        size_in_mbytes = (uint64_t)atoi(optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'h':
        goto usage;
        break;
      default:
        goto usage;
    }
  }

  if (optind != argc)
  {
usage:
    fprintf(stdout, "CUDA Bench - Explicit Memory Transfers Throughput evaluation with NUMA consideration 1.0.0\n");
    fprintf(stdout, "usage: numa_explicit.exe\n\t[-s size in MB]\n\t[-h print this help]\n");
    fprintf(stdout, "\nPlot results using python3:\n");
    fprintf(stdout, "numa_explicit.exe -s <arg> && python3 plot.py <arg>\n");
    exit(EXIT_SUCCESS);
  }

  // Setup phase
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int numcores = sysconf(_SC_NPROCESSORS_ONLN) / 2; // divided by 2 because of hyperthreading
  int numanodes = numa_num_configured_nodes();

  int gpucount = -1;
  cudaGetDeviceCount(&gpucount);

  int *tgpu = (int*)malloc(sizeof(int) * numcores * gpucount);
  double *dummy = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *grouped = (double*)malloc(sizeof(double) * numcores * gpucount);
  memset(tgpu, -1, sizeof(int) * numcores * gpucount);
  memset(dummy, 0, sizeof(int) * numcores * gpucount);
  memset(grouped, 0, sizeof(int) * numcores * gpucount);

  int coreId = 0;

  while( coreId < numcores)
  {

    if(coreId < 0 || coreId >= numcores)
    {
      fprintf(stdout, "FATAL ERROR! Invalid core id\n");
      exit(EXIT_FAILURE);
    }

    if(verbose)
    {
      fprintf(stdout, "Target core %d\n", coreId);
    }
    /* Set affinity mask to include CPUs coreId */

    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_setaffinity_np");

    /* Check the actual affinity mask assigned to the thread */

    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0)
      handle_error_en(s, "pthread_getaffinity_np");

    for (j = 0; j < CPU_SETSIZE; j++)
    {
      if (CPU_ISSET(j, &cpuset))
      {
        cpu = j;
        break;
      }
    }

    if(j == CPU_SETSIZE)
    {
      fprintf(stdout, "FATAL ERROR! Don't know on which core the thread is placed\n");
      exit(EXIT_FAILURE);
    }

    int cur_numanode = numa_node_of_cpu(cpu);
    if(verbose)
    {
      fprintf(stdout, "Running on CPU %d of %d\n", cpu, numcores);
      fprintf(stdout, "Running on NUMA %d of %d\n", cur_numanode, numanodes);
    }

    for(int deviceId = 0; deviceId < gpucount; ++deviceId)
    {
      cudaSetDevice(deviceId);

      if(verbose)
      {
        fprintf(stdout, "Set Device to %d\n", deviceId);
      }
      tgpu[coreId * gpucount + deviceId] = deviceId;

	    int* a; int* b; int* c;

	    int j;
	    int i;
      uint64_t res1, res2;
      double t0 = 0., t1 = 0., duration = 0.;
      double t2 = 0., t3 = 0., duration2 = 0.;

	    cudaMallocManaged(&a, N * sizeof(uint64_t));
	    cudaMallocManaged(&b, N * sizeof(uint64_t));
	    cudaMallocManaged(&c, N * sizeof(uint64_t));

	    for(i=0; i < N; i++)
      {
        a[i] = i;
        b[i] = 2;
        c[i] = 0;
      }

      // Prefetch the data to the GPU
      int device = -1;
      cudaGetDevice(&device);
      cudaMemPrefetchAsync(a, N * sizeof(uint64_t), device, NULL);
      cudaMemPrefetchAsync(b, N * sizeof(uint64_t), device, NULL);

      int blockSize = 256;
      int numBlocks = (N + blockSize - 1) / blockSize;


      t0 = get_elapsedtime();
	    for(j=0; j<ITER; j++)
	    {
      	cudaMemPrefetchAsync(c, N * sizeof(uint64_t), device, NULL);
	    	kernel<<<numBlocks, blockSize>>>(c, a, b);
	    	cudaDeviceSynchronize();
	    	reduction(res1, c, N);
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
	    	reduction(res2, c, N);
	    }
      t3 = get_elapsedtime();

      duration = (t1 - t0);
      duration2 = (t3 - t2);
      if(verbose)
      {
	      printf("time ping-pong = %lf | time grouped = %lf\n", duration, duration2);
      }

      dummy[coreId * gpucount + deviceId] = duration;
      grouped[coreId * gpucount + deviceId] = duration2;

      cudaFree(a);
      cudaFree(b);
      cudaFree(c);
    }
    coreId += 1;
  }

  char buff_explicit_time[100];
  snprintf(buff_explicit_time, 100, "%lu-MB_managed-bench_time.csv", size_in_mbytes);
  FILE * outputFile;
  outputFile = fopen( buff_explicit_time, "w+" );
  if (outputFile == NULL)
  {
    printf( "Cannot open file %s\n", buff_explicit_time );
    exit(EXIT_FAILURE);
  }

  fprintf(outputFile, "core\tgpu\tPing-Pong\tGrouped\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\n", i, tgpu[i * gpucount + d], dummy[i * gpucount + d], grouped[i * gpucount + d]);
    }
  }

  fclose(outputFile);
  free(tgpu);
  free(dummy);
  free(grouped);

	return 0;
}
