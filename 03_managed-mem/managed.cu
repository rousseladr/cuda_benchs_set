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

__global__ void kernel(uint64_t* y, uint64_t* a, uint64_t* b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	y[i] = a[i] * b[i];
}

__host__ void reduction(uint64_t *resultat, uint64_t * a, uint64_t N)
{

	int i;
  *resultat = (uint64_t)0;
	for(i=0; i< N; i++)
  {
    *resultat += a[i];
  }
}

int main(int argc, char *argv[])
{

  bool verbose = false;
  int cpu = -1;
  int s, j;
  uint64_t size_in_mbytes = 100;
  int niter = 1000;

  int opt;
  while ((opt = getopt(argc, argv, "s:i:vh")) != -1)
  {
    switch (opt)
    {
      case 's':
        size_in_mbytes = (uint64_t)atoi(optarg);
        break;
      case 'i':
        niter = (int)atoi(optarg);
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

  double size_in_bytes = size_in_mbytes * 1E6;
  uint64_t N = (size_in_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);

  // Setup phase
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();

  int numcores = sysconf(_SC_NPROCESSORS_ONLN) / 2; // divided by 2 because of hyperthreading
  int numanodes = numa_num_configured_nodes();

  int gpucount = -1;
  cudaGetDeviceCount(&gpucount);

  int *tgpu = (int*)malloc(sizeof(int) * numcores * gpucount);
  double *pingpong = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *batch = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *pasync = (double*)malloc(sizeof(double) * numcores * gpucount);
  double *basync = (double*)malloc(sizeof(double) * numcores * gpucount);
  memset(tgpu, -1, sizeof(int) * numcores * gpucount);
  memset(pingpong, 0, sizeof(int) * numcores * gpucount);
  memset(batch, 0, sizeof(int) * numcores * gpucount);
  memset(pasync, 0, sizeof(int) * numcores * gpucount);
  memset(basync, 0, sizeof(int) * numcores * gpucount);

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

	    uint64_t* a; uint64_t* b; uint64_t* c;

	    int j;
	    int i;
      uint64_t res1 = 0x0, res2 = 0x0, res3 = 0x0, res4 = 0x0;
      double t0 = 0., t1 = 0., duration = 0.;
      double t2 = 0., t3 = 0., duration2 = 0.;
      double t4 = 0., t5 = 0., duration3 = 0.;
      double t6 = 0., t7 = 0., duration4 = 0.;

	    cudaMallocManaged(&a, N * sizeof(uint64_t));
	    cudaMallocManaged(&b, N * sizeof(uint64_t));
	    cudaMallocManaged(&c, N * sizeof(uint64_t));

	    uint64_t* t3_ha;
	    uint64_t* t3_hb;
	    uint64_t* t3_hc;

	    uint64_t* t3_da;
	    uint64_t* t3_db;
	    uint64_t* t3_dc;

      cudaMallocHost(&t3_ha, N * sizeof(uint64_t));
      cudaMallocHost(&t3_hb, N * sizeof(uint64_t));
      cudaMallocHost(&t3_hc, N * sizeof(uint64_t));

      cudaMalloc(&t3_da, N * sizeof(uint64_t));
      cudaMalloc(&t3_db, N * sizeof(uint64_t));
      cudaMalloc(&t3_dc, N * sizeof(uint64_t));

      cudaStream_t stream1;
      cudaStreamCreate(&stream1);

	    for(i=0; i < N; i++)
      {
        a[i] = t3_ha[i] = i;
        b[i] = t3_hb[i] = 2;
        c[i] = t3_hc[i] = 0;
      }

      // Prefetch the data to the CPU
      int device = -1;
      cudaGetDevice(&device);
      cudaMemPrefetchAsync(a, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);
      cudaMemPrefetchAsync(b, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);
      cudaMemPrefetchAsync(c, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);

      int blockSize = 256;
      int numBlocks = (N + blockSize - 1) / blockSize;

      cudaDeviceSynchronize();

      // TEST 1
      // Ping-Pong at each test iteration CPU <-> GPU with CUDA Managed Memory
      t0 = get_elapsedtime();
	    for(j=0; j < niter; j++)
	    {
	    	kernel<<<numBlocks, blockSize>>>(c, a, b);
	    	reduction(&res1, c, N);
	    }
      t1 = get_elapsedtime();

      cudaMemPrefetchAsync(a, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);
      cudaMemPrefetchAsync(b, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);
      cudaMemPrefetchAsync(c, N * sizeof(uint64_t), cudaCpuDeviceId, NULL);
      cudaDeviceSynchronize();

      // TEST 2
      // Batch Reduction all compute on GPU THEN reduction on GPU with CUDA Managed Memory
      t2 = get_elapsedtime();
	    for(j=0; j < niter; j++)
	    {
	    	kernel<<<numBlocks, blockSize>>>(c, a, b);
	    }

 	    for(j=0; j < niter; j++)
	    {
	    	reduction(&res2, c, N);
	    }
      t3 = get_elapsedtime();

      cudaDeviceSynchronize();

      // TEST 3
      // Ping-Pong at each test iteration CPU <-> GPU with Explicit GPU allocation (cudaMalloc) + Asynchronous cuda Memory copies
      t4 = get_elapsedtime();

      cudaMemcpyAsync(t3_da, t3_ha, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(t3_db, t3_hb, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	    for(j=0; j < niter; j++)
	    {
        cudaMemcpyAsync(t3_dc, t3_hc, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	    	kernel<<<numBlocks, blockSize>>>(t3_dc, t3_da, t3_db);

        cudaMemcpyAsync(t3_hc, t3_dc, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);

	    	reduction(&res3, t3_hc, N);
	    }

      t5 = get_elapsedtime();

      cudaDeviceSynchronize();

      // TEST 4
      // Batch Reduction all compute on GPU THEN reduction on GPU with Explicit GPU allocation (cudaMalloc) + Asynchronous cuda Memory copies
      t6 = get_elapsedtime();

      cudaMemcpyAsync(t3_da, t3_ha, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(t3_db, t3_hb, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
      cudaMemcpyAsync(t3_dc, t3_hc, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	    for(j=0; j < niter; j++)
	    {
	    	kernel<<<numBlocks, blockSize>>>(t3_dc, t3_da, t3_db);
	    }

      cudaMemcpyAsync(t3_hc, t3_dc, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      cudaStreamSynchronize(0);

 	    for(j=0; j < niter; j++)
	    {
	    	reduction(&res4, t3_hc, N);
	    }

      t7 = get_elapsedtime();

      cudaDeviceSynchronize();

      if((res1 != res2) && (res2 != res3) && (res3 != res4))
      {
        fprintf(stderr, "Fatal Error...\n");

        cudaFree(a);
        cudaFree(b);
        cudaFree(c);

        cudaFree(t3_da);
        cudaFree(t3_db);
        cudaFree(t3_dc);

        cudaFreeHost(t3_ha);
        cudaFreeHost(t3_hb);
        cudaFreeHost(t3_hc);

        free(tgpu);
        free(pingpong);
        free(batch);
        free(pasync);
        free(basync);

        exit(EXIT_FAILURE);
      }

      duration  = (t1 - t0);
      duration2 = (t3 - t2);
      duration3 = (t5 - t4);
      duration4 = (t7 - t6);

      if(verbose)
      {
	      printf("time ping-pong = %lf | time batch = %lf | time ping-pong memcpyAsync = %lf | time Batch memcpyAsync = %lf\n", duration, duration2, duration3, duration4);
      }

      pingpong[coreId * gpucount + deviceId] = duration;
      batch[coreId * gpucount + deviceId] = duration2;
      pasync[coreId * gpucount + deviceId] = duration3;
      basync[coreId * gpucount + deviceId] = duration4;

      cudaFree(a);
      cudaFree(b);
      cudaFree(c);

      cudaFree(t3_da);
      cudaFree(t3_db);
      cudaFree(t3_dc);

      cudaFreeHost(t3_ha);
      cudaFreeHost(t3_hb);
      cudaFreeHost(t3_hc);
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

  fprintf(outputFile, "core\tgpu\tPing-Pong\tBatch\tMemcpyAsync-PingPong\tMemcpyAsync-Batch\n");
  for(int i = 0; i < numcores; ++i)
  {
    for(int d = 0; d < gpucount; ++d)
    {
      fprintf(outputFile, "%d\t%d\t%lf\t%lf\t%lf\t%lf\n", i, tgpu[i * gpucount + d], pingpong[i * gpucount + d], batch[i * gpucount + d], pasync[i * gpucount + d], basync[i * gpucount +d]);
    }
  }

  fclose(outputFile);
  free(tgpu);
  free(pingpong);
  free(batch);
  free(pasync);
  free(basync);

	return 0;
}
