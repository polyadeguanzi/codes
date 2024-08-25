#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                                                                                                                                     \
    {                                                                                                                                                                              \
        cudaError_t err = call;                                                                                                                                                    \
        if (err != cudaSuccess) {                                                                                                                                                  \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));                                                             \
            exit(EXIT_FAILURE);                                                                                                                                                    \
        }                                                                                                                                                                          \
    }

__global__ void gemm(float *M, float *N, float *P, int width) {
    // 计算线性索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < width; k++) { sum += M[idx * width + k] * N[k * 64 + threadIdx.x]; }
    for (int t = 0; t < 64; t++) { P[idx * 64 + t] = sum; }
}

int main() {
    struct timeval start1, end1, start2, end2;
    gettimeofday(&start1, NULL);

    // 主机上创建矩阵
    float *A = (float *)malloc(sizeof(float) * 1024 * 1920);
    float *B = (float *)malloc(sizeof(float) * 1920 * 64);
    float *C = (float *)malloc(sizeof(float) * 1024 * 64);
    // malloc device memory cpu->gpu
    float *d_dataA, *d_dataB, *d_dataC;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataA, sizeof(float) * 1024 * 1920));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataB, sizeof(float) * 1920 * 64));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataC, sizeof(float) * 1024 * 64));

    // set value随机初始化矩阵A和矩阵B
    for (int i = 0; i < 1024 * 1920; i++) {
        A[i] = i; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1920 * 64; i++) {
        B[i] = i; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1024 * 64; i++) {
        C[i] = 0; // Random integers between 0 and 99
    }

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataA, A, sizeof(float) * 1024 * 1920, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataB, B, sizeof(float) * 1920 * 64, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataC, C, sizeof(float) * 1024 * 64, cudaMemcpyHostToDevice));

    // 核函数配置
    dim3 threadPerBlock(64); // 每个块中的线程数
    dim3 blockNumber(16);    // 块数
    CHECK_CUDA_ERROR(cudaGetLastError());
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    gettimeofday(&start2, NULL);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //  调用核函数
    gemm<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, 1920);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 检查核函数调用的错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    gettimeofday(&end2, NULL);
    float timeuse_kernel = 1000000 * (end2.tv_sec - start2.tv_sec) + end2.tv_usec - start2.tv_usec;
    printf("Kernel execution time is %f ms\n", timeuse_kernel / 1000);

    // 拷贝计算结果到主机
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_dataC, sizeof(float) * 1024 * 64, cudaMemcpyDeviceToHost));

    // 输出结果
    printf("Result matrix first element: %f\n", C[0]);
    printf("Result matrix last element: %f\n", C[1024 * 64 - 1]);
    // 释放内存
    free(A);
    free(B);
    free(C);
    CHECK_CUDA_ERROR(cudaFree(d_dataA));
    CHECK_CUDA_ERROR(cudaFree(d_dataB));
    CHECK_CUDA_ERROR(cudaFree(d_dataC));

    gettimeofday(&end1, NULL);
    float timeuse = 1000000 * (end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec;
    printf("Total time is %f ms\n", timeuse / 1000);

    return 0;
}
