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

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
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
        C[i] =0; // Random integers between 0 and 99
    }

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataA, A, sizeof(float) * 1024 * 1920, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataB, B, sizeof(float) * 1920 * 64, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataC, C, sizeof(float) * 1024 * 64, cudaMemcpyHostToDevice));


    CHECK_CUDA_ERROR(cudaGetLastError());
    //CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    gettimeofday(&start2, NULL);

    // 调用核函数
    const int BM = 32, BN = 32;
    const int M = 512, N = 512, K = 512;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, 1920);

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
    printf("Result matrix last element: %f\n", C[1024 * 64 - 3]);

    // for (int i = 0; i < 1024; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("%6.2f ", C[i*64+j]);
    //     }
    //     printf("\n");
    // }


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
