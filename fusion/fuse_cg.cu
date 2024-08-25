// fusion for concanate and gemm
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

__global__ void concatenate_gemm(float *param_0, // Source tensor 1
                            float *param_1, // Source tensor 2
                            float *param_2, // Destination tensor
                            int c1, int c2, int c3,
                            float *N, 
                            float *P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < c1; i++) { // We have 4 warps processing 64 elements each

        // Calculate index for source and destination
        int src_idx = idx * c1 + i;
        int dest_idx = idx * c3 + i;

        float src_val = param_0[src_idx];
        param_2[dest_idx] = src_val;
    }
    for (int j = 0; j < c2; j++) { // We have 4 warps processing 64 elements each
        // Calculate index for source and destination
        int src_idx = idx * c2 + j;
        int dest_idx = idx * c3 + c1 + j;
        float src_val2 = param_1[src_idx];
        // Write values to destination tensor
        param_2[dest_idx] = src_val2;
        // printf("%f ",param_2[dest_idx]);
    }


    __syncthreads();
/*gemm*/
    float sum = 0;
    for (int k = 0; k < c3; k++) {
        //printf("%f",M[idx * width + k] * N[k * 64 + threadIdx.x]);
        sum += param_2[idx * c3 + k] * N[k * 64 + threadIdx.x];
    }
    
    for (int t = 0; t < 64; t++) { 
        P[idx * 64 + t] = sum;
        //printf("%f ",P[idx * 64 + t] );
    }
}

 
int main() {
    // struct timeval start1, end1, start2, end2;

    // 主机上创建矩阵
    float *A = (float *)malloc(sizeof(float) * 1024 * 1728);
    float *B = (float *)malloc(sizeof(float) * 1024 * 192);
    float *C = (float *)malloc(sizeof(float) * 1024 * 1920);
    float *W = (float *)malloc(sizeof(float) * 1920 * 64);
    float *R = (float *)malloc(sizeof(float) * 1024 * 64);
    // malloc device memory cpu->gpu
    float *d_dataA, *d_dataB, *d_dataC, *d_W, *d_dataD;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataA, sizeof(float) * 1024 * 1728));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataB, sizeof(float) * 1024 * 192));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataC, sizeof(float) * 1024 * 1920));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_W, sizeof(float) * 1920 * 64));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataD, sizeof(float) * 1024 * 64));

    // set value随机初始化矩阵A和矩阵B
    for (int i = 0; i < 1024 * 1728; i++) {
        A[i] = i % 10 + 1; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1024 * 192; i++) {
        B[i] = i % 10 + 1; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1024 * 1920; i++) {
        C[i] = 0; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1920 * 64; i++) {
        W[i] = i % 10 + 1; //权重矩阵不能为0
    }
    for (int i = 0; i < 1024 * 64; i++) {
        R[i] = 0; // Random integers between 0 and 99
    }

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataA, A, sizeof(float) * 1024 * 1728, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataB, B, sizeof(float) * 1024 * 192, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERROR(cudaMemcpy(d_dataC, C, sizeof(float) * 1024 * 1920, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, W, sizeof(float) * 1920 * 64, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataD, R, sizeof(float) * 1024 * 64, cudaMemcpyHostToDevice));

    // 核函数配置
    dim3 threadPerBlock(64); // 每个块中的线程数
    dim3 blockNumber(16);    // 块数

    concatenate_gemm<<<blockNumber, threadPerBlock,0,0>>>(d_dataA, d_dataB, d_dataC, 1728, 192, 1920,d_W,d_dataD);

    // 检查核函数调用的错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(R, d_dataD, sizeof(float) * 1024 * 64, cudaMemcpyDeviceToHost))
    // 输出结果
    printf("Result matrix first element: %f\n", R[100]);
    printf("Result matrix last element: %f\n", R[1024 * 64 - 1]);
    // 释放内存
    free(A);
    free(B);
    free(C);
    free(W);
    free(R);
    CHECK_CUDA_ERROR(cudaFree(d_dataA));
    CHECK_CUDA_ERROR(cudaFree(d_dataB));
    CHECK_CUDA_ERROR(cudaFree(d_dataC));
    CHECK_CUDA_ERROR(cudaFree(d_W));
    CHECK_CUDA_ERROR(cudaFree(d_dataD));

    return 0;
}
