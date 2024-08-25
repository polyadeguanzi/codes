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

__global__ void concatenate(float *param_0, // Source tensor 1
                            float *param_1, // Source tensor 2
                            float *param_2,// Destination tensor
                            int c1,int c2, int c3        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i< c1;i++) { // We have 4 warps processing 64 elements each
        
        // Calculate index for source and destination
        int src_idx = idx * c1+ i;
        int dest_idx = idx * c3+ i;

        float src_val = param_0[src_idx];
        param_2[dest_idx] = src_val;
    }
    for (int j=0; j< c2;j++) { // We have 4 warps processing 64 elements each
        // Calculate index for source and destination
        int src_idx = idx * c2 + j;
        int dest_idx = idx * c3 + c1+j;
        float src_val2 = param_1[src_idx];
        // Write values to destination tensor
        param_2[dest_idx] = src_val2;
        printf("%f ",param_2[dest_idx]);

    }
}


int main() {
    struct timeval start1, end1, start2, end2;
    gettimeofday(&start1, NULL);

    // 主机上创建矩阵
    float *A = (float *)malloc(sizeof(float) * 1024 * 1728);
    float *B = (float *)malloc(sizeof(float) * 1024* 192);
    float *C = (float *)malloc(sizeof(float) * 1024 * 1920);
    // malloc device memory cpu->gpu
    float *d_dataA, *d_dataB, *d_dataC;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataA, sizeof(float) * 1024 * 1728));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataB, sizeof(float) * 1024* 192));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataC, sizeof(float) * 1024 * 1920));

    // set value随机初始化矩阵A和矩阵B
    for (int i = 0; i < 1024 * 1728; i++) {
        A[i] = i; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1024* 192; i++) {
        B[i] = i; // Random integers between 0 and 99
    }
    for (int i = 0; i < 1024 * 1920; i++) {
        C[i] =0; // Random integers between 0 and 99
    }

    // 复制数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataA, A, sizeof(float) * 1024 * 1728, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataB, B, sizeof(float) * 1024* 192, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataC, C, sizeof(float) * 1024 * 1920, cudaMemcpyHostToDevice));

    // 核函数配置
    dim3 threadPerBlock(64); // 每个块中的线程数
    dim3 blockNumber(16);     // 块数
    CHECK_CUDA_ERROR(cudaGetLastError());
    //CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    gettimeofday(&start2, NULL);
    CHECK_CUDA_ERROR(cudaGetLastError());
    //CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // 调用核函数
    concatenate<<<blockNumber, threadPerBlock>>>(d_dataA, d_dataB, d_dataC, 1728,192, 1920);

    // 检查核函数调用的错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    gettimeofday(&end2, NULL);
    float timeuse_kernel = 1000000 * (end2.tv_sec - start2.tv_sec) + end2.tv_usec - start2.tv_usec;
    printf("Kernel execution time is %f ms\n", timeuse_kernel / 1000);

    // 拷贝计算结果到主机
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_dataC, sizeof(float) * 1024 * 1920, cudaMemcpyDeviceToHost));

    // 输出结果
    printf("Result matrix first element: %f\n", C[100]);
    printf("Result matrix last element: %f\n", C[1024 * 1920 - 1]);
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
