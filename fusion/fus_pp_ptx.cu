__global__ void concatenate(const float *param_0, // Source tensor 1
                            const float *param_1, // Source tensor 2
                            float *param_2        // Destination tensor
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 64; // Assuming each warp processes 64 elements
    int offset = tid % 64;

    if (warp_id < 4) { // We have 4 warps processing 64 elements each
        // Calculate index for source and destination
        int src_idx = warp_id * 192 + offset;
        int dest_idx = warp_id * 64 + offset;

        // Load values from source tensors
        float src_val = param_0[src_idx];
        float src_val_2 = param_1[src_idx];

        // Write values to destination tensor
        param_2[dest_idx] = src_val;
        param_2[dest_idx + 64] = src_val_2;
    }
}

__global__ void fusion_1(const float *param_0, // Source tensor
                         float *param_1        // Destination tensor
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 64; // Assuming each warp processes 64 elements
    int offset = tid % 64;

    if (warp_id < 2) { // We have 2 warps processing 64 elements each
        // Calculate index
        int idx = warp_id * 64 + offset;

        // Load value from source tensor
        float src_val = param_0[idx];

        // Add value from the destination tensor
        float dest_val = param_1[idx];
        float result = src_val + dest_val;

        // Write result to destination tensor
        param_1[idx] = fmaxf(result, 0.0f); // Clamp to non-negative value
    }
}

__global__ void fusion(const float *param_0, // Source tensor 1
                       const float *param_1, // Source tensor 2
                       const float *param_2, // Source tensor 3
                       float *param_3        // Destination tensor
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 64; // Assuming each warp processes 64 elements
    int offset = tid % 64;

    if (warp_id < 2) { // We have 2 warps processing 64 elements each
        // Calculate index
        int idx = warp_id * 64 + offset;

        // Load values from source tensors
        float val0 = param_0[idx];
        float val1 = param_1[idx];
        float val2 = param_2[idx];

        // Compute result
        float result = val0 + val1 + val2;
        result = fminf(fmaxf(result, 0.0f), 1.0f); // Sigmoid  Clamp to [0, 1]

        // Write result to destination tensor
        param_3[idx] = result;
    }
}
