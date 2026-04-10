#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)



static void cpu_softmax_rowwise(
    std::vector<float>& out,
    const std::vector<float>& in,
    int n_rows,
    int n_cols
) {
    for (int r = 0; r < n_rows; ++r) {
        float mx = -FLT_MAX;
        for (int c = 0; c < n_cols; ++c) {
            mx = std::max(mx, in[r * n_cols + c]);
        }

        float sum = 0.0f;
        for (int c = 0; c < n_cols; ++c) {
            sum += std::exp(in[r * n_cols + c] - mx);
        }

        for (int c = 0; c < n_cols; ++c) {
            out[r * n_cols + c] = std::exp(in[r * n_cols + c] - mx) / sum;
        }
    }
}


static __device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

static __device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void softmax_row_kernel(
    float* output_ptr,
    const float* input_ptr,
    int input_row_stride,
    int output_row_stride,
    int n_rows,
    int n_cols
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* row_in = input_ptr + (long long)row * input_row_stride;
    float* row_out = output_ptr + (long long)row * output_row_stride;

    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    int tid = threadIdx.x;
    //x % n == x & (n - 1) if n equals 2^i
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    //row max
    float local_max = -FLT_MAX;

    int n_cols4 = n_cols / 4;   
    int rem = n_cols % 4;      

    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

    for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
        float4 v = row_in4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    int tail_base = n_cols4 * 4;
    if (tid < rem) {
        local_max = fmaxf(local_max, row_in[tail_base + tid]);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) {
        smem_max[warp_id] = warp_max;
    }
    __syncthreads();

    float block_max = -FLT_MAX;
    if (warp_id == 0) {
        block_max = (tid < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            row_max_shared = block_max;
        }
    }
    __syncthreads();

    float row_max = row_max_shared;


    //sum(exp(x - row_max))
    float local_sum = 0.0f;

    for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
        float4 v = row_in4[i];
        local_sum += __expf(v.x - row_max);
        local_sum += __expf(v.y - row_max);
        local_sum += __expf(v.z - row_max);
        local_sum += __expf(v.w - row_max);
    }

    if (tid < rem) {
        local_sum += __expf(row_in[tail_base + tid] - row_max);
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (tid < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            row_sum_shared = block_sum;
        }
    }
    __syncthreads();

    float row_sum = row_sum_shared;
    float inv_sum = 1.0f / row_sum;


    //write back
    float4* row_out4 = reinterpret_cast<float4*>(row_out);

    for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
        float4 v = row_in4[i];
        float4 o;
        o.x = __expf(v.x - row_max) * inv_sum;
        o.y = __expf(v.y - row_max) * inv_sum;
        o.z = __expf(v.z - row_max) * inv_sum;
        o.w = __expf(v.w - row_max) * inv_sum;
        row_out4[i] = o;
    }

    if (tid < rem) {
        float x = row_in[tail_base + tid];
        row_out[tail_base + tid] = __expf(x - row_max) * inv_sum;
    }
}
int main() {
    constexpr int BLOCK_SIZE = 128;
    
    const int n_rows = 4096;
    const int n_cols = 1024;  
    
    const int input_row_stride = n_cols;
    const int output_row_stride = n_cols;
    
    size_t numel = (size_t)n_rows * n_cols;
    
    std::vector<float> h_input(numel);
    for (size_t i = 0; i < numel; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
    
    std::vector<float> h_output(numel, 0.0f);
    std::vector<float> h_ref(numel, 0.0f);
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    
    CHECK_CUDA(cudaMalloc(&d_input, numel * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, numel * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(
        d_input, h_input.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    
    softmax_row_kernel<BLOCK_SIZE><<<grid, block>>>(
        d_output, d_input,
        input_row_stride,
        output_row_stride,
        n_rows, n_cols
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < 50; ++i) {
        softmax_row_kernel<BLOCK_SIZE><<<grid, block>>>(
            d_output, d_input,
            input_row_stride,
            output_row_stride,
            n_rows, n_cols
        );
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    std::cout << "Kernel time (50 iters): " << ms << " ms\n";
    std::cout << "Avg per iter: " << ms / 50 << " ms\n";
    
    CHECK_CUDA(cudaMemcpy(
        h_output.data(), d_output, numel * sizeof(float), cudaMemcpyDeviceToHost));
    
    cpu_softmax_rowwise(h_ref, h_input, n_rows, n_cols);
    
    float max_err = 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        max_err = std::max(max_err, std::fabs(h_output[i] - h_ref[i]));
    }
    
    std::cout << "max abs error = " << max_err << "\n";
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    return 0;
}
    