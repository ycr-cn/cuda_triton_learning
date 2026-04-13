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

//[Deprecated] unsafe: assumes float4 alignment
// template <int BLOCK_SIZE>
// __global__ void softmax_row_kernel(
//     float* output_ptr,
//     const float* input_ptr,
//     int input_row_stride,
//     int output_row_stride,
//     int n_rows,
//     int n_cols
// ) {
//     int row = blockIdx.x;
//     if (row >= n_rows) return;

//     const float* row_in = input_ptr + (long long)row * input_row_stride;
//     float* row_out = output_ptr + (long long)row * output_row_stride;

//     constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
//     __shared__ float smem_max[NUM_WARPS];
//     __shared__ float smem_sum[NUM_WARPS];
//     __shared__ float row_max_shared;
//     __shared__ float row_sum_shared;

//     int tid = threadIdx.x;
//     //x % n == x & (n - 1) if n equals 2^i
//     int lane = tid & (WARP_SIZE - 1);
//     int warp_id = tid / WARP_SIZE;

//     //row max
//     float local_max = -FLT_MAX;

//     int n_cols4 = n_cols / 4;   
//     int rem = n_cols % 4;      

//     const float4* row_in4 = reinterpret_cast<const float4*>(row_in);

//     for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
//         float4 v = row_in4[i];
//         local_max = fmaxf(local_max, v.x);
//         local_max = fmaxf(local_max, v.y);
//         local_max = fmaxf(local_max, v.z);
//         local_max = fmaxf(local_max, v.w);
//     }

//     int tail_base = n_cols4 * 4;
//     if (tid < rem) {
//         local_max = fmaxf(local_max, row_in[tail_base + tid]);
//     }

//     float warp_max = warp_reduce_max(local_max);
//     if (lane == 0) {
//         smem_max[warp_id] = warp_max;
//     }
//     __syncthreads();

//     float block_max = -FLT_MAX;
//     if (warp_id == 0) {
//         block_max = (tid < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
//         block_max = warp_reduce_max(block_max);
//         if (tid == 0) {
//             row_max_shared = block_max;
//         }
//     }
//     __syncthreads();

//     float row_max = row_max_shared;


//     //sum(exp(x - row_max))
//     float local_sum = 0.0f;

//     for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
//         float4 v = row_in4[i];
//         local_sum += __expf(v.x - row_max);
//         local_sum += __expf(v.y - row_max);
//         local_sum += __expf(v.z - row_max);
//         local_sum += __expf(v.w - row_max);
//     }

//     if (tid < rem) {
//         local_sum += __expf(row_in[tail_base + tid] - row_max);
//     }

//     float warp_sum = warp_reduce_sum(local_sum);
//     if (lane == 0) {
//         smem_sum[warp_id] = warp_sum;
//     }
//     __syncthreads();

//     float block_sum = 0.0f;
//     if (warp_id == 0) {
//         block_sum = (tid < NUM_WARPS) ? smem_sum[lane] : 0.0f;
//         block_sum = warp_reduce_sum(block_sum);
//         if (tid == 0) {
//             row_sum_shared = block_sum;
//         }
//     }
//     __syncthreads();

//     float row_sum = row_sum_shared;
//     float inv_sum = 1.0f / row_sum;


//     //write back
//     float4* row_out4 = reinterpret_cast<float4*>(row_out);

//     for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
//         float4 v = row_in4[i];
//         float4 o;
//         o.x = __expf(v.x - row_max) * inv_sum;
//         o.y = __expf(v.y - row_max) * inv_sum;
//         o.z = __expf(v.z - row_max) * inv_sum;
//         o.w = __expf(v.w - row_max) * inv_sum;
//         row_out4[i] = o;
//     }

//     if (tid < rem) {
//         float x = row_in[tail_base + tid];
//         row_out[tail_base + tid] = __expf(x - row_max) * inv_sum;
//     }
// }


template <int BLOCK_SIZE>
__global__ void softmax_row_kernel_scalar(
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
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    //reduce max
    float local_max = -FLT_MAX;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) {
        smem_max[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (tid < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            row_max_shared = block_max;
        }
    }
    __syncthreads();

    float row_max = row_max_shared;

    //reduce sum(exp(x - max))
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        local_sum += __expf(row_in[i] - row_max);
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (tid < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            row_sum_shared = block_sum;
        }
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum_shared;

    //write back
    for (int i = tid; i < n_cols; i += BLOCK_SIZE) {
        row_out[i] = __expf(row_in[i] - row_max) * inv_sum;
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_row_kernel_vec4(
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

    const float4* row_in4 = reinterpret_cast<const float4*>(row_in);
    float4* row_out4 = reinterpret_cast<float4*>(row_out);

    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem_max[NUM_WARPS];
    __shared__ float smem_sum[NUM_WARPS];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    int n_cols4 = n_cols / 4;
    int rem = n_cols % 4;
    int tail_base = n_cols4 * 4;

    //reduce max
    float local_max = -FLT_MAX;

    for (int i = tid; i < n_cols4; i += BLOCK_SIZE) {
        float4 v = row_in4[i];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }

    if (tid < rem) {
        local_max = fmaxf(local_max, row_in[tail_base + tid]);
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) {
        smem_max[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (tid < NUM_WARPS) ? smem_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            row_max_shared = block_max;
        }
    }
    __syncthreads();

    float row_max = row_max_shared;

    //reduce sum(exp(x - max))
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

    if (warp_id == 0) {
        float block_sum = (tid < NUM_WARPS) ? smem_sum[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            row_sum_shared = block_sum;
        }
    }
    __syncthreads();

    float inv_sum = 1.0f / row_sum_shared;

    //write back
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

static inline bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0;
}

template <int BLOCK_SIZE>
void launch_softmax_row_kernel(
    float* output_ptr,
    const float* input_ptr,
    int input_row_stride,
    int output_row_stride,
    int n_rows,
    int n_cols,
    cudaStream_t stream = 0
) {
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    //vec4使用前提
    bool can_use_vec4 =
        is_aligned_16(input_ptr) &&
        is_aligned_16(output_ptr) &&
        (input_row_stride % 4 == 0) &&
        (output_row_stride % 4 == 0);

    if (can_use_vec4) {
        softmax_row_kernel_vec4<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            output_ptr,
            input_ptr,
            input_row_stride,
            output_row_stride,
            n_rows,
            n_cols
        );
    } else {
        softmax_row_kernel_scalar<BLOCK_SIZE><<<grid, block, 0, stream>>>(
            output_ptr,
            input_ptr,
            input_row_stride,
            output_row_stride,
            n_rows,
            n_cols
        );
    }
}

int main() {
    constexpr int BLOCK_SIZE = 256;

    int n_rows = 1024;
    int n_cols = 1003;              
    int input_row_stride = 1004;    
    int output_row_stride = 1004;

    size_t in_numel = (size_t)n_rows * input_row_stride;
    size_t out_numel = (size_t)n_rows * output_row_stride;
    size_t in_bytes = in_numel * sizeof(float);
    size_t out_bytes = out_numel * sizeof(float);

    std::vector<float> h_in(in_numel, 0.0f);
    std::vector<float> h_out_cpu((size_t)n_rows * n_cols, 0.0f);
    std::vector<float> h_out_scalar(out_numel, 0.0f);
    std::vector<float> h_out_vec4(out_numel, 0.0f);
    std::vector<float> h_out_launch(out_numel, 0.0f);

    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            h_in[(size_t)r * input_row_stride + c] =
                (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f;
        }
    }

    {
        std::vector<float> h_in_compact((size_t)n_rows * n_cols);
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                h_in_compact[(size_t)r * n_cols + c] = h_in[(size_t)r * input_row_stride + c];
            }
        }
        cpu_softmax_rowwise(h_out_cpu, h_in_compact, n_rows, n_cols);
    }

    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice));

    auto check_result = [&](const std::vector<float>& out, const char* name) {
        float max_abs_err = 0.0f;
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                float got = out[(size_t)r * output_row_stride + c];
                float ref = h_out_cpu[(size_t)r * n_cols + c];
                max_abs_err = std::max(max_abs_err, std::fabs(got - ref));
            }
        }
        std::cout << "[" << name << "] max abs error = " << max_abs_err << "\n";
    };

    auto bench = [&](auto launch_fn, std::vector<float>& h_out, const char* name) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaMemset(d_out, 0, out_bytes));
        CHECK_CUDA(cudaEventRecord(start));

        launch_fn();

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

        std::cout << "[" << name << "] time = " << ms << " ms\n";
        check_result(h_out, name);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    };

    //scalar
    bench([&]() {
        std::cout << "Launching scalar kernel directly...\n";
        softmax_row_kernel_scalar<BLOCK_SIZE><<<n_rows, BLOCK_SIZE>>>(
            d_out, d_in,
            input_row_stride, output_row_stride,
            n_rows, n_cols
        );
    }, h_out_scalar, "scalar-direct");

    //vec4
    bool can_use_vec4 =
        ((reinterpret_cast<std::uintptr_t>(d_in) & 0xF) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(d_out) & 0xF) == 0) &&
        (input_row_stride % 4 == 0) &&
        (output_row_stride % 4 == 0);

    if (can_use_vec4) {
        bench([&]() {
            std::cout << "Launching vec4 kernel directly...\n";
            softmax_row_kernel_vec4<BLOCK_SIZE><<<n_rows, BLOCK_SIZE>>>(
                d_out, d_in,
                input_row_stride, output_row_stride,
                n_rows, n_cols
            );
        }, h_out_vec4, "vec4-direct");
    } else {
        std::cout << "[vec4-direct] skipped\n";
    }

    //launcher, auto distribution
    bench([&]() {
        std::cout << "Launching through launcher...\n";
        launch_softmax_row_kernel<BLOCK_SIZE>(
            d_out, d_in,
            input_row_stride, output_row_stride,
            n_rows, n_cols
        );
    }, h_out_launch, "launcher");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}