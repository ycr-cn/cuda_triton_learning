

## Background

This project explores performance optimization of a simple CUDA vector addition kernel.

I implement three versions:
- Naive implementation (one thread per element)
- Strided implementation (one thread handles multiple elements)
- Vectorized implementation using float4

## Profiling Setup

- GPU: 3070-8G
- Tool: NVIDIA Nsight Compute (ncu)
- Problem size: N = 3072000
- Grid size: N / 256 * 1 * 1，256 * 1 * 1，256 * 1 * 1
- Block size: 256 * 1 * 1

## Performance Results

| Version | Time (ms) | Bandwidth (GB/s) | Memory throughput (%) |
| ------- | --------- | ---------------- | --------------------- |
| V1      | 89.63     | 411.291          | 90.93                 |
| V2      | 90.37     | 407.923          | 90.74                 |
| V3      | 90.37     | 407.923          | 90.42                 |

## Nsight Compute Analysis

All kernels are memory-bound.

Key observations:
- High DRAM utilization
- Warp stalls dominated by memory dependency
- Load/store throughput near hardware limits

## Why float4 Does Not Improve Performance

Although float4 reduces instruction count and loop overhead, it does not reduce total memory traffic.

Vector addition has extremely low arithmetic intensity:
- 2 global loads + 1 store per element
- Only 1 floating-point addition

Therefore, performance is limited by memory bandwidth.

Even with vectorized memory access:
- Total bytes transferred remain unchanged
- Memory subsystem is already saturated

Additionally:
- Possible register pressure increase
- Potential alignment constraints

As a result, float4 provides little to no performance gain.

## Key Takeaways

- Vector addition is a memory-bound problem
- Optimizations that reduce instruction count have limited impact
- Memory bandwidth is the true bottleneck
- Kernel fusion or reducing memory traffic is more effective than vectorization

## Future Work

- Kernel fusion with downstream operations
- Exploring half precision (FP16)
- Comparing with cuBLAS or Thrust implementations