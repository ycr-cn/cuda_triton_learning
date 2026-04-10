#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define GRID_SIZE 256

// __global__ void vector_add(const float* x, const float* y, float* out, const int N){
//     int stride = gridDim.x * blockDim.x * 4;
//     int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

//     for(int i = idx; i < N; i += stride){
//         if(i + 3 < N){ 
//             float4 a = *reinterpret_cast<const float4*>(&x[i]);
//             float4 b = *reinterpret_cast<const float4*>(&y[i]);
//             float4 c = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
//             *reinterpret_cast<float4*>(&out[i]) = c;
//         }
//         else{
//             while(i < N){
//                 out[i] = x[i] + y[i];
//                 ++i;
//             }
//         }
//     }
// }

__global__ void vector_add(const float* x, const float* y, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* y4 = reinterpret_cast<const float4*>(y);
    float4* out4 = reinterpret_cast<float4*>(out);

    int N4 = N / 4;

    for (int i = tid; i < N4; i += stride) {
        float4 a = x4[i];
        float4 b = y4[i];
        out4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    int tail = N % 4;
    int tail_start = N4 * 4;

    if (tid < tail) {
        int i = tail_start + tid;
        out[i] = x[i] + y[i];
    }
}

int main(){
    int N = 3072000;
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_out = new float[N];

    for(int i = 0; i < N; ++i){
        h_x[i] = i * 1.0f;
        h_y[i] = i * 2.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    vector_add<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_out, N);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 128; i < 132; ++i){
        std::cout << h_x[i] << " + " << h_y[i] << " = " << h_out[i] << std::endl;
    }

    delete[] h_x;
    delete[] h_y;
    delete[] h_out;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    return 0;
}