#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void vector_add(const float* x, const float* y, float* out, const int N){
    int offsets = blockIdx.x * blockDim.x + threadIdx.x;
    if(offsets < N){
        out[offsets] = x[offsets] + y[offsets];
    }
}


int main() {
    int N = 3072000;
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_out = new float[N];


    for (int i = 0; i < N; i++) {
        h_x[i] = i * 1.0f;
        h_y[i] = i * 2.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<gridSize, BLOCK_SIZE>>>(d_x, d_y, d_out, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
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