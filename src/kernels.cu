#include <cstdio>
#include <cstring>
#include <cstdlib>

// divergent: lanes 16-31 do work, 0-15 masked off (half the warp diverges)
__global__ void divergent(float *out, const float *in, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    if (tid < n) {
        float v = in[tid];
        if (lane & 0xF0F0F0F0) {        // true only for lane >= 16
            for (int i = 0; i < 64; ++i) v = v * 1.0001f + 0.5f;
        }
        out[tid] = v;
    }
}

// uniform: every lane does the same work, no divergence
__global__ void uniform(float *out, const float *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        for (int i = 0; i < 64; ++i) v = v * 1.0001f + 0.5f;
        out[tid] = v;
    }
}

int main(int argc, char **argv) {
    const char *which = (argc > 1) ? argv[1] : "divergent";
    int block = (argc > 2) ? atoi(argv[2]) : 256;
    int n = 1 << 20;                       // 1M elements

    float *in, *out;
    cudaMalloc(&in,  n * sizeof(float));
    cudaMalloc(&out, n * sizeof(float));
    cudaMemset(in, 0, n * sizeof(float));

    int grid = (n + block - 1) / block;

    if (strcmp(which, "uniform") == 0)
        uniform<<<grid, block>>>(out, in, n);
    else
        divergent<<<grid, block>>>(out, in, n);

    cudaError_t e = cudaDeviceSynchronize();
    printf("ran %s | block=%d grid=%d | sync: %s\n",
           which, block, grid, cudaGetErrorString(e));

    cudaFree(in); cudaFree(out);
    return 0;
}
