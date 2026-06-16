#include <cstdio>
__global__ void hello() { printf("thread %d alive\n", threadIdx.x); }
int main() {
    hello<<<1, 4>>>();
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync: %s\n", cudaGetErrorString(e));
    return 0;
}
