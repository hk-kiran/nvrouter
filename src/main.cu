#include "kernel.cu"

int main() {
    generateIPv4PacketsKernel(1000, false);
    generateIPv6PacketsKernel(1000, false);
    generateIPv4PacketsKernel(1000, false);
    return 0;
}