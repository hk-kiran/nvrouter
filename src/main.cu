#include "kernel.cu"

int main() {
    generateIPv4PacketsKernel();
    generateIPv6PacketsKernel();
    return 0;
}