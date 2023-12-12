#include "kernel.cu"
#include "types.cu"

int main() {
  // Add semicolon
  GlobalPacketData globalPacketData;
  generateIPv4PacketsKernel(10, false, globalPacketData);
  generateIPv6PacketsKernel(10, false, globalPacketData);
  createRoutingTable(globalPacketData, false, 10);
  return 0;
}