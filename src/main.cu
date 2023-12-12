#include "kernel.cu"
#include "types.cu"

int main() {
  // Add semicolon
  GlobalPacketData globalPacketData;
  generateIPv4PacketsKernel(1000, false, globalPacketData);
  generateIPv6PacketsKernel(1000, false, globalPacketData);
  createRoutingTable(globalPacketData);
  return 0;
}