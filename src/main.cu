#include "kernel.cu"
#include "types.cu"
#include <stdio.h>

int main() {
  GlobalPacketData globalPacketData;
  RoutingTableIPv4 routingTableIPv4;
  RoutingTableIPv6 routingTableIPv6;
  generateIPv4PacketsKernel(10, false, globalPacketData);
  generateIPv6PacketsKernel(10, false, globalPacketData);
  createRoutingTable(globalPacketData, false, 10, routingTableIPv4,  routingTableIPv6);
  
  return 0;
}