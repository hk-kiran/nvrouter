#include "router.cu"
#include <stdio.h>

int main() {
  GlobalPacketData* globalPacketData;
  RoutingTableIPv4* routingTableIPv4;
  RoutingTableIPv6* routingTableIPv6;
  NextHops* nextHops;

  int numPackets = NUM_PACKETS;
  bool debug = false;


  cudaMallocManaged(&globalPacketData, sizeof(GlobalPacketData));
  cudaMallocManaged(&routingTableIPv4, sizeof(RoutingTableIPv4));
  cudaMallocManaged(&routingTableIPv6, sizeof(RoutingTableIPv6));
  cudaMallocManaged(&nextHops, numPackets*sizeof(NextHops));

  generateIPv4PacketsKernel(numPackets, debug, *globalPacketData);
  generateIPv6PacketsKernel(numPackets, debug, *globalPacketData);
  createRoutingTable(*globalPacketData, debug, numPackets, *routingTableIPv4,  *routingTableIPv6);
  router(numPackets, globalPacketData, routingTableIPv4, routingTableIPv6, nextHops);

  // for (int i = 0; i < numPackets; i++) {
  //   printf("Next Hop for Packet %d: %d\n", i, nextHops->hops[i]);
  // }
  return 0;
}
