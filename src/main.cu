#include "router.cu"
#include <stdio.h>
#include <stdint.h>
#include <chrono>

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

  NextHops* nextHopsCPU = static_cast<NextHops*>(malloc(numPackets * sizeof(NextHops)));

  auto start = std::chrono::high_resolution_clock::now();
  ipv4packetProcessingCPU(numPackets, globalPacketData, routingTableIPv4, nextHopsCPU);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  printf("Time elapsed to route %d IPv4 packets on CPU: %f ms\n",numPackets, duration);
  double timePerPacketInNs = 1000000 * (duration / numPackets);
  printf("Time required per packet on CPU: %f\n", timePerPacketInNs);
  float bitsPerNs = double(MTU * 8) / timePerPacketInNs;
  printf("Throughput on CPU: %f Gb/s\n", bitsPerNs * 1e9 / (1024 * 1024 * 1024));

  verify(nextHops, nextHopsCPU);
  return 0;
}
