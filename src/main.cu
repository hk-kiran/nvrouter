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

  int numBlockListedIPs = numPackets * 0.1;
  int* randomPackets;
  cudaMallocManaged(&randomPackets, numBlockListedIPs * sizeof(int));

  for (int i = 0; i < numBlockListedIPs; i++) {
    randomPackets[i] = rand() % numPackets;
  }

  uint32_t* blocklistedIPs;
  cudaMallocManaged(&blocklistedIPs, numBlockListedIPs * sizeof(int));
  for (int i = 0; i < numBlockListedIPs; i++) {
    int packetIndex = randomPackets[i];
    blocklistedIPs[i] = globalPacketData->ipv4Packets[packetIndex].destinationAddress;
  }
  if (debug) {
    for (int i = 0; i < numBlockListedIPs; i++) {
      printf("Blocked IPv4 Address %d: %d.%d.%d.%d\n", i, ( blocklistedIPs[i] >> 24) & 0xFF,
      ( blocklistedIPs[i] >> 16) & 0xFF, ( blocklistedIPs[i] >> 8) & 0xFF,  blocklistedIPs[i] & 0xFF);
    }
  }

  router(numPackets, numBlockListedIPs, globalPacketData, routingTableIPv4, routingTableIPv6, blocklistedIPs, nextHops);


  NextHops* nextHopsCPU = static_cast<NextHops*>(malloc(numPackets * sizeof(NextHops)));

  auto start = std::chrono::high_resolution_clock::now();
  ipv4packetProcessingCPU(numPackets, numBlockListedIPs, globalPacketData, routingTableIPv4, blocklistedIPs, nextHopsCPU);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
  printf("Time elapsed to route %d IPv4 packets on CPU: %f ms\n",numPackets, duration);
  double timePerPacketInNs = 1000000 * (duration / numPackets);
  printf("Time required per packet on CPU: %f ns\n", timePerPacketInNs);
  float bitsPerNs = double(MTU * 8) / timePerPacketInNs;
  printf("Throughput on CPU: %f Gb/s\n", bitsPerNs * 1e9 / (1024 * 1024 * 1024));

  verify(nextHops, nextHopsCPU);
  return 0;
}
