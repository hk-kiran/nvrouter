#include "kernel.cu"
#include "lib/types.hpp"

__global__ void blocklistedIPsKernel(int numPackets, int numBlockListedIPs, GlobalPacketData* globalPacketData,
    uint32_t* blocklistedIPs, NextHops* nextHops) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numPackets) {
        IPv4Packet packet = globalPacketData->ipv4Packets[index];
        for (int i = 0; i < numBlockListedIPs; i ++) {
            if (packet.destinationAddress == blocklistedIPs[i]) {
                nextHops->hops[index] = 255;
                break;
            }
        }
    }
}

__global__ void ipv4packetProcessingKernel(int numPackets, GlobalPacketData* globalPacketData,
     RoutingTableIPv4* routingTableIPv4, NextHops* nextHops) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // get routing table in shared memory
    extern __shared__ RoutingEntryIPV4 sharedRoutingTableIPv4[];
    if (threadIdx.x < TABLE_SIZE) {
        sharedRoutingTableIPv4[threadIdx.x] = routingTableIPv4->ipv4Entries[threadIdx.x];
    }
    __syncthreads();
    if (index < numPackets)
    {
        IPv4Packet packet = globalPacketData->ipv4Packets[index];
        // Extract the destination IP address from the packet
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            
            if ((packet.destinationAddress & sharedRoutingTableIPv4[i].subnetMask) == 
            sharedRoutingTableIPv4[i].destinationAddress) {
                // Next hop is the interface for this entry
                // printf("Packet %d: Destination Address: %u.%u.%u.%u Routing Table Destination Address: %u.%u.%u.%u Interface: %d\n",
                // index,
                // (packet.destinationAddress >> 24) & 0xFF, (packet.destinationAddress >> 16) & 0xFF,
                // (packet.destinationAddress >> 8) & 0xFF, packet.destinationAddress & 0xFF, 
                // (sharedRoutingTableIPv4[i].destinationAddress >> 24) & 0xFF, (sharedRoutingTableIPv4[i].destinationAddress >> 16) & 0xFF,
                // (sharedRoutingTableIPv4[i].destinationAddress >> 8) & 0xFF, sharedRoutingTableIPv4[i].destinationAddress & 0xFF,sharedRoutingTableIPv4[i].interface);
                nextHops->hops[index] = sharedRoutingTableIPv4[i].interface;
                break;
            }
        }
    }
}

void router(int numPackets, int numBlockListedIPs, GlobalPacketData* globalPacketData, RoutingTableIPv4* routingTableIPv4, RoutingTableIPv6* routingTableIPv6,
    uint32_t* blocklistedIPs, NextHops* nextHops) {

    int shared_mem_size = sizeof(routingTableIPv4);
    
    dim3 blockDim(256);
    dim3 gridDim((numPackets + blockDim.x - 1) / blockDim.x);
    
    cudaMemset(nextHops, 0, numPackets*sizeof(NextHops));
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ipv4packetProcessingKernel<<<gridDim, blockDim, shared_mem_size>>>(numPackets, globalPacketData, routingTableIPv4, nextHops);
    blocklistedIPsKernel<<<gridDim, blockDim>>>(numPackets, numBlockListedIPs, globalPacketData, blocklistedIPs, nextHops);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed to route %d IPv4 packets : %f milliseconds\n", numPackets, milliseconds);
    float timePerPacketInNs = 1000000 * (milliseconds / numPackets);
    printf("Time required per packet: %f ns\n", timePerPacketInNs);
    float bitsPerNs = float(MTU * 8) / timePerPacketInNs;
    printf("Throughput: %f Gb/s\n", bitsPerNs * 1e9 / (1024 * 1024 * 1024));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
}
