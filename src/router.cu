#include "kernel.cu"
#include "lib/types.hpp"

__global__ void ipv4packetProcessingKernel(int numPackets, GlobalPacketData* globalPacketData, RoutingTableIPv4* routingTableIPv4,
    NextHops* nextHops) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // get routing table in shared memory
    extern __shared__ RoutingEntryIPV4 sharedRoutingTableIPv4[];
    if (index < TABLE_SIZE) {
        sharedRoutingTableIPv4[index] = routingTableIPv4->ipv4Entries[index];
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
                nextHops->hops[index] = sharedRoutingTableIPv4[i].interface;

            }
        }
    }
}

void router(int numPackets, GlobalPacketData* globalPacketData, RoutingTableIPv4* routingTableIPv4, RoutingTableIPv6* routingTableIPv6,
    NextHops* nextHops) {

    int shared_mem_size = sizeof(routingTableIPv4);
    
    dim3 blockDim(256);
    dim3 gridDim((numPackets + blockDim.x - 1) / blockDim.x);
    
    cudaMemset(nextHops, 0, numPackets*sizeof(NextHops));
    cudaDeviceSynchronize();
    ipv4packetProcessingKernel<<<gridDim, blockDim, shared_mem_size>>>(numPackets, globalPacketData, routingTableIPv4, nextHops);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
}
