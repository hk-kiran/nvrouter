#include <curand_kernel.h>
#include "types.cu"

__device__ int deviceStrlen(const char* str) {
    int len = 0;
    while(str[len] != '\0') {
        len++;
    }
    return len;
}

// Function to build an IPv4 packet with payload
__device__ IPv4Packet buildIPv4PacketWithPayload(uint32_t sourceAddress, uint32_t destinationAddress) {
    IPv4Packet packet;
    
    // Set some dummy values of the packet fields
    packet.version = 4;
    packet.headerLength = 5; // 5 * 32-bit words
    packet.typeOfService = 0;
    packet.totalLength = sizeof(IPv4Packet);
    packet.identification = 12345;
    packet.flagsAndFragmentOffset = 0;
    packet.timeToLive = 64;
    packet.protocol = 6; // TCP
    packet.headerChecksum = 0;
    packet.sourceAddress = sourceAddress;
    packet.destinationAddress = destinationAddress;
    
    // Set the payload data
    const char* payloadData = "Dummy IPv4 payload data";
    memcpy(packet.payload, payloadData, deviceStrlen(payloadData));
    
    return packet;
}

// Kernel function to generate IPv4 packets
__device__ uint32_t randomizeAddress(curandState_t* state) {
    uint32_t address = 0;
    for (int i = 0; i < 4; i++) {
        address |= (curand(state) % 256) << (i * 8); 
    }
    return address;
}

__global__ void generateIPv4Packets(IPv4Packet* packets, int numPackets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPackets) {
        curandState_t state;
        curand_init(clock64(), idx, 0, &state); // Initialize random number generator
        
        uint32_t sourceAddress = randomizeAddress(&state); // Generate random source address
        uint32_t destinationAddress = randomizeAddress(&state); // Generate random destination address
        
        packets[idx] = buildIPv4PacketWithPayload(sourceAddress, destinationAddress);
    }
}

// Function to build an IPv6 packet with payload
__device__ IPv6Packet buildIPv6PacketWithPayload(uint8_t* sourceAddress, uint8_t* destinationAddress) {
    IPv6Packet packet;
    
    // Set some dummy values of the packet fields
    packet.version = 6;
    packet.trafficClass = 0;
    packet.flowLabel = 0;
    packet.payloadLength = sizeof(IPv6Packet);
    packet.nextHeader = 6; // TCP
    packet.hopLimit = 64;
    memcpy(packet.sourceAddress, sourceAddress, sizeof(packet.sourceAddress));
    memcpy(packet.destinationAddress, destinationAddress, sizeof(packet.destinationAddress));
    
    // Set the payload data
    const char* payloadData = "Dummy IPv6 payload data";
    memcpy(packet.payload, payloadData, deviceStrlen(payloadData));
    
    return packet;
}

__device__ uint8_t* randomizeIPv6Address(curandState_t* state) {
    uint8_t* address = new uint8_t[16];
    for (int i = 0; i < 16; i++) {
        address[i] = curand(state) % 256;
    }
    return address;
}

__global__ void generateIPv6Packets(IPv6Packet* packets, int numPackets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPackets) {
        curandState_t state;
        curand_init(clock64(), idx, 0, &state); // Initialize random number generator
        
        uint8_t* sourceAddress = randomizeIPv6Address(&state); // Generate random source address
        uint8_t* destinationAddress = randomizeIPv6Address(&state); // Generate random destination address
        
        packets[idx] = buildIPv6PacketWithPayload(sourceAddress, destinationAddress);
        
        delete[] sourceAddress;
        delete[] destinationAddress;
    }
}
