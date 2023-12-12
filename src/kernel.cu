#include "types.cu"
#include "utils.cu"

void generateIPv4PacketsKernel() {
    int numPackets = 2; // Number of packets to generate
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPackets + blockSize - 1) / blockSize; // Calculate the number of blocks
    
    // Allocate memory for the packets on the GPU
    IPv4Packet* d_packets;
    cudaMalloc((void**)&d_packets, numPackets * sizeof(IPv4Packet));
    
    // Call the utils kernel function
    generateIPv4Packets<<<numBlocks, blockSize>>>(d_packets, numPackets);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Copy the packets back to the host
    IPv4Packet* h_packets = new IPv4Packet[numPackets];
    cudaMemcpy(h_packets, d_packets, numPackets * sizeof(IPv4Packet), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numPackets; i++) {
        IPv4Packet packet = h_packets[i];
        // Print the source and destination addresses
        printf("Packet %d: Source Address: %u.%u.%u.%u, Destination Address: %u.%u.%u.%u, Payload: %s\n",
        i, 
        (packet.sourceAddress >> 24) & 0xFF, (packet.sourceAddress >> 16) & 0xFF, (packet.sourceAddress >> 8) & 0xFF, packet.sourceAddress & 0xFF,
        (packet.destinationAddress >> 24) & 0xFF, (packet.destinationAddress >> 16) & 0xFF, (packet.destinationAddress >> 8) & 0xFF, packet.destinationAddress & 0xFF, 
        packet.payload);
    }

    
    // Free the memory on the GPU
    cudaFree(d_packets);
    
    // Free the memory on the host
    delete[] h_packets;
}

void generateIPv6PacketsKernel() {
    int numPackets = 2; // Number of packets to generate
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPackets + blockSize - 1) / blockSize; // Calculate the number of blocks
    
    // Allocate memory for the IPv6 packets on the GPU
    IPv6Packet* d_ipv6Packets;
    cudaMalloc((void**)&d_ipv6Packets, numPackets * sizeof(IPv6Packet));
    
    // Call the utils kernel function to generate IPv6 packets
    generateIPv6Packets<<<numBlocks, blockSize>>>(d_ipv6Packets, numPackets);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Copy the IPv6 packets back to the host
    IPv6Packet* h_ipv6Packets = new IPv6Packet[numPackets];
    cudaMemcpy(h_ipv6Packets, d_ipv6Packets, numPackets * sizeof(IPv6Packet), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numPackets; i++) {
        IPv6Packet packet = h_ipv6Packets[i];
        // Print the source and destination addresses
        printf("IPv6 Packet %d: Source Address: ", i);
        for (int j = 0; j < 8; j++) {
            printf("%02x", packet.sourceAddress[j]);
            if (j < 7) {
                printf(":");
            }
        }
        printf(", Destination Address: ");
        for (int j = 0; j < 8; j++) {
            printf("%02x", packet.destinationAddress[j]);
            if (j < 7) {
                printf(":");
            }
        }
        printf(", Payload: %s\n", packet.payload);
    }

    
    // Free the memory for IPv6 packets on the GPU
    cudaFree(d_ipv6Packets);
    
    // Free the memory for IPv6 packets on the host
    delete[] h_ipv6Packets;
}