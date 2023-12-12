#include "types.cu"
#include "utils.cu"

void generatePacketsKernel() {
    int numPackets = 1000; // Number of packets to generate
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPackets + blockSize - 1) / blockSize; // Calculate the number of blocks
    
    // Allocate memory for the packets on the GPU
    IPv4Packet* d_packets;
    cudaMalloc((void**)&d_packets, numPackets * sizeof(IPv4Packet));
    
    // Call the utils kernel function
    generatePackets<<<numBlocks, blockSize>>>(d_packets, numPackets);
    
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
