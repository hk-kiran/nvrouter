#include "lib/types.hpp"
#include "lib/const.hpp"
#include "utils.cu"
#include <curand_kernel.h>


void generateIPv4PacketsKernel(int numPackets, bool debug,GlobalPacketData& globalPacketData) {
    
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPackets + blockSize - 1) / blockSize; // Calculate the number of blocks
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory for the packets on the GPU
    IPv4Packet* d_packets;
    cudaMalloc((void**)&d_packets, numPackets * sizeof(IPv4Packet));
    
    // Call the utils kernel function
    cudaEventRecord(start);
    generateIPv4Packets<<<numBlocks, blockSize>>>(d_packets, numPackets);
    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(error));
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    printf("Time elapsed to generate %d IPv4 packets : %f milliseconds\n", numPackets, milliseconds);
    // Copy the packets back to the host
    IPv4Packet* h_packets = new IPv4Packet[numPackets];
    cudaMemcpy(h_packets, d_packets, numPackets * sizeof(IPv4Packet), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numPackets; i++) {
      globalPacketData.ipv4Packets[i] = h_packets[i];
    }
    if (debug) {
        for (int i = 0; i < numPackets; i++) {
            IPv4Packet packet = h_packets[i];
            // Print the source and destination addresses
            printf("Packet %d: Source Address: %u.%u.%u.%u, Destination Address: %u.%u.%u.%u, Payload: %s\n",
            i, 
            (packet.sourceAddress >> 24) & 0xFF, (packet.sourceAddress >> 16) & 0xFF, (packet.sourceAddress >> 8) & 0xFF, packet.sourceAddress & 0xFF,
            (packet.destinationAddress >> 24) & 0xFF, (packet.destinationAddress >> 16) & 0xFF, (packet.destinationAddress >> 8) & 0xFF, packet.destinationAddress & 0xFF, 
            packet.payload);
        }
    }

    
    // Free the memory on the GPU
    cudaFree(d_packets);
    
    // Free the memory on the host
    delete[] h_packets;
}

void generateIPv6PacketsKernel(int numPackets, bool debug, GlobalPacketData& globalPacketData) {
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPackets + blockSize - 1) / blockSize; // Calculate the number of blocks
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate memory for the IPv6 packets on the GPU
    IPv6Packet* d_ipv6Packets;
    cudaMalloc((void**)&d_ipv6Packets, numPackets * sizeof(IPv6Packet));
    
    // Call the utils kernel function to generate IPv6 packets
    cudaEventRecord(start);
    generateIPv6Packets<<<numBlocks, blockSize>>>(d_ipv6Packets, numPackets);
    cudaEventRecord(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel failed: %s\n", cudaGetErrorString(error));
    }
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time elapsed to generate %d IPv6 packets : %f milliseconds\n", numPackets, milliseconds);
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Copy the IPv6 packets back to the host
    IPv6Packet* h_ipv6Packets = new IPv6Packet[numPackets];
    cudaMemcpy(h_ipv6Packets, d_ipv6Packets, numPackets * sizeof(IPv6Packet), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numPackets; i++) {
      globalPacketData.ipv6Packets[i] = h_ipv6Packets[i];
    }
    if (debug) {
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
    }

    
    // Free the memory for IPv6 packets on the GPU
    cudaFree(d_ipv6Packets);
    
    // Free the memory for IPv6 packets on the host
    delete[] h_ipv6Packets;
}

uint32_t generateRandomIPv4AddressHost() {
  return rand() % 0xFFFFFFFF; // Generate and return a random IPv4 address
}

uint8_t* generateRandomIPv6AddressHost() {
  uint8_t* address = new uint8_t[16]; // Allocate memory for the address


  for (int i = 0; i < 16; ++i) {
    address[i] = static_cast<uint8_t>(std::rand() % 256); // Generate a random byte for each octet
  }
  return address; // Return the generated address
}
void createRoutingTable(GlobalPacketData& globalPacketData, bool debug, int numPackets, RoutingTableIPv4& routingTableIPv4, RoutingTableIPv6& routingTableIPv6) {
  uint8_t subnetMask[16];
  for (int i = 0; i < 8; ++i) {
  subnetMask[i] = 0xFF; // Set all bytes to 0xFF for the network portion
    }
  for (int i = 8; i < 16; ++i) {
      subnetMask[i] = 0;} // Set remaining bytes to 0 for the host portion

  
  for (int i = 0; i < TABLE_SIZE-len_address_pool; i++) {
    // Populate IPv4 table
    if (i < numPackets) {
      routingTableIPv4.ipv4Entries[i].destinationAddress = globalPacketData.ipv4Packets[i].destinationAddress;
      
    } 
    else {
      routingTableIPv4.ipv4Entries[i].destinationAddress =  generateRandomIPv4AddressHost();
    }
    
    routingTableIPv4.ipv4Entries[i].subnetMask = 0xFFFFFFFF;
    routingTableIPv4.ipv4Entries[i].interface = i;

    // Populate IPv6 table
    if (i < numPackets) {
      memcpy(routingTableIPv6.ipv6Entries[i].destinationAddress,
       globalPacketData.ipv6Packets[i].destinationAddress, 16);
    } else {
      uint8_t* address = generateRandomIPv6AddressHost();
      memcpy(routingTableIPv6.ipv6Entries[i].destinationAddress, address, 16);
      delete[] address; // Free allocated memory
    }
    memcpy(routingTableIPv6.ipv6Entries[i].subnetMask, subnetMask, 16);
    routingTableIPv6.ipv6Entries[i].interface = i;
  }

  for (int i = TABLE_SIZE - len_address_pool; i < TABLE_SIZE; i++) {

    routingTableIPv4.ipv4Entries[i].subnetMask = 0xFF000000;
    routingTableIPv4.ipv4Entries[i].interface = i;
    routingTableIPv4.ipv4Entries[i].destinationAddress = (address_pool[i - (TABLE_SIZE - len_address_pool)] << 24);

  }
 
  if(debug){
  // Print routing tables
  printf("Routing Table (IPv4):\n");
  for (int i = 0; i < TABLE_SIZE; ++i) {
    printf("Entry %d: Destination Address: %u.%u.%u.%u, Subnet Mask: %u.%u.%u.%u, Interface: %u\n",
          i,
          (routingTableIPv4.ipv4Entries[i].destinationAddress >> 24) & 0xFF,
          (routingTableIPv4.ipv4Entries[i].destinationAddress >> 16) & 0xFF,
          (routingTableIPv4.ipv4Entries[i].destinationAddress >> 8) & 0xFF,
          routingTableIPv4.ipv4Entries[i].destinationAddress & 0xFF,
          (routingTableIPv4.ipv4Entries[i].subnetMask >> 24) & 0xFF,
          (routingTableIPv4.ipv4Entries[i].subnetMask >> 16) & 0xFF,
          (routingTableIPv4.ipv4Entries[i].subnetMask >> 8) & 0xFF,
          routingTableIPv4.ipv4Entries[i].subnetMask & 0xFF,
          routingTableIPv4.ipv4Entries[i].interface
          );
  }

  printf("Routing Table (IPv6):\n");
    for (int i = 0; i < TABLE_SIZE; ++i) {
      printf("Entry %d: Destination Address: ", i);
      for (int j = 0; j < 16; ++j) {
        printf("%02x", routingTableIPv6.ipv6Entries[i].destinationAddress[j]);


      }
      printf(", Subnet Mask: %d", i);
      for (int j = 0; j < 16; ++j) {
        printf("%02x", routingTableIPv6.ipv6Entries[i].subnetMask[j]);
      }
      printf(", Interface: %u\n", routingTableIPv6.ipv6Entries[i].interface);
    }
  }
}

