// file for data types
#include <cstdint>
#include <stdio.h>

#ifndef IPV4_PACKET_H
#define IPV4_PACKET_H

struct IPv4Packet {
    uint8_t version;
    uint8_t headerLength;
    uint8_t typeOfService;
    uint16_t totalLength;
    uint16_t identification;
    uint16_t flagsAndFragmentOffset;
    uint8_t timeToLive;
    uint8_t protocol;
    uint16_t headerChecksum;
    uint32_t sourceAddress;
    uint32_t destinationAddress;
    uint8_t payload[1500]; // Maximum payload size of 1500 bytes
};

#endif // IPV4_PACKET_H