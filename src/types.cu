// file for data types
#include <cstdint>
#include <stdio.h>

#ifndef ROUTINGENTRYIPV4
#define ROUTINGENTRYIPV4
struct RoutingEntryIPV4 {
    uint32_t destinationAddress;
    uint32_t subnetMask;
    uint32_t interface;
};
#endif

#ifndef ROUTINGENTRYIPV6
#define ROUTINGENTRYIPV6
struct RoutingEntryIPV6 {
  uint8_t destinationAddress[16]; // IPv6 address
  uint8_t subnetMask[16]; // IPv6 subnet mask
  int interface; // Interface number
};
#endif


#ifndef ROUTINGTABLEIPV4
#define ROUTINGTABLEIPV4
const int TABLE_SIZE = 10;
struct RoutingTableIPv4 {
  RoutingEntryIPV4 ipv4Entries[TABLE_SIZE];
};
#endif

#ifndef ROUTINGTABLEIPV6
#define ROUTINGTABLEIPV6
struct RoutingTableIPv6 {
  RoutingEntryIPV6 ipv6Entries[TABLE_SIZE]; // Store 16 bytes for each IPv6 address
};
#endif



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

#ifndef IPV6_PACKET_H
#define IPV6_PACKET_H

struct IPv6Packet {
    uint32_t version : 4;
    uint32_t trafficClass : 8;
    uint32_t flowLabel : 20;
    uint16_t payloadLength;
    uint8_t nextHeader;
    uint8_t hopLimit;
    uint16_t sourceAddress[8];
    uint16_t destinationAddress[8];
    uint8_t payload[1500]; // Maximum payload size of 1500 bytes
};

#endif 
#ifndef GLOBALPACKETDATA
#define GLOBALPACKETDATA
struct GlobalPacketData {
    IPv4Packet* ipv4Packets;
    IPv6Packet* ipv6Packets;
};
#endif// IPV6_PACKET_H