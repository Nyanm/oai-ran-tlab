#include "crc.h"

#include "defs_aiot_common.h"
#include "defs_aiot_r2d.h"

int main() {
    uint8_t buffer[8] = { 0xAF, 0x93, 0x42, 0x65, 0x88, 0x00, 0x00, 0x00 };
    int bufferBitLen = 12;//37; // 37 bits
    uint32_t crc = 0;

    // Initialize CRC tables
    crcTableInit();

    // Calculate the byte and bit position to insert the CRC
    int bytePos = bufferBitLen / 8;
    int bitPos = bufferBitLen % 8;

    // Insert CRC after bufferBitLen bits
    if (bufferBitLen > 24) {
        // CRC-16: insert 16 bits (left justified in crc)
        uint16_t crc16_val = crc16(buffer, bufferBitLen) >> 16; // Take the upper 16 bits
        if (bitPos == 0) {
            buffer[bytePos]     = (crc16_val >> 8) & 0xFF;
            buffer[bytePos + 1] = crc16_val & 0xFF;
        } else {
            buffer[bytePos]     |= (crc16_val >> (8 + bitPos)) & (0xFF >> bitPos);
            buffer[bytePos + 1]  = (crc16_val >> bitPos) & 0xFF;
            buffer[bytePos + 2]  = (crc16_val << (8 - bitPos)) & 0xFF;
        }

        bufferBitLen += 16; // total bits after CRC-16
    } else {
        // CRC-6: insert 6 bits (left justified in crc)
        uint8_t crc6_val = crc6(buffer, bufferBitLen) >> 26; // Take the upper 6 bits
        if (bitPos <= 2) {
            buffer[bytePos] |= (crc6_val & 0x3F) << (2 - bitPos);
        } else {
            buffer[bytePos] |= (crc6_val >> (bitPos - 2)) & (0xFF >> bitPos);
            buffer[bytePos + 1] = (crc6_val << (10 - bitPos)) & 0xFF;
        }

        bufferBitLen += 6; // total bits after CRC-6
    }
 
    // Line encoding: each bit is replaced by two bits (0 -> 01, 1 -> 10)
    uint8_t lineEncodingBuf[2 * ((bufferBitLen + 7) / 8)];
    memset(lineEncodingBuf, 0, sizeof(lineEncodingBuf));

    for (int i = 0; i < bufferBitLen; i++) {
        int byteIdx = i / 8;
        int bitIdx = 7 - (i % 8);
        int bit = (buffer[byteIdx] >> bitIdx) & 0x01;

        // Line encoding: 0 -> 01, 1 -> 10
        int outBitPos = 2 * i;
        int outByteIdx = outBitPos / 8;
        int outBitIdx = 7 - (outBitPos % 8);

        if (bit == 0) {
            // 0 -> 01
            lineEncodingBuf[outByteIdx] |= (1 << (outBitIdx - 1));
        } else {
            // 1 -> 10
            lineEncodingBuf[outByteIdx] |= (1 << outBitIdx);
        }
    }

    int txBufferLen = N_R_TAS_SIP + N_R_TAS_CAP + bufferBitLen + N_R2D_POSTAMBLE;
    uint8_t txBuffer[(txBufferLen + 7) / 8] = {0};

    txBuffer[0] = R_TAS_SIP; // R_TAS_SIP
    txBuffer[1] = R_TAS_CAP << 4; // R_TAS_CAP

    for (int i = 0; i < (txBufferLen + 7) / 8; i++) {
        txBuffer[i + 1] |= (lineEncodingBuf[i] >> 4) & 0x0F;
        txBuffer[i + 2] = (lineEncodingBuf[i + 1] & 0x0F) << 4;
    }

    int postambleBit = N_R_TAS_SIP + N_R_TAS_CAP + bufferBitLen;
    txBuffer[postambleBit] |= R2D_POSTAMBLE << (7 - (postambleBit % 8)); // R2D_POSTAMBLE



    return 0;
}