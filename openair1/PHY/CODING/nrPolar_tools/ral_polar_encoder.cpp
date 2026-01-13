#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"
//#include "src/utils/bits_to_bytes.hpp"
#include <cstring>
#include <mutex>
#include <iostream>
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>



// 16-entry nibble table for reversing 4-bit nibbles
static const uint8_t bitrev_nibble[16] = {
    0x0,0x8,0x4,0xC,0x2,0xA,0x6,0xE,
    0x1,0x9,0x5,0xD,0x3,0xB,0x7,0xF
};

void reverse_bits_per_byte_neon_aarch64(uint8_t *buf, size_t len) {
    size_t i = 0;

    // Load 16-byte nibble table into NEON register
    uint8x16_t tbl = vld1q_u8(bitrev_nibble);

    // Process 64 bytes per iteration (4 vectors × 16 bytes)
    for (; i + 64 <= len; i += 64) {
        // Load 4 NEON vectors
        uint8x16_t v0 = vld1q_u8(buf + i);
        uint8x16_t v1 = vld1q_u8(buf + i + 16);
        uint8x16_t v2 = vld1q_u8(buf + i + 32);
        uint8x16_t v3 = vld1q_u8(buf + i + 48);

        // Pointer array to simplify loop
        uint8x16_t *vecs[4] = { &v0, &v1, &v2, &v3 };

        // Process each vector
        for (int j = 0; j < 4; j++) {
            uint8x16_t v = *vecs[j];

            // Split high and low nibbles
            uint8x16_t hi = vshrq_n_u8(v, 4);              // high nibble
            uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F)); // low nibble

            // Lookup reversed nibbles
            uint8x16_t rev_hi = vqtbl1q_u8(tbl, hi);
            uint8x16_t rev_lo = vqtbl1q_u8(tbl, lo);

            // Combine nibbles
            *vecs[j] = vorrq_u8(vshlq_n_u8(rev_lo, 4), rev_hi);
        }

        // Store back
        vst1q_u8(buf + i, v0);
        vst1q_u8(buf + i + 16, v1);
        vst1q_u8(buf + i + 32, v2);
        vst1q_u8(buf + i + 48, v3);
    }

    // Handle remaining bytes (<64)
    for (; i < len; i++) {
        uint8_t byte = buf[i];
        buf[i] = (bitrev_nibble[byte & 0x0F] << 4) | bitrev_nibble[byte >> 4];
    }
}




extern "C" {

armral_status ral_polar_encoder(
    const uint8_t *data_in,
    uint32_t K,
    uint32_t E,
    uint32_t N,
    armral_polar_ibil_type i_bil,
    uint8_t *data_crc,
    uint8_t *frozen_mask,
    uint8_t *data_interleaved,
    uint8_t *data_encoded,
    uint8_t *data_out
) {
    // Step 1: CRC attachment
    // Input: data_in_bits (payload)
    // Output: data_crc_bits (payload + CRC)
    uint32_t crc_bits = 24;                 // CRC-24 (L = 24)
    uint32_t msg_bits = K - crc_bits; // message length (A = K - L)
    armral_polar_crc_attachment(data_in, msg_bits, data_crc); // CRC-24

    // Step 2: Frozen mask creation
    // Marks which bits are frozen and which carry information
    armral_polar_frozen_mask(N, E, K, 0, 0, frozen_mask);
    // Step 3: Subchannel interleaving
    // Permutes input bits according to Polar rate-matching interleaver
    armral_polar_subchannel_interleave(N, K, frozen_mask, data_crc, data_interleaved);


    // Step 4: Polar encoding
    // Encodes N bits from interleaved input bits
    armral_polar_encode_block(N, data_interleaved, data_encoded);

    // Step 5: Rate matching
    // Converts N bits to E bits for transmission
    armral_polar_rate_matching(N, E, K, i_bil, data_encoded, data_out);
    reverse_bits_per_byte_neon_aarch64(data_out,E);
    return ARMRAL_SUCCESS;
}

} // extern "C"
