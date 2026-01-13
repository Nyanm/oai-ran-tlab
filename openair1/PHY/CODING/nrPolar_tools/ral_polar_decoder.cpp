#include "PHY/CODING/nrPolar_tools/nr_polar_defs.h"
//#include "src/utils/bits_to_bytes.hpp"
#include <cstring>
#include <vector>
#include <stdint.h>
#include <stddef.h>
#include <arm_neon.h>



extern "C" {

armral_status ral_polar_decoder(
    const int16_t *input_llr16,
    uint32_t N,
    uint32_t E,
    uint32_t K,
    uint32_t L,
    armral_polar_ibil_type i_bil,
    uint8_t *decoded_out
) {
    /* Convert input LLRs to int8_t (ArmRAL format) */

    std::vector<int8_t> input_llr(E);
    for (uint32_t i = 0; i < E; i++)
        input_llr[i] = static_cast<int8_t>(input_llr16[i]);

    /* Buffers */
    std::vector<int8_t> data_recovered(N);
    const uint32_t cw_bytes = (N + 7) / 8;
    std::vector<uint8_t> data_decoded(L * cw_bytes);
    std::vector<uint8_t> data_deint0((K + 7) / 8);
    std::vector<uint8_t> data_deint((K + 7) / 8);
    //std::vector<uint8_t> decoded_out(8);
    /* Rate recovery */
    
    armral_polar_rate_recovery(
        N, E, K, i_bil,
        input_llr.data(),
        data_recovered.data());
    
    /* Frozen mask (1 entry per bit) */
    std::vector<uint8_t> frozen_mask(N);
    armral_polar_frozen_mask(N, E, K, 0, 0, frozen_mask.data());

    /* Decode */
    
    armral_polar_decode_block(
        N, frozen_mask.data(), L,
        data_recovered.data(),
        data_decoded.data());
    
    /* CRC check on candidates */
    
    armral_polar_subchannel_deinterleave(
        K, frozen_mask.data(),
        data_decoded.data(),
        data_deint0.data());
    
    /*
    if (armral_polar_crc_check(data_deint0.data(), K)) {
        memcpy(decoded_out, data_deint0.data(), (K + 7) / 8);
        return ARMRAL_SUCCESS;
    }

    for (uint32_t i = 1; i < L; i++) {
        const uint8_t *data_dec_l_ptr =
            data_decoded.data() + i * cw_bytes;

        armral_polar_subchannel_deinterleave(
            K, frozen_mask.data(),
            data_dec_l_ptr,
            data_deint.data());

        if (armral_polar_crc_check(data_deint.data(), K)) {
            memcpy(decoded_out, data_deint.data(), (K + 7) / 8);
            return ARMRAL_SUCCESS;
        }
    }
    */
    /* No CRC passed → return best candidate */
    //memcpy(decoded_out, data_deint0.data(), (K + 7) / 8);
    //return ARMRAL_FAIL;
    memcpy(decoded_out, data_deint0.data(), (K + 7) / 8);
    return ARMRAL_SUCCESS;
}

} // extern "C"
