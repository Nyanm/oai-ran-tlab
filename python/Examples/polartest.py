import math
import random
import os
import sys
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import oaipylib as oai
import numpy as np

def nr_bit2byte_uint32_8(inp, array_size):
    array_ind = (array_size + 31) // 32
    out = [0] * array_size

    for i in range(array_ind - 1):
        for j in range(32):
            out[i * 32 + j] = (inp[i] >> j) & 1

    start = (array_ind - 1) * 32
    for j in range(array_size - start):
        out[start + j] = (inp[array_ind - 1] >> j) & 1

    return out


def nr_byte2bit_uint8_32(inp, array_size):
    array_ind = (array_size + 31) // 32
    out = [0] * array_ind

    for i in range(array_ind):
        val = 0
        for j in range(31, 0, -1):
            idx = i * 32 + j
            bit = inp[idx] if idx < array_size else 0
            val |= bit
            val <<= 1
        bit0 = inp[i * 32] if i * 32 < array_size else 0
        val |= bit0
        out[i] = val & 0xFFFFFFFF

    return out

def nr_bit2byte_uint32_8(inp, array_size):
    array_ind = (array_size + 31) // 32
    out = [0] * array_size

    for i in range(array_ind - 1):
        for j in range(32):
            out[i * 32 + j] = (inp[i] >> j) & 1

    start = (array_ind - 1) * 32
    for j in range(array_size - start):
        out[start + j] = (inp[array_ind - 1] >> j) & 1

    return out


def bpsk_from_uint32_words(inp, array_size):
    bits = nr_bit2byte_uint32_8(inp, array_size)
    scale = 1.0 / math.sqrt(2.0)
    return [scale if b == 0 else -scale for b in bits]



def add_awgn_and_convert_q15(samples, SNRdB):
    """
    Add zero-mean Gaussian noise and convert to signed Q15.

    Parameters
    ----------
    samples : sequence of float
        Input real-valued samples, e.g. BPSK symbols.
    noise_variance : float
        Variance of the Gaussian noise.

    Returns
    -------
    list[int]
        Noisy samples quantized to Q15 int16 values.
    """

    SNR_lin = 10**(SNRdB / 10.0)
    noise_variance = (1 / (2.0 * SNR_lin))
    if noise_variance < 0:
        raise ValueError("noise_variance must be non-negative")

    sigma = math.sqrt(noise_variance)
    out = []

    for x in samples:
        noisy = x + random.gauss(0.0, sigma)

        # Q15 scaling: float in approximately [-1, 1) -> int8_t range coded in int16_t
        q15 = int(round(noisy * 8.0))

        # Saturate to int16 range
        if q15 > 127:
            q15 = 127
        elif q15 < -128:
            q15 = -128

        out.append(q15)

    return out

## Testing

oai.init()
# this is a 32-bit input with format 0 (PBCH) which has 864 encoded bits 
encoder_input = np.array([0x12345678], dtype=np.uint64)
encoded_output = oai.nr_polar_encoder(encoder_input,0,0,0,32,0)
bpsk_out = bpsk_from_uint32_words(encoded_output,864)
#print(bpsk_out)
SNRdB = 0;
decoder_input = add_awgn_and_convert_q15(bpsk_out, SNRdB)
#print(decoder_input)

# here we pass the parameters to the OAI polar decoder, decoder_input (Q15 input), ones_flag = 0, messageType=0, messageLength = 864, aggregation_level = 0
decoder_output = oai.nr_polar_decoder(decoder_input,0,0,864,0)
#print(len(decoder_output))
print(hex(decoder_output[0]))

oai.shutdown()
