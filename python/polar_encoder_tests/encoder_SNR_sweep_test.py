import math
import os
import sys
import random
import json
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
import oaipylib as oai
import numpy as np

# --- Test Configuration ---
SNR_START = -10.0
SNR_END = 5.0
SNR_STEP = 0.5
SNR_RANGE = np.arange(SNR_START, SNR_END + SNR_STEP, SNR_STEP)

NUM_RUNS_PER_SNR = 5000            
ORIGINAL_INPUT = 0x12345678      
MESSAGE_LENGTH = 32              
AGGREGATION_LEVELS = [1, 2, 4, 8, 16] 
MESSAGE_TYPE = 1                 # DCI

# --- Helper Functions ---
def bpsk(inp, array_size):
    np_input = np.array(inp, dtype=np.uint32)
    bits = ((np_input[:, None] >> np.arange(32, dtype=np.uint32)) & 1).flatten()[:array_size]
    return (1 - 2 * bits.astype(np.int8)) / np.sqrt(2.0)

def add_awgn_and_quantize(samples, SNRdB):
    SNR_lin = 10**(SNRdB / 10.0)
    noise_variance = (1 / (2.0 * SNR_lin))
    sigma = math.sqrt(noise_variance)
    decoder_input = []
    for x in samples:
        noisy = x + random.gauss(0.0, sigma)
        q15 = int(round(noisy * 8.0))
        if q15 > 127: q15 = 127
        elif q15 < -128: q15 = -128
        decoder_input.append(q15)
    return decoder_input

# --- Main script ---
oai.init()


global_results = {}

try:
    for agg_level in AGGREGATION_LEVELS:
        encoded_len = 108 * agg_level
        #print(f"\n>> Aggregation Level: {agg_level} (E={encoded_len})")
        
        encoder_input = np.array([ORIGINAL_INPUT], dtype='Q')
        encoded_output = oai.nr_polar_encoder(encoder_input, 0, 0, MESSAGE_TYPE, MESSAGE_LENGTH, agg_level)
        bpsk_symbols = bpsk(encoded_output, encoded_len)
        
        results = {}
        for snr in SNR_RANGE:
            num_failures = 0
            for _ in range(NUM_RUNS_PER_SNR):
                decoder_input = add_awgn_and_quantize(bpsk_symbols, snr)
                decoder_output = oai.nr_polar_decoder(decoder_input, 0, MESSAGE_TYPE, MESSAGE_LENGTH, agg_level)
                if decoder_output[0] != ORIGINAL_INPUT:
                    num_failures += 1

            bler = num_failures / NUM_RUNS_PER_SNR
            # Use string key for SNR to make it JSON serializable
            results[f"{snr:.2f}"] = bler
            #print(f"   SNR: {snr:5.2f} dB | BLER: {bler:.4f}")
        
        global_results[str(agg_level)] = results

    output_file = "polar_results.json"
    with open(output_file, 'w') as f:
        json.dump(global_results, f, indent=4)
    print(f"\nResults saved to {output_file}")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd


        # Convert results to a DataFrame for easy plotting
        df = pd.DataFrame(global_results).transpose()
        df.index.name = 'Aggregation Level'
        df.columns.name = 'SNR (dB)'
        # Sort index and columns to ensure correct order
        df.index = df.index.astype(int)
        df = df.sort_index()
        df.columns = df.columns.astype(float)
        df = df.reindex(sorted(df.columns), axis=1)

        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=False, cmap='plasma', cbar_kws={'label': 'BLER'})
        plt.title(f"Polar Decoder BLER Heatmap (K={MESSAGE_LENGTH})")
        plt.ylabel("Aggregation Level (E = 108 * Agg)")
        plt.xlabel("SNR (dB)")
        
        plot_file = "polar_bler_heatmap.png"
        plt.savefig(plot_file)
        print(f"Heatmap saved to {plot_file}")
        
    except ImportError:
        print("\nMatplotlib/Seaborn not found. Skipping plotting.")
    except Exception as e:
        print(f"\nPlotting failed: {e}")

finally:
    oai.shutdown()
