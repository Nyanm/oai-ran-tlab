#ifndef __NR_LDPC_CUDA_SHARED_Z__H__
#define __NR_LDPC_CUDA_SHARED_Z__H__
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CHECK(call) ErrorCheck((call), __FILE__, __LINE__)
/**
 * @brief Checks CUDA error status and prints detailed diagnostic info if an error occurred.
 *
 * @param error_code The CUDA error code returned from a CUDA runtime API call.
 * @param filename   The name of the source file where the error occurred.
 * @param lineNumber The line number in the source file where the error occurred.
 * @return cudaError_t Returns the same error code passed in, for optional further handling.
 */
inline cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber)
{
  if (error_code != cudaSuccess) {
    printf("[CUDA ERROR] %s (%d): %s\nOccurred in file: %s at line %d\n",
           cudaGetErrorName(error_code),
           error_code,
           cudaGetErrorString(error_code),
           filename,
           lineNumber);
  }
  return error_code;
}

#define BG1_ROW 46      // number of rows in base graph (BG1)
#define BG1_COL 68      // number of cols in base graph (BG1)

#define BG2_ROW 42      // number of rows in base graph (BG2)
#define BG2_COL 52      // number of cols in base graph (BG2)

#define BG1_MAX_NNZ 316  // Maximum number of non-zero elements in BG1
#define BG1_MAX_ROW_DEGREE 19  // Maximum row degree in BG1

#define MAX_Z 384     // Maximum lift size in 5G NR

#define ALPHA 0.75f    // Scaling factor for Min-Sum algorithm

// Constants for quantization

#define SIMD_MAX_LLR 0x7F7F7F7FU
#define SIMD_WIDTH 4


#define BG1_MAX_CW_LEN 26112    // Maximum codeword length for BG1
#define BG1_MAX_INFO_LEN 8448
#define BG1_R13_Z384_Kb 22


#define MAX_STREAMS 4 // Maximum number of CUDA streams
#define GROUPS_PER_STREAM 4 // Number of groups per stream
#define CWS_PER_STREAM (GROUPS_PER_STREAM * SIMD_WIDTH) // Number of codewords per stream
#define CWS_PER_BATCH (MAX_STREAMS * CWS_PER_STREAM) // Number of codewords per batch

#define BG1_R23_Z384_CN 13
#define BG1_R23_Z384_VN 35
#define BG1_R23_Z384_NNZ 144

#define BG1_R13_Z384_CN 46
#define BG1_R13_Z384_VN 68
#define BG1_R13_Z384_NNZ 316

// ======================================= BG1 rate 1/3 Z=384 ========================================
// CSR representation of base graph 1 rate 1/3 parity-check matrix for Z=384
const int bg1_r13_z384_row_ptr[BG1_R13_Z384_CN + 1] = {0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 122, 129, 137, 144, 150, 157, 164, 170, 176, 182, 
    188, 194, 200, 205, 210, 216, 221, 226, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 279, 284, 
    289, 293, 298, 302, 307, 312, 316};

const int bg1_r13_z384_col_idx[BG1_R13_Z384_NNZ] = { 0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 0, 
    2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24, 0, 1, 
    2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 24, 25, 0, 1, 3, 
    4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 25, 0, 1, 26, 0, 
    1, 3, 12, 16, 21, 22, 27, 0, 6, 10, 11, 13, 17, 18, 20, 28, 0, 1, 4, 7, 
    8, 14, 29, 0, 1, 3, 12, 16, 19, 21, 22, 24, 30, 0, 1, 10, 11, 13, 17, 18, 
    20, 31, 1, 2, 4, 7, 8, 14, 32, 0, 1, 12, 16, 21, 22, 23, 33, 0, 1, 10, 
    11, 13, 18, 34, 0, 3, 7, 20, 23, 35, 0, 12, 15, 16, 17, 21, 36, 0, 1, 10, 
    13, 18, 25, 37, 1, 3, 11, 20, 22, 38, 0, 14, 16, 17, 21, 39, 1, 12, 13, 18, 
    19, 40, 0, 1, 7, 8, 10, 41, 0, 3, 9, 11, 22, 42, 1, 5, 16, 20, 21, 43, 
    0, 12, 13, 17, 44, 1, 2, 10, 18, 45, 0, 3, 4, 11, 22, 46, 1, 6, 7, 14, 
    47, 0, 2, 4, 15, 48, 1, 6, 8, 49, 0, 4, 19, 21, 50, 1, 14, 18, 25, 51, 
    0, 10, 13, 24, 52, 1, 7, 22, 25, 53, 0, 12, 14, 24, 54, 1, 2, 11, 21, 55, 
    0, 7, 15, 17, 56, 1, 6, 12, 22, 57, 0, 14, 15, 18, 58, 1, 13, 23, 59, 0, 
    9, 10, 12, 60, 1, 3, 7, 19, 61, 0, 8, 17, 62, 1, 3, 9, 18, 63, 0, 4, 
    24, 64, 1, 16, 18, 25, 65, 0, 7, 9, 22, 66, 1, 6, 10, 67};

const int bg1_r13_z384_shift_cn[BG1_R13_Z384_NNZ] = { 307, 19, 50, 369, 181, 216, 317, 288, 109, 17, 357, 215, 106, 242, 180, 330, 346, 1, 0, 76, 
    76, 73, 288, 144, 331, 331, 178, 295, 342, 217, 99, 354, 114, 331, 112, 0, 0, 0, 205, 250, 
    328, 332, 256, 161, 267, 160, 63, 129, 200, 88, 53, 131, 240, 205, 13, 0, 0, 276, 87, 0, 
    275, 199, 153, 56, 132, 305, 231, 341, 212, 304, 300, 271, 39, 357, 1, 0, 332, 181, 0, 195, 
    14, 115, 166, 241, 51, 157, 0, 278, 257, 1, 351, 92, 253, 18, 225, 0, 9, 62, 316, 333, 
    290, 114, 0, 307, 179, 165, 18, 39, 224, 368, 67, 170, 0, 366, 232, 321, 133, 57, 303, 63, 
    82, 0, 101, 339, 274, 111, 383, 354, 0, 48, 102, 8, 47, 188, 334, 115, 0, 77, 186, 174, 
    232, 50, 74, 0, 313, 177, 266, 115, 370, 0, 142, 248, 137, 89, 347, 12, 0, 241, 2, 210, 
    318, 55, 269, 0, 13, 338, 57, 289, 57, 0, 260, 303, 81, 358, 375, 0, 130, 163, 280, 132, 
    4, 0, 145, 213, 344, 242, 197, 0, 187, 206, 264, 341, 59, 0, 205, 102, 328, 213, 97, 0, 
    30, 11, 233, 22, 0, 24, 89, 61, 27, 0, 298, 158, 235, 339, 234, 0, 72, 17, 383, 312, 
    0, 71, 81, 76, 136, 0, 194, 194, 101, 0, 222, 19, 244, 274, 0, 252, 5, 147, 78, 0, 
    159, 229, 260, 90, 0, 100, 215, 258, 256, 0, 102, 201, 175, 287, 0, 323, 8, 361, 105, 0, 
    230, 148, 202, 312, 0, 320, 335, 2, 266, 0, 210, 313, 297, 21, 0, 269, 82, 115, 0, 185, 
    177, 289, 214, 0, 258, 93, 346, 297, 0, 175, 37, 312, 0, 52, 314, 139, 288, 0, 113, 14, 
    218, 0, 113, 132, 114, 168, 0, 80, 78, 163, 274, 0, 135, 149, 15, 0};


// CSC representation of base graph 1 rate 1/3 parity-check matrix for Z=384
const int bg1_r13_z384_col_ptr[BG1_R13_Z384_VN + 1] = {0, 30, 58, 65, 76, 85, 89, 97, 109, 117, 124, 136, 146, 158, 169, 179, 186, 196, 206, 219, 
    226, 234, 245, 257, 262, 268, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 
    308, 309, 310, 311, 312, 313, 314, 315, 316};

const int bg1_r13_z384_row_idx[BG1_R13_Z384_NNZ] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 19, 20, 22, 24, 
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 
    12, 15, 16, 18, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 0, 1, 
    2, 10, 23, 26, 33, 0, 1, 3, 5, 8, 13, 16, 20, 24, 39, 41, 1, 2, 3, 7, 
    10, 24, 26, 28, 42, 0, 1, 2, 21, 0, 2, 3, 6, 25, 27, 35, 45, 1, 2, 3, 
    7, 10, 13, 19, 25, 31, 34, 39, 44, 1, 2, 3, 7, 10, 19, 27, 40, 0, 1, 2, 
    20, 38, 41, 44, 0, 2, 3, 6, 9, 12, 15, 19, 23, 30, 38, 45, 0, 1, 3, 6, 
    9, 12, 16, 20, 24, 33, 0, 1, 3, 5, 8, 11, 14, 18, 22, 32, 35, 38, 0, 2, 
    3, 6, 9, 12, 15, 18, 22, 30, 37, 1, 2, 3, 7, 10, 17, 25, 29, 32, 36, 0, 
    1, 2, 14, 26, 34, 36, 0, 1, 3, 5, 8, 11, 14, 17, 21, 43, 1, 2, 3, 6, 
    9, 14, 17, 22, 34, 40, 0, 2, 3, 6, 9, 12, 15, 18, 23, 29, 36, 41, 43, 0, 
    1, 2, 8, 18, 28, 39, 0, 2, 3, 6, 9, 13, 16, 21, 0, 1, 3, 5, 8, 11, 
    14, 17, 21, 28, 33, 0, 1, 3, 5, 8, 11, 16, 20, 24, 31, 35, 44, 0, 1, 11, 
    13, 37, 1, 2, 8, 30, 32, 42, 2, 3, 15, 29, 31, 43, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45};

const int bg1_r13_z384_shift_vn[BG1_R13_Z384_NNZ]  = {307, 76, 205, 276, 332, 195, 278, 9, 307, 366, 48, 77, 313, 142, 241, 260, 145, 187, 30, 298, 
    71, 222, 159, 102, 230, 210, 185, 175, 113, 80, 19, 250, 87, 181, 14, 62, 179, 232, 101, 102, 
    186, 2, 13, 130, 213, 205, 24, 72, 194, 252, 100, 323, 320, 269, 258, 52, 113, 135, 50, 76, 
    328, 339, 89, 81, 8, 369, 73, 0, 115, 165, 177, 338, 206, 158, 93, 314, 288, 332, 275, 316, 
    274, 235, 76, 19, 14, 181, 144, 256, 102, 216, 161, 199, 257, 17, 194, 335, 149, 331, 267, 153, 
    333, 111, 266, 344, 383, 215, 148, 346, 78, 331, 160, 56, 290, 383, 242, 101, 37, 317, 178, 63, 
    264, 177, 139, 163, 288, 129, 132, 1, 321, 174, 210, 197, 61, 229, 289, 15, 109, 295, 305, 351, 
    133, 232, 57, 341, 339, 361, 17, 342, 231, 166, 18, 8, 248, 163, 11, 201, 2, 214, 357, 200, 
    341, 92, 57, 50, 318, 280, 233, 260, 82, 217, 88, 212, 114, 354, 303, 312, 5, 175, 313, 215, 
    99, 53, 137, 136, 202, 297, 106, 354, 304, 241, 39, 47, 89, 81, 328, 132, 114, 131, 300, 253, 
    303, 347, 358, 22, 312, 312, 242, 240, 271, 18, 63, 74, 55, 132, 27, 147, 21, 288, 114, 180, 
    331, 205, 224, 4, 244, 297, 330, 13, 39, 225, 82, 115, 289, 213, 346, 112, 357, 51, 368, 188, 
    12, 375, 97, 274, 105, 1, 0, 1, 157, 67, 334, 57, 59, 234, 258, 266, 274, 0, 0, 115, 
    370, 115, 0, 0, 170, 90, 287, 218, 0, 0, 269, 78, 256, 168, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const int bg1_r13_z384_csc2csr[BG1_R13_Z384_NNZ] =  {0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 129, 137, 144, 150, 157, 170, 182, 188, 200, 210, 
    221, 230, 240, 250, 260, 270, 279, 289, 298, 307, 1, 39, 58, 77, 80, 97, 104, 114, 122, 130, 
    138, 158, 164, 176, 183, 194, 205, 216, 226, 235, 245, 255, 265, 275, 284, 293, 302, 312, 2, 20, 
    40, 123, 206, 222, 256, 3, 21, 59, 81, 105, 145, 165, 189, 211, 285, 294, 22, 41, 60, 98, 
    124, 212, 223, 231, 299, 4, 23, 42, 195, 5, 43, 61, 88, 217, 227, 266, 313, 24, 44, 62, 
    99, 125, 146, 184, 218, 246, 261, 286, 308, 25, 45, 63, 100, 126, 185, 228, 290, 6, 26, 46, 
    190, 280, 295, 309, 7, 47, 64, 89, 115, 139, 159, 186, 207, 241, 281, 314, 8, 27, 65, 90, 
    116, 140, 166, 191, 213, 257, 9, 28, 66, 82, 106, 131, 151, 177, 201, 251, 267, 282, 10, 48, 
    67, 91, 117, 141, 160, 178, 202, 242, 276, 29, 49, 68, 101, 127, 171, 219, 236, 252, 271, 11, 
    30, 50, 152, 224, 262, 272, 12, 31, 69, 83, 107, 132, 153, 172, 196, 303, 32, 51, 70, 92, 
    118, 154, 173, 203, 263, 291, 13, 52, 71, 93, 119, 142, 161, 179, 208, 237, 273, 296, 304, 14, 
    33, 53, 108, 180, 232, 287, 15, 54, 72, 94, 120, 147, 167, 197, 16, 34, 73, 84, 109, 133, 
    155, 174, 198, 233, 258, 17, 35, 74, 85, 110, 134, 168, 192, 214, 247, 268, 310, 18, 36, 135, 
    148, 277, 37, 55, 111, 243, 253, 300, 56, 75, 162, 238, 248, 305, 78, 86, 95, 102, 112, 121, 
    128, 136, 143, 149, 156, 163, 169, 175, 181, 187, 193, 199, 204, 209, 215, 220, 225, 229, 234, 239, 
    244, 249, 254, 259, 264, 269, 274, 278, 283, 288, 292, 297, 301, 306, 311, 315};


// ======================================== BG1 rate 2/3 Z=384 ========================================
// CSR representation of base graph 1 rate 2/3 parity-check matrix for Z=384
const int bg1_r23_z384_row_ptr[BG1_R23_Z384_CN + 1] = {0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 122, 129, 137, 144};
const int bg1_r23_z384_col_idx[BG1_R23_Z384_NNZ] = { 0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 
    0, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24, 
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 24, 25,
    0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 25, 
    0, 1, 26,
    0, 1, 3, 12, 16, 21, 22, 27, 
    0, 6, 10, 11, 13, 17, 18, 20, 28, 
    0, 1, 4, 7, 8, 14, 29, 
    0, 1, 3, 12, 16, 19, 21, 22, 24, 30,
    0, 1, 10, 11, 13, 17, 18, 20, 31, 
    1, 2, 4, 7, 8, 14, 32,
    0, 1, 12, 16, 21, 22, 23, 33, 
    0, 1, 10, 11, 13, 18, 34};

const int bg1_r23_z384_shift_cn[BG1_R23_Z384_NNZ] = {307, 19, 50, 369, 181, 216, 317, 288, 109, 17, 357, 215, 106, 242, 180, 330, 346, 1, 0, 
    76, 76, 73, 288, 144, 331, 331, 178, 295, 342, 217, 99, 354, 114, 331, 112, 0, 0, 0, 
    205, 250, 328, 332, 256, 161, 267, 160, 63, 129, 200, 88, 53, 131, 240, 205, 13, 0, 0, 
    276, 87, 0, 275, 199, 153, 56, 132, 305, 231, 341, 212, 304, 300, 271, 39, 357, 1, 0, 
    332, 181, 0, 
    195, 14, 115, 166, 241, 51, 157, 0, 
    278, 257, 1, 351, 92, 253, 18, 225, 0, 
    9, 62, 316, 333, 290, 114, 0, 
    307, 179, 165, 18, 39, 224, 368, 67, 170, 0, 
    366, 232, 321, 133, 57, 303, 63, 82, 0, 
    101, 339, 274, 111, 383, 354, 0, 
    48, 102, 8, 47, 188, 334, 115, 0, 
    77, 186, 174, 232, 50, 74, 0};

// CSC representation of base graph 1 rate 2/3 parity-check matrix for Z=384
const int bg1_r23_z384_col_ptr[BG1_R23_Z384_VN + 1] = {0, 12, 23, 27, 32, 37, 40, 44, 49, 54, 57, 63, 69, 75, 81, 86, 89, 95, 100, 106,
     110, 115, 121, 127, 130, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144};

const int bg1_r23_z384_row_idx[BG1_R23_Z384_NNZ] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 0, 2, 3, 4, 5, 7, 8, 9, 
    10, 11, 12, 0, 1, 2, 10, 0, 1, 3, 5, 8, 1, 2, 3, 7, 10, 0, 1, 2, 
    0, 2, 3, 6, 1, 2, 3, 7, 10, 1, 2, 3, 7, 10, 0, 1, 2, 0, 2, 3, 
    6, 9, 12, 0, 1, 3, 6, 9, 12, 0, 1, 3, 5, 8, 11, 0, 2, 3, 6, 9, 
    12, 1, 2, 3, 7, 10, 0, 1, 2, 0, 1, 3, 5, 8, 11, 1, 2, 3, 6, 9, 
    0, 2, 3, 6, 9, 12, 0, 1, 2, 8, 0, 2, 3, 6, 9, 0, 1, 3, 5, 8, 
    11, 0, 1, 3, 5, 8, 11, 0, 1, 11, 1, 2, 8, 2, 3, 4, 5, 6, 7, 8, 
    9, 10, 11, 12};

const int bg1_r23_z384_shift_vn[BG1_R23_Z384_NNZ]  = {307, 76, 205, 276, 332, 195, 278, 9, 307, 366, 48, 77, 19, 250, 87, 181, 14, 62, 179, 232, 
    101, 102, 186, 50, 76, 328, 339, 369, 73, 0, 115, 165, 288, 332, 275, 316, 274, 181, 144, 256, 
    216, 161, 199, 257, 331, 267, 153, 333, 111, 331, 160, 56, 290, 383, 317, 178, 63, 288, 129, 132, 
    1, 321, 174, 109, 295, 305, 351, 133, 232, 17, 342, 231, 166, 18, 8, 357, 200, 341, 92, 57, 
    50, 217, 88, 212, 114, 354, 215, 99, 53, 106, 354, 304, 241, 39, 47, 114, 131, 300, 253, 303, 
    242, 240, 271, 18, 63, 74, 180, 331, 205, 224, 330, 13, 39, 225, 82, 346, 112, 357, 51, 368, 
    188, 1, 0, 1, 157, 67, 334, 0, 0, 115, 0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0};

const int bg1_r23_z384_csc2csr[BG1_R23_Z384_NNZ] = {0, 19, 38, 57, 76, 79, 87, 96, 103, 113, 129, 137, 1, 39, 58, 77, 80, 97, 104, 114, 
    122, 130, 138, 2, 20, 40, 123, 3, 21, 59, 81, 105, 22, 41, 60, 98, 124, 4, 23, 42, 
    5, 43, 61, 88, 24, 44, 62, 99, 125, 25, 45, 63, 100, 126, 6, 26, 46, 7, 47, 64, 
    89, 115, 139, 8, 27, 65, 90, 116, 140, 9, 28, 66, 82, 106, 131, 10, 48, 67, 91, 117, 
    141, 29, 49, 68, 101, 127, 11, 30, 50, 12, 31, 69, 83, 107, 132, 32, 51, 70, 92, 118, 
    13, 52, 71, 93, 119, 142, 14, 33, 53, 108, 15, 54, 72, 94, 120, 16, 34, 73, 84, 109, 
    133, 17, 35, 74, 85, 110, 134, 18, 36, 135, 37, 55, 111, 56, 75, 78, 86, 95, 102, 112, 
    121, 128, 136, 143};


struct CompressedH_cn {
    int* row_ptr;     
    int* col_idx;     
    int* shift_cn;    
    int nnz;          
};

struct CompressedH_vn {
    int* col_ptr;    
    int* row_idx;     
    int* shift_vn;    
    int* csc2csr;     
    int nnz; 
};

struct ldpc_params {
    int rate;                     // Code rate
    int bg_index;                   // Index of the 5G NR base graph (1 or 2)
    int Z;                          // Lifting factor
    int N;                          // Codeword length (N = nb * Z)
    int K;                          // Number of information bits (K = (nb - mb) * Z)
    int Kb;                         // Number of information bits in the base graph (Kb = nb - mb)
    int mb;                         // Number of rows in the base graph (i.e., number of parity check equations)
    int nb;                         // Number of columns in the base graph (i.e., total variable nodes in base graph)
    int* H;                         // Pointer to the parity-check matrix H 
    struct CompressedH_cn* compH_cn; // Compressed representation of H for check-node (CN) processing (CSR)
    struct CompressedH_vn* compH_vn; // Compressed representation of H for variable-node (VN) processing (CSC)
    int n_iterations;               // Maximum number of decoding iterations
};

struct kernel_compH_cn {
    int row_ptr[BG1_ROW + 1];       // Row pointer array (CSR format)

    int col_idx[BG1_MAX_NNZ];       // Column indices
    int shift_cn[BG1_MAX_NNZ];      // Shift values for check nodes
    int nnz;                        // Number of non-zero elements
};

struct kernel_compH_vn {
    int col_ptr[BG1_COL + 1];       // Column pointer array (CSC format)

    int row_idx[BG1_MAX_NNZ];       // Row indices
    int shift_vn[BG1_MAX_NNZ];      // Shift values for variable nodes
    int csc2csr[BG1_MAX_NNZ];       // Mapping from CSC to CSR indexing
    int nnz;                        // Number of non-zero elements
};


struct host_memory {
    // Large pinned (page-locked) host memory blocks
    int8_t* h_big_pinned_llr;        // Pinned buffer for LLRs
    uint8_t* h_big_hard_bits;        // Pinned buffer for hard bits

    // Arrays of pointers
    int8_t** h_pinned_llr;           // Pointers to initial LLR values in pinned memory
    uint8_t** h_pinned_hard;         // Pointers to hard decision results in pinned memory
};

struct device_memory {
    // Regular device memory
    int8_t* d_big_init_llr;          // Large buffer for initial LLRs on device
    uint32_t* d_big_app;             // Large buffer for APP values on device
    int8_t* d_big_app_reordered;     // Large buffer for reordered APP values on device
    uint32_t* d_big_c2v;             // Large buffer for check-to-variable messages on device
    uint32_t* d_big_delta_c2v;       // Large buffer for delta (min-sum correction) in C2V messages on device
    uint8_t* d_big_hard_bits;        // Large buffer for hard decision bits on device

    // Arrays of device pointers
    int8_t** d_init_llr;             // Pointers to initial LLR values on device
    uint32_t** d_app;                // Pointers to APP values on device
    int8_t** d_app_reordered;        // Pointers to reordered APP values on device
    uint32_t** d_c2v;                // Pointers to check-to-variable (C2V) messages on device
    uint32_t** d_delta_c2v;          // Pointers to delta C2V messages on device
    uint8_t** d_hard_bits;           // Pointers to hard decision results on device
};


struct cuda_grid{
    dim3 order_grid;    // Order kernel grid
    dim3 order_block;   // Order kernel block
    dim3 reorder_grid; // Reorder kernel grid
    dim3 reorder_block; // Reorder kernel block
    dim3 cnp_grid;      // CNP kernel grid
    dim3 cnp_block;     // CNP kernel block
    dim3 vnp_grid;      // VNP kernel grid
    dim3 vnp_block;     // VNP kernel block
    dim3 hard_grid;    // Hard decision kernel grid
    dim3 hard_block;   // Hard decision kernel block
    size_t cnp_shared_size; // Shared memory size for CNP kernel
};

void ldpc_decoder_cuda_init();
void ldpc_decoder_cuda_free();


// Global variables
extern struct ldpc_params *params;

// Create Streams and Graphs, and execute graphs
extern cudaStream_t* cudaStreams;
extern cudaGraph_t* cudaGraphs_R13;
extern cudaGraph_t* cudaGraphs_R23;
extern cudaGraphExec_t* cudaGraphExecs_R13;
extern cudaGraphExec_t* cudaGraphExecs_R23;

// Define host and device memory
extern struct host_memory *host_mem;
extern struct device_memory *dev_mem;

#ifdef __cplusplus
}
#endif

#endif
