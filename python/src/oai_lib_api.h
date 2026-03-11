#ifndef OAI_LIB_API_H
#define OAI_LIB_API_H

#ifdef __cplusplus
extern "C" {
#endif

int oai_lib_init(void);
void oai_lib_shutdown(void);

//double oai_lib_add(double a, double b);

int oai_lib_nr_polar_decoder(int16_t *x, 
                             uint64_t *out, 
                             uint8_t ones_flag, 
                             int8_t messageType, 
                             uint16_t messageLength, 
                             uint8_t aggregation_level);

const char *oai_lib_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
