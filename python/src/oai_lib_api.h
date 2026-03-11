#ifndef OAI_LIB_API_H
#define OAI_LIB_API_H

#ifdef __cplusplus
extern "C" {
#endif

int oai_lib_init(void);
void oai_lib_shutdown(void);

double oai_lib_add(double a, double b);

int oai_lib_run_algorithm(const double *x, int n, double alpha, double *out);

const char *oai_lib_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
