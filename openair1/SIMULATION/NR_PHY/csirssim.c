/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "sim_base.h"
#include "nr_unitary_defs.h"

typedef struct {
  PhySim base;
  int codebook_type;
} CsirsSim;

static void print_usage(void)
{
  printf(
      "CsirsSim options\n"
      "\t-t <int> Codebook type [default: 1]\n");
}

#define OPTSTR_LEN 50

static void parse_args(CsirsSim *self, int argc, char **argv)
{
  const char opts[] = "t:h";
  char allopts[BASE_OPTSTR_LEN + OPTSTR_LEN];
  snprintf(allopts, sizeof(allopts), "%s%s", self->base.optstr, opts);

  optind = 1; // Start from first again after parsing common arguments
  int c = 0;
  while ((c = getopt(argc, argv, allopts)) != -1) {
    switch (c) {
      case 't':
        self->codebook_type = atoi(optarg);
        break;

      case '?':
      case ':':
      case 'h':
        physim_print_usage();
        print_usage();
        exit(-1);
    }
  }
}

CsirsSim *csirssim_new()
{
  CsirsSim *s = malloc(sizeof(*s));
  return s;
}

static int init(CsirsSim *self)
{
  physim_init(&self->base);

  return 0;
}

static void run_slot(CsirsSim *self)
{
}

static void print_stats(CsirsSim *self)
{
}

static void cleanup(CsirsSim *self)
{
  physim_cleanup(&self->base);
}

int main(int argc, char **argv)
{
  CsirsSim *sim = csirssim_new(argc, argv);

  physim_parse_common_args(&sim->base, argc, argv);

  parse_args(sim, argc, argv);

  init(sim);

  run_slot(sim);

  print_stats(sim);

  cleanup(sim);
  return 0;
}