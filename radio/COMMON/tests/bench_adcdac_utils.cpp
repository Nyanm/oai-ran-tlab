#include <benchmark/benchmark.h>
#include "adcdac_utils.h"
#include <vector>
#include <numeric>
#include <simde/x86/avx2.h>
#include "common/platform_types.h"
#include "openair1/PHY/TOOLS/phy_test_tools.hpp"

// --- Reference Implementations ---
static void int16_to_msb_ref(int16_t *source, int16_t *dest, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    dest[i] = source[i] << 4;
  }
}

static void int16_from_msb_ref(int16_t *source, int16_t *dest, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    dest[i] = source[i] >> 4;
  }
}

// --- Benchmarks ---

static void BM_int16_to_msb(benchmark::State& state) {
  size_t n = state.range(0);
  AlignedVector512<int16_t> input(n);
  AlignedVector512<int16_t> output(n);
  std::iota(input.begin(), input.end(), 0);

  for (auto _ : state) {
    int16_to_msb(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}

static void BM_int16_to_msb_ref(benchmark::State& state) {
  size_t n = state.range(0);
  AlignedVector512<int16_t> input(n);
  AlignedVector512<int16_t> output(n);
  std::iota(input.begin(), input.end(), 0);

  for (auto _ : state) {
    int16_to_msb_ref(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}

static void BM_int16_from_msb(benchmark::State& state) {
  size_t n = state.range(0);
  AlignedVector512<int16_t> input(n);
  AlignedVector512<int16_t> output(n);
  std::iota(input.begin(), input.end(), 0);
  // Pre-shift input to be realistic
  for(auto& val : input) val <<= 4;

  for (auto _ : state) {
    int16_from_msb(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}

static void BM_int16_from_msb_ref(benchmark::State& state) {
  size_t n = state.range(0);
  AlignedVector512<int16_t> input(n);
  AlignedVector512<int16_t> output(n);
  std::iota(input.begin(), input.end(), 0);
  for(auto& val : input) val <<= 4;

  for (auto _ : state) {
    int16_from_msb_ref(input.data(), output.data(), n);
    benchmark::DoNotOptimize(output.data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_int16_to_msb)->Range(1<<10, 1<<15);
BENCHMARK(BM_int16_to_msb_ref)->Range(1<<10, 1<<15);
BENCHMARK(BM_int16_from_msb)->Range(1<<10, 1<<15);
BENCHMARK(BM_int16_from_msb_ref)->Range(1<<10, 1<<15);

BENCHMARK_MAIN();
