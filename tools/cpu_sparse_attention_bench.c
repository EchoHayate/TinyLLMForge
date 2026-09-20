// Can a CPU compute the selected-token attention inside the GPU's weight-read window?
//
// The gather bandwidth gate (2026-09-16) answered a narrower question: how fast can this
// CPU *move* scattered KV. It did not touch the arithmetic, the softmax, or the fact that
// a real offload kernel reads K, keeps 4 scores per token per kv head, and then reads V
// again. Those extra passes are where estimates usually die, so this measures the whole
// per-step attention instead of the memcpy that precedes it.
//
// Shape modelled: one decoding sequence, KV resident in host DRAM, laid out
//   K[layer][token][kv_head][dim], V likewise, bf16 storage (what the engine stores)
// Per decode step, for every layer:
//   - pick tokens_per_step tokens as whole `granularity`-sized units (query-aware
//     selection is assumed to have already happened; its cost is measured elsewhere)
//   - for each kv head, stream K once computing group_size dot products per token,
//     softmax over the selected tokens, then stream V once accumulating the output
//
// No staging buffer: a real kernel converts and multiplies while the line is hot, so
// charging a separate gather pass would inflate the cost by reading KV twice. The gather
// gate already measured what a separate pass would cost.
//
// Build:
//   gcc -O3 -march=native -fopenmp -o cpu_sparse_attention_bench cpu_sparse_attention_bench.c -lm
// Self test (fp64 reference, exits non-zero on mismatch):
//   ./cpu_sparse_attention_bench --self-test
// Measure:
//   numactl --interleave=all ./cpu_sparse_attention_bench --layers 36 --seq 8192 \
//       --tokens-per-step 912 --threads 64 --iters 20

#define _GNU_SOURCE
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
// So the self test can be compiled and run on a laptop without OpenMP; the measurement
// itself is meaningless single-threaded and the report says so.
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

typedef uint16_t bf16;

static inline float bf16_to_f32(bf16 v) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)v) << 16;
    return c.f;
}

static inline bf16 f32_to_bf16(float f) {
    union { uint32_t u; float f; } c;
    c.f = f;
    // round-to-nearest-even, so the self test is not comparing against a truncation bias
    uint32_t lsb = (c.u >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    return (bf16)((c.u + rounding) >> 16);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static inline uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

typedef struct {
    int layers;
    int seq;
    int kv_heads;
    int group_size;      // q heads per kv head
    int dim;
    int granularity;
    int tokens_per_step;
    int iters;
    int threads;
    int warmup;
} Config;

// One (layer, kv_head) attention over the selected token list, for an arbitrary KV layout.
// scores scratch: group_size * n_sel floats; out: group_size * dim floats.
//
// Structure matters more than it looks. The first version of this loop nested the q-head
// index inside the channel loop and let the compiler deal with the bf16 conversion; it ran
// at 10.9 GB/s effective, a tenth of what the same access pattern achieves in the gather
// benchmark, and it made the CPU look incapable when in fact the kernel was. The AVX-512
// path below converts 16 bf16 lanes at a time and keeps one accumulator register per q
// head, so K and V are each streamed exactly once and the group is amortised over that
// stream.
//
// The layout is passed in as strides rather than derived from `kv_heads`, because the
// correctness harness mirrors the engine's cache as [1, KVH, S, D] (head-major) while this
// benchmark models [S, KVH, D] (token-major). Two copies of the loop below would let the
// measured kernel and the verified kernel drift apart, which is exactly the failure this
// whole line of work is trying to avoid, so there is one implementation and the callers
// describe their own layout.
static void attend_head_strided(const bf16 *restrict kbase, const bf16 *restrict vbase,
                                const float *restrict q, const int *restrict sel, int n_sel,
                                size_t token_stride, size_t head_off, int dim, int group_size,
                                float scale, float *restrict scores, float *restrict out) {
#if defined(__AVX512F__)
    const int nchunk = dim / 16;
    for (int t = 0; t < n_sel; ++t) {
        const bf16 *kv = kbase + (size_t)sel[t] * token_stride + head_off;
        __m512 acc[8];
        for (int g = 0; g < group_size; ++g) acc[g] = _mm512_setzero_ps();
        for (int c = 0; c < nchunk; ++c) {
            const __m256i raw = _mm256_loadu_si256((const __m256i *)(kv + c * 16));
            const __m512 kvec = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
            for (int g = 0; g < group_size; ++g) {
                const __m512 qv = _mm512_loadu_ps(q + (size_t)g * dim + c * 16);
                acc[g] = _mm512_fmadd_ps(qv, kvec, acc[g]);
            }
        }
        for (int g = 0; g < group_size; ++g)
            scores[(size_t)g * n_sel + t] = _mm512_reduce_add_ps(acc[g]) * scale;
    }
#else
    for (int t = 0; t < n_sel; ++t) {
        const bf16 *kv = kbase + (size_t)sel[t] * token_stride + head_off;
        float acc[8];
        for (int g = 0; g < group_size; ++g) acc[g] = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float kd = bf16_to_f32(kv[d]);
            for (int g = 0; g < group_size; ++g) acc[g] += q[g * dim + d] * kd;
        }
        for (int g = 0; g < group_size; ++g) scores[g * n_sel + t] = acc[g] * scale;
    }
#endif

    for (int g = 0; g < group_size; ++g) {
        float *row = scores + (size_t)g * n_sel;
        float m = row[0];
        for (int t = 1; t < n_sel; ++t) if (row[t] > m) m = row[t];
        float sum = 0.0f;
        for (int t = 0; t < n_sel; ++t) { row[t] = expf(row[t] - m); sum += row[t]; }
        const float inv = 1.0f / sum;
        for (int t = 0; t < n_sel; ++t) row[t] *= inv;
        memset(out + (size_t)g * dim, 0, sizeof(float) * (size_t)dim);
    }

#if defined(__AVX512F__)
    for (int t = 0; t < n_sel; ++t) {
        const bf16 *vv = vbase + (size_t)sel[t] * token_stride + head_off;
        __m512 p[8];
        for (int g = 0; g < group_size; ++g)
            p[g] = _mm512_set1_ps(scores[(size_t)g * n_sel + t]);
        for (int c = 0; c < nchunk; ++c) {
            const __m256i raw = _mm256_loadu_si256((const __m256i *)(vv + c * 16));
            const __m512 vvec = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
            for (int g = 0; g < group_size; ++g) {
                float *dst = out + (size_t)g * dim + c * 16;
                _mm512_storeu_ps(dst, _mm512_fmadd_ps(p[g], vvec, _mm512_loadu_ps(dst)));
            }
        }
    }
#else
    for (int t = 0; t < n_sel; ++t) {
        const bf16 *vv = vbase + (size_t)sel[t] * token_stride + head_off;
        for (int d = 0; d < dim; ++d) {
            const float vd = bf16_to_f32(vv[d]);
            for (int g = 0; g < group_size; ++g) out[g * dim + d] += scores[g * n_sel + t] * vd;
        }
    }
#endif
}

// The benchmark's own layout: K[token][kv_head][dim].
static inline void attend_head(const bf16 *restrict kbase, const bf16 *restrict vbase,
                               const float *restrict q, const int *restrict sel, int n_sel,
                               int kv_heads, int head, int dim, int group_size,
                               float scale, float *restrict scores, float *restrict out) {
    attend_head_strided(kbase, vbase, q, sel, n_sel,
                        (size_t)kv_heads * (size_t)dim, (size_t)head * (size_t)dim,
                        dim, group_size, scale, scores, out);
}

// Draw whole units without replacement; always keep unit 0 and the last unit, which is
// what every arm in the end-to-end gate does (the last unit holds the current token).
static int draw_units(int *sel, int seq, int granularity, int tokens_per_step) {
    const int n_units = (seq + granularity - 1) / granularity;
    int want = tokens_per_step / granularity;
    if (want < 2) want = 2;
    if (want > n_units) want = n_units;

    static int *pool = NULL;
    static int pool_units = 0;
    if (pool_units != n_units) {
        free(pool);
        pool = (int *)malloc(sizeof(int) * (size_t)n_units);
        pool_units = n_units;
    }
    for (int i = 0; i < n_units; ++i) pool[i] = i;
    // Fisher-Yates over the interior; unit 0 and the last unit are forced in afterwards.
    for (int i = n_units - 1; i > 0; --i) {
        int j = (int)(xorshift64() % (uint64_t)(i + 1));
        int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
    }
    int chosen = 0;
    int units[n_units];
    units[chosen++] = 0;
    if (n_units > 1) units[chosen++] = n_units - 1;
    for (int i = 0; i < n_units && chosen < want; ++i) {
        int u = pool[i];
        if (u == 0 || u == n_units - 1) continue;
        units[chosen++] = u;
    }
    int n_sel = 0;
    for (int i = 0; i < chosen; ++i) {
        const int start = units[i] * granularity;
        for (int d = 0; d < granularity && start + d < seq; ++d) sel[n_sel++] = start + d;
    }
    return n_sel;
}

static int self_test(void) {
    // Small deterministic problem, every token selected, compared against fp64.
    // dim is a multiple of 16 so that the AVX-512 path is the one under test on the server;
    // testing the scalar fallback there would test code that never runs in the measurement.
    const int seq = 24, kv_heads = 2, group = 2, dim = 32;
    const size_t n = (size_t)seq * kv_heads * dim;
    bf16 *k = (bf16 *)malloc(sizeof(bf16) * n);
    bf16 *v = (bf16 *)malloc(sizeof(bf16) * n);
    float *q = (float *)malloc(sizeof(float) * (size_t)kv_heads * group * dim);
    rng_state = 12345;
    for (size_t i = 0; i < n; ++i) {
        k[i] = f32_to_bf16((float)((int)(xorshift64() % 200) - 100) / 100.0f);
        v[i] = f32_to_bf16((float)((int)(xorshift64() % 200) - 100) / 100.0f);
    }
    for (int i = 0; i < kv_heads * group * dim; ++i)
        q[i] = (float)((int)(xorshift64() % 200) - 100) / 100.0f;

    int *sel = (int *)malloc(sizeof(int) * seq);
    for (int t = 0; t < seq; ++t) sel[t] = t;
    const float scale = 1.0f / sqrtf((float)dim);

    float *scores = (float *)malloc(sizeof(float) * (size_t)group * seq);
    float *out = (float *)malloc(sizeof(float) * (size_t)group * dim);
    double max_err = 0.0;

    for (int h = 0; h < kv_heads; ++h) {
        attend_head(k, v, q + (size_t)h * group * dim, sel, seq, kv_heads, h, dim, group,
                    scale, scores, out);
        for (int g = 0; g < group; ++g) {
            double sc[64], m = -1e300, sum = 0.0, ref[64];
            for (int t = 0; t < seq; ++t) {
                double acc = 0.0;
                for (int d = 0; d < dim; ++d)
                    acc += (double)q[(h * group + g) * dim + d] *
                           (double)bf16_to_f32(k[(size_t)t * kv_heads * dim + h * dim + d]);
                sc[t] = acc * (double)scale;
                if (sc[t] > m) m = sc[t];
            }
            for (int t = 0; t < seq; ++t) { sc[t] = exp(sc[t] - m); sum += sc[t]; }
            for (int d = 0; d < dim; ++d) ref[d] = 0.0;
            for (int t = 0; t < seq; ++t) {
                const double p = sc[t] / sum;
                for (int d = 0; d < dim; ++d)
                    ref[d] += p * (double)bf16_to_f32(v[(size_t)t * kv_heads * dim + h * dim + d]);
            }
            for (int d = 0; d < dim; ++d) {
                double e = fabs(ref[d] - (double)out[g * dim + d]);
                if (e > max_err) max_err = e;
            }
        }
    }
    printf("self_test max_abs_err=%.3e\n", max_err);
    free(k); free(v); free(q); free(sel); free(scores); free(out);
    if (!(max_err < 1e-5)) {
        fprintf(stderr, "self test FAILED\n");
        return 1;
    }
    printf("self test OK\n");
    return 0;
}

// The pipeline prototype links this same translation unit as a shared library so that the
// scheduling experiment uses the exact kernel that produced the 2.051 ms number, instead of
// a re-implementation that could quietly differ. It defines this guard to drop main().
#ifndef CPU_SPARSE_ATTENTION_NO_MAIN
int main(int argc, char **argv) {
    Config cfg = {.layers = 36, .seq = 8192, .kv_heads = 8, .group_size = 4, .dim = 128,
                  .granularity = 32, .tokens_per_step = 912, .iters = 20, .threads = 0,
                  .warmup = 3};
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--self-test")) return self_test();
        else if (!strcmp(argv[i], "--layers")) cfg.layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq")) cfg.seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-heads")) cfg.kv_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group-size")) cfg.group_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")) cfg.dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--granularity")) cfg.granularity = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tokens-per-step")) cfg.tokens_per_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) cfg.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) cfg.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")) cfg.warmup = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (cfg.group_size > 8) { fprintf(stderr, "group_size > 8 not supported\n"); return 2; }
#if defined(__AVX512F__)
    if (cfg.dim % 16) {
        fprintf(stderr, "dim must be a multiple of 16 for the AVX-512 path (got %d)\n", cfg.dim);
        return 2;
    }
#endif
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);

    const size_t per_layer = (size_t)cfg.seq * cfg.kv_heads * cfg.dim;
    const size_t total = per_layer * (size_t)cfg.layers;
    bf16 *K = (bf16 *)malloc(sizeof(bf16) * total);
    bf16 *V = (bf16 *)malloc(sizeof(bf16) * total);
    float *Q = (float *)malloc(sizeof(float) * (size_t)cfg.layers * cfg.kv_heads *
                               cfg.group_size * cfg.dim);
    if (!K || !V || !Q) { fprintf(stderr, "allocation failed\n"); return 3; }

    // Touch on many threads so first-touch does not pin the pool to one NUMA node; the
    // gather gate already showed interleaving beats pinning for this access pattern.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        K[i] = (bf16)(0x3f00 + (i & 0x3f));
        V[i] = (bf16)(0x3e80 + (i & 0x3f));
    }
    rng_state = 0x1234567;
    for (size_t i = 0; i < (size_t)cfg.layers * cfg.kv_heads * cfg.group_size * cfg.dim; ++i)
        Q[i] = (float)((int)(xorshift64() % 200) - 100) / 100.0f;

    const int max_sel = cfg.seq;
    int *sel = (int *)malloc(sizeof(int) * (size_t)max_sel * cfg.layers);
    const int n_threads = omp_get_max_threads();
    float *scratch = (float *)malloc(sizeof(float) * (size_t)n_threads *
                                     ((size_t)cfg.group_size * max_sel + (size_t)cfg.group_size * cfg.dim));
    if (!sel || !scratch) { fprintf(stderr, "allocation failed\n"); return 3; }
    const size_t scratch_stride = (size_t)cfg.group_size * max_sel + (size_t)cfg.group_size * cfg.dim;
    const float scale = 1.0f / sqrtf((float)cfg.dim);

    double checksum = 0.0;
    double best_ms = 1e300, worst_ms = 0.0, total_ms = 0.0;
    int n_sel_last = 0;

    for (int it = 0; it < cfg.warmup + cfg.iters; ++it) {
        int n_sel = 0;
        for (int l = 0; l < cfg.layers; ++l)
            n_sel = draw_units(sel + (size_t)l * max_sel, cfg.seq, cfg.granularity,
                               cfg.tokens_per_step);
        n_sel_last = n_sel;

        const double t0 = now_sec();
        double local_sum = 0.0;
        #pragma omp parallel for collapse(2) schedule(static) reduction(+:local_sum)
        for (int l = 0; l < cfg.layers; ++l) {
            for (int h = 0; h < cfg.kv_heads; ++h) {
                const int tid = omp_get_thread_num();
                float *scores = scratch + (size_t)tid * scratch_stride;
                float *out = scores + (size_t)cfg.group_size * max_sel;
                attend_head(K + (size_t)l * per_layer, V + (size_t)l * per_layer,
                            Q + ((size_t)l * cfg.kv_heads + h) * cfg.group_size * cfg.dim,
                            sel + (size_t)l * max_sel, n_sel, cfg.kv_heads, h, cfg.dim,
                            cfg.group_size, scale, scores, out);
                for (int g = 0; g < cfg.group_size; ++g) local_sum += out[g * cfg.dim];
            }
        }
        const double ms = (now_sec() - t0) * 1e3;
        checksum += local_sum;
        if (it >= cfg.warmup) {
            total_ms += ms;
            if (ms < best_ms) best_ms = ms;
            if (ms > worst_ms) worst_ms = ms;
        }
    }

    const double mean_ms = total_ms / (double)cfg.iters;
    // bytes: K and V each read once per selected token per kv head per layer
    const double bytes = 2.0 * (double)n_sel_last * cfg.kv_heads * cfg.dim *
                         sizeof(bf16) * (double)cfg.layers;
    // flops: QK and PV, 2 flops per MAC
    const double flops = 2.0 * 2.0 * (double)n_sel_last * cfg.kv_heads * cfg.group_size *
                         cfg.dim * (double)cfg.layers;

    printf("threads=%d layers=%d seq=%d kv_heads=%d group=%d dim=%d gran=%d\n",
           n_threads, cfg.layers, cfg.seq, cfg.kv_heads, cfg.group_size, cfg.dim,
           cfg.granularity);
    printf("tokens_per_step_requested=%d tokens_selected=%d (%.2f%% of context)\n",
           cfg.tokens_per_step, n_sel_last, 100.0 * n_sel_last / cfg.seq);
    printf("step_ms mean=%.3f best=%.3f worst=%.3f\n", mean_ms, best_ms, worst_ms);
    printf("bytes_per_step=%.1f MiB eff_bandwidth=%.1f GB/s eff_gflops=%.1f\n",
           bytes / (1024.0 * 1024.0), bytes / (mean_ms * 1e-3) / 1e9,
           flops / (mean_ms * 1e-3) / 1e9);
    printf("checksum=%.6f\n", checksum);

    free(K); free(V); free(Q); free(sel); free(scratch);
    return 0;
}
#endif  // CPU_SPARSE_ATTENTION_NO_MAIN
