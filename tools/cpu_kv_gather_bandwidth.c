/* cpu_kv_gather_bandwidth.c
 *
 * Answers the one question that gates CPU-side query-aware attention:
 * when you gather SCATTERED KV blocks out of host DRAM, what bandwidth do you
 * actually get, and how many cores does it take to get there?
 *
 * Why this shape: with query-aware selection the engine does not stream KV
 * sequentially. It picks k of L tokens, so it reads scattered chunks. Peak
 * DRAM numbers from spec sheets are sequential numbers and do not apply.
 * The design needs >=100 GB/s effective; below ~80 GB/s the whole direction is
 * dead, so this benchmark is deliberately built to be able to kill it.
 *
 * Chunk sizes map onto real KV layout for Qwen3-8B:
 *   4 KiB   = one token, one layer  (8 KV heads * 128 dim * 2 (K,V) * 2 B)
 *   128 KiB = a 32-token block, one layer
 *   1 MiB   = a 256-token block, one layer (current KVCACHE_BLOCK_SIZE)
 *
 * Build:  cc -O3 -o cpu_kv_gather_bandwidth cpu_kv_gather_bandwidth.c -lpthread
 * Run:    ./cpu_kv_gather_bandwidth [buffer_gib] [max_threads]
 * Table goes to stderr, machine-readable JSON to stdout.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    const uint8_t *buf;
    const uint32_t *idx;
    size_t n_idx;
    size_t chunk_bytes;
    uint64_t sink;
} task_t;

static void *worker(void *arg) {
    task_t *t = (task_t *)arg;
    const size_t words = t->chunk_bytes / sizeof(uint64_t);
    uint64_t acc = 0;
    for (size_t i = 0; i < t->n_idx; i++) {
        const uint64_t *p =
            (const uint64_t *)(t->buf + (size_t)t->idx[i] * t->chunk_bytes);
        /* 8 u64 = one 64 B cache line per iteration; cheap enough that this
         * stays memory bound rather than turning into an ALU benchmark. */
        for (size_t w = 0; w + 8 <= words; w += 8) {
            acc ^= p[w] ^ p[w + 1] ^ p[w + 2] ^ p[w + 3] ^
                   p[w + 4] ^ p[w + 5] ^ p[w + 6] ^ p[w + 7];
        }
    }
    t->sink = acc;
    return NULL;
}

static volatile uint64_t g_keep_alive;

/* Fisher-Yates with a fixed seed so runs are comparable across machines. */
static void shuffle(uint32_t *a, size_t n, uint64_t seed) {
    for (size_t i = n - 1; i > 0; i--) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (size_t)((seed >> 33) % (i + 1));
        uint32_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

static double run_once(const uint8_t *buf, const uint32_t *idx, size_t n_idx,
                       size_t chunk_bytes, int nthreads) {
    pthread_t th[256];
    task_t tk[256];
    size_t per = n_idx / (size_t)nthreads;
    if (per == 0) return 0.0;
    double t0 = now_s();
    for (int i = 0; i < nthreads; i++) {
        tk[i].buf = buf;
        tk[i].idx = idx + (size_t)i * per;
        tk[i].n_idx = per;
        tk[i].chunk_bytes = chunk_bytes;
        tk[i].sink = 0;
        pthread_create(&th[i], NULL, worker, &tk[i]);
    }
    uint64_t s = 0;
    for (int i = 0; i < nthreads; i++) {
        pthread_join(th[i], NULL);
        s ^= tk[i].sink;
    }
    double dt = now_s() - t0;
    g_keep_alive = s;
    double bytes = (double)per * (double)nthreads * (double)chunk_bytes;
    return bytes / dt / 1e9; /* GB/s */
}

int main(int argc, char **argv) {
    double buf_gib = (argc > 1) ? atof(argv[1]) : 8.0;
    int max_threads = (argc > 2) ? atoi(argv[2]) : 0;
    if (max_threads <= 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        max_threads = (n > 0) ? (int)n : 8;
    }
    if (max_threads > 256) max_threads = 256;

    size_t bytes = (size_t)(buf_gib * 1073741824.0);
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, bytes) != 0 || !buf) {
        fprintf(stderr, "alloc of %.1f GiB failed\n", buf_gib);
        return 1;
    }
    /* Touch every page so we measure DRAM, not page-fault handling. */
    for (size_t i = 0; i < bytes; i += 4096) buf[i] = (uint8_t)(i >> 12);

    const size_t chunk_sizes[] = {4096, 131072, 1048576};
    const char *chunk_names[] = {"4KiB   (1 tok x 1 layer)",
                                 "128KiB (32 tok block)",
                                 "1MiB   (256 tok block)"};
    int thread_grid[16], n_tg = 0;
    for (int t = 1; t <= max_threads; t *= 2) thread_grid[n_tg++] = t;
    if (thread_grid[n_tg - 1] != max_threads) thread_grid[n_tg++] = max_threads;

    fprintf(stderr, "buffer %.1f GiB, cores online %d\n", buf_gib, max_threads);
    fprintf(stderr, "GB/s; SEQ = in-order chunks, RAND = shuffled chunks "
                    "(what query-aware selection actually does)\n");

    printf("{\"buffer_gib\":%.1f,\"cores\":%d,\"results\":[\n", buf_gib,
           max_threads);
    int first = 1;
    for (int c = 0; c < 3; c++) {
        size_t cb = chunk_sizes[c];
        size_t n_chunks = bytes / cb;
        uint32_t *seq = malloc(n_chunks * sizeof(uint32_t));
        uint32_t *rnd = malloc(n_chunks * sizeof(uint32_t));
        if (!seq || !rnd) { fprintf(stderr, "idx alloc failed\n"); return 1; }
        for (size_t i = 0; i < n_chunks; i++) seq[i] = (uint32_t)i;
        memcpy(rnd, seq, n_chunks * sizeof(uint32_t));
        shuffle(rnd, n_chunks, 0x9E3779B97F4A7C15ULL);

        fprintf(stderr, "\n=== chunk %s ===\n", chunk_names[c]);
        fprintf(stderr, "%8s %10s %10s %10s\n", "threads", "SEQ", "RAND",
                "RAND/SEQ");
        for (int i = 0; i < n_tg; i++) {
            int nt = thread_grid[i];
            double bs = 0, br = 0;
            for (int r = 0; r < 3; r++) { /* best of 3 */
                double v = run_once(buf, seq, n_chunks, cb, nt);
                if (v > bs) bs = v;
                v = run_once(buf, rnd, n_chunks, cb, nt);
                if (v > br) br = v;
            }
            fprintf(stderr, "%8d %10.1f %10.1f %9.2f\n", nt, bs, br,
                    bs > 0 ? br / bs : 0.0);
            printf("%s  {\"chunk_bytes\":%zu,\"threads\":%d,\"seq_gb_s\":%.2f,"
                   "\"rand_gb_s\":%.2f,\"rand_over_seq\":%.4f}",
                   first ? "" : ",\n", cb, nt, bs, br,
                   bs > 0 ? br / bs : 0.0);
            first = 0;
        }
        free(seq);
        free(rnd);
    }
    printf("\n]}\n");
    free(buf);
    return 0;
}
