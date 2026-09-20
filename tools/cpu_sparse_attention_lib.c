// Shared-library wrapper around the CPU sparse attention kernel.
//
// The pipeline prototype needs to run the *same* kernel that produced the measured
// 2.051 ms/step, from inside a Python process that is simultaneously driving the GPU. A
// re-implementation in numpy or a second C file could quietly differ in vectorisation and
// would make the scheduling result meaningless, so this file includes the benchmark's
// translation unit and only adds an entry point that runs one decode step's worth of
// attention over an already-allocated KV pool.
//
// Build:
//   gcc -O3 -march=native -fopenmp -shared -fPIC -o libcpu_sparse_attn.so \
//       cpu_sparse_attention_lib.c -lm

#define CPU_SPARSE_ATTENTION_NO_MAIN
#include "cpu_sparse_attention_bench.c"

#include <pthread.h>

typedef struct {
    Config cfg;
    // Async submission lives here, in C, on purpose. The first pipeline measurement drove the
    // overlap from a Python thread and came out *worse* than the serial schedule in every
    // arm, including with CUDA graphs. That is not evidence about the mechanism: a Python
    // worker has to take the GIL to start, to return, and to hand the result back, and the
    // default switch interval is 5 ms - the same order as the entire effect being measured.
    // With submit/wait below, Python issues a non-blocking call and the work runs on a
    // detached pthread that never touches the interpreter, which is also what a real
    // integration would do.
    pthread_t worker;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int job_pending;
    int job_done;
    int shutdown;
    int job_tokens;
    int job_threads;
    double job_elapsed;
    int worker_started;
    bf16 *k;
    bf16 *v;
    float *q;
    int *sel;
    float *scores;   // per-thread scratch
    float *out;      // per-thread scratch
    int scratch_threads;
    size_t layer_elems;
} CsaHandle;

// Allocates the KV pool once. First-touch is done by the same thread that allocates, which
// is wrong for NUMA locality, so the caller is expected to run under
// `numactl --interleave=all` exactly like the standalone benchmark does.
void *csa_create(int layers, int seq, int kv_heads, int group_size, int dim,
                 int granularity, int max_threads) {
    CsaHandle *h = (CsaHandle *)calloc(1, sizeof(CsaHandle));
    if (!h) return NULL;
    pthread_mutex_init(&h->mu, NULL);
    pthread_cond_init(&h->cv, NULL);
    h->cfg.layers = layers;
    h->cfg.seq = seq;
    h->cfg.kv_heads = kv_heads;
    h->cfg.group_size = group_size;
    h->cfg.dim = dim;
    h->cfg.granularity = granularity;

    h->layer_elems = (size_t)seq * (size_t)kv_heads * (size_t)dim;
    const size_t total = h->layer_elems * (size_t)layers;
    h->k = (bf16 *)malloc(total * sizeof(bf16));
    h->v = (bf16 *)malloc(total * sizeof(bf16));
    const int q_heads = kv_heads * group_size;
    h->q = (float *)malloc((size_t)layers * (size_t)q_heads * (size_t)dim * sizeof(float));
    h->sel = (int *)malloc((size_t)seq * sizeof(int));
    h->scratch_threads = max_threads > 0 ? max_threads : 1;
    h->scores = (float *)malloc((size_t)h->scratch_threads * (size_t)group_size *
                                (size_t)seq * sizeof(float));
    h->out = (float *)malloc((size_t)h->scratch_threads * (size_t)group_size *
                             (size_t)dim * sizeof(float));
    if (!h->k || !h->v || !h->q || !h->sel || !h->scores || !h->out) {
        return NULL;
    }
    for (size_t i = 0; i < total; ++i) {
        h->k[i] = f32_to_bf16(((float)(xorshift64() % 2000) - 1000.0f) / 1000.0f);
        h->v[i] = f32_to_bf16(((float)(xorshift64() % 2000) - 1000.0f) / 1000.0f);
    }
    const size_t qn = (size_t)layers * (size_t)q_heads * (size_t)dim;
    for (size_t i = 0; i < qn; ++i) {
        h->q[i] = ((float)(xorshift64() % 2000) - 1000.0f) / 1000.0f;
    }
    return h;
}

// One decode step: every layer, every kv head, over a freshly drawn selection.
// Returns elapsed seconds. `threads` <= max_threads passed to csa_create.
double csa_step(void *handle, int tokens_per_step, int threads) {
    CsaHandle *h = (CsaHandle *)handle;
    if (!h) return -1.0;
    const Config *c = &h->cfg;
    if (threads > h->scratch_threads) threads = h->scratch_threads;
    omp_set_num_threads(threads);

    const int n_sel = draw_units(h->sel, c->seq, c->granularity, tokens_per_step);
    const float scale = 1.0f / sqrtf((float)c->dim);
    const int q_heads = c->kv_heads * c->group_size;

    const double t0 = now_sec();
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int layer = 0; layer < c->layers; ++layer) {
        for (int head = 0; head < c->kv_heads; ++head) {
            const int tid = omp_get_thread_num();
            float *scores = h->scores + (size_t)tid * (size_t)c->group_size * (size_t)c->seq;
            float *out = h->out + (size_t)tid * (size_t)c->group_size * (size_t)c->dim;
            const bf16 *kbase = h->k + (size_t)layer * h->layer_elems;
            const bf16 *vbase = h->v + (size_t)layer * h->layer_elems;
            const float *q = h->q + ((size_t)layer * (size_t)q_heads +
                                     (size_t)head * (size_t)c->group_size) * (size_t)c->dim;
            attend_head(kbase, vbase, q, h->sel, n_sel, c->kv_heads, head, c->dim,
                        c->group_size, scale, scores, out);
        }
    }
    return now_sec() - t0;
}

static void *csa_worker_main(void *arg) {
    CsaHandle *h = (CsaHandle *)arg;
    pthread_mutex_lock(&h->mu);
    for (;;) {
        while (!h->job_pending && !h->shutdown) {
            pthread_cond_wait(&h->cv, &h->mu);
        }
        if (h->shutdown) break;
        const int tokens = h->job_tokens;
        const int threads = h->job_threads;
        h->job_pending = 0;
        pthread_mutex_unlock(&h->mu);

        const double elapsed = csa_step(h, tokens, threads);

        pthread_mutex_lock(&h->mu);
        h->job_elapsed = elapsed;
        h->job_done = 1;
        pthread_cond_broadcast(&h->cv);
    }
    pthread_mutex_unlock(&h->mu);
    return NULL;
}

// Non-blocking: queues one step for the worker thread and returns immediately.
int csa_submit(void *handle, int tokens_per_step, int threads) {
    CsaHandle *h = (CsaHandle *)handle;
    if (!h) return -1;
    pthread_mutex_lock(&h->mu);
    if (!h->worker_started) {
        h->worker_started = 1;
        pthread_mutex_unlock(&h->mu);
        if (pthread_create(&h->worker, NULL, csa_worker_main, h) != 0) return -2;
        pthread_mutex_lock(&h->mu);
    }
    h->job_tokens = tokens_per_step;
    h->job_threads = threads;
    h->job_done = 0;
    h->job_pending = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
    return 0;
}

// Blocks until the queued step finishes; returns its elapsed seconds.
double csa_wait(void *handle) {
    CsaHandle *h = (CsaHandle *)handle;
    if (!h) return -1.0;
    pthread_mutex_lock(&h->mu);
    while (!h->job_done) {
        pthread_cond_wait(&h->cv, &h->mu);
    }
    const double elapsed = h->job_elapsed;
    pthread_mutex_unlock(&h->mu);
    return elapsed;
}

// One decode step's attention over a KV cache the *caller* owns, for one layer.
//
// Why this exists, separately from csa_step: everything above allocates its own random KV pool
// and draws its own selection, which is what a throughput measurement wants and what a
// correctness check must not accept. The correctness harness mirrors the engine's real cache
// into pinned CPU memory, gets its indices from a GPU-side Quest selector, and has to see this
// exact kernel produce the output tensor that goes back to the GPU. Otherwise the fast path and
// the verified path are two different pieces of code and the verification proves nothing about
// the thing that was benchmarked.
//
// Layout is described by strides so the caller does not have to transpose: for a torch tensor
// [1, KVH, S, D] that is contiguous, head_stride = S*D and token_stride = D. `sel` holds
// absolute token positions, so no staging buffer is needed - the kernel gathers while the line
// is hot, exactly as the benchmark does.
//
//   k, v   : bf16, kv_heads * head_stride elements addressable
//   q      : fp32 [kv_heads * group_size, dim], q heads of one kv head contiguous
//            (which is how repeat_kv orders them: q head = kv * group_size + r)
//   out    : fp32 [kv_heads * group_size, dim], fully written
//
// Threading note, so the number this feeds is not over-read: the work here is one layer, so
// the only parallelism available is over kv_heads (8 for Qwen3-8B). The 2.051 ms/step figure
// came from collapse(2) over 36 layers x 8 heads and is not reproducible from this entry point.
// This is a correctness path that runs the benchmarked arithmetic, not a second benchmark.
//
// Returns 0 on success, negative on a rejected argument.
int csa_attend_external(const void *k_ptr, const void *v_ptr, const float *q,
                        const int *sel, int n_sel, int kv_heads, int group_size, int dim,
                        long long head_stride, long long token_stride,
                        float scale, int threads, float *out) {
    if (!k_ptr || !v_ptr || !q || !sel || !out) return -1;
    if (kv_heads < 1 || group_size < 1 || group_size > 8 || dim < 1) return -2;
    if (n_sel < 1) return -3;
    if (head_stride < 0 || token_stride < 1) return -4;
#if defined(__AVX512F__)
    // The vector path converts 16 bf16 lanes at a time; a tail loop would be a second
    // implementation of the inner loop, and head_dim is 128 everywhere this is used.
    if (dim % 16) return -5;
#endif
    if (threads > 0) omp_set_num_threads(threads);

    const bf16 *K = (const bf16 *)k_ptr;
    const bf16 *V = (const bf16 *)v_ptr;
    int alloc_failed = 0;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int head = 0; head < kv_heads; ++head) {
        // Per-thread scratch: group_size * n_sel floats. Allocated per call rather than kept
        // in the handle because this entry point is deliberately handle-free - the caller's
        // buffers are the state, and a hidden pool would be one more thing that could be stale.
        float *scores = (float *)malloc(sizeof(float) * (size_t)group_size * (size_t)n_sel);
        if (!scores) {
            alloc_failed = 1;
            continue;
        }
        attend_head_strided(K, V, q + (size_t)head * (size_t)group_size * (size_t)dim,
                            sel, n_sel, (size_t)token_stride,
                            (size_t)head * (size_t)head_stride, dim, group_size, scale,
                            scores, out + (size_t)head * (size_t)group_size * (size_t)dim);
        free(scores);
    }
    return alloc_failed ? -6 : 0;
}

int csa_selected_tokens(void *handle, int tokens_per_step) {
    CsaHandle *h = (CsaHandle *)handle;
    if (!h) return -1;
    return draw_units(h->sel, h->cfg.seq, h->cfg.granularity, tokens_per_step);
}

void csa_destroy(void *handle) {
    CsaHandle *h = (CsaHandle *)handle;
    if (!h) return;
    if (h->worker_started) {
        pthread_mutex_lock(&h->mu);
        h->shutdown = 1;
        pthread_cond_broadcast(&h->cv);
        pthread_mutex_unlock(&h->mu);
        pthread_join(h->worker, NULL);
    }
    free(h->k); free(h->v); free(h->q); free(h->sel); free(h->scores); free(h->out);
    free(h);
}
