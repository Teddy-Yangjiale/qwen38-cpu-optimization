// Standalone check of the CPU repack MUL_MAT_ID path: runs one mul_mat_id with Q4_K experts held in the
// repack buffer type and prints the output checksum plus a few values. Run twice, once with
// GGML_REPACK_MMID_GEMV=1 (one gemv per token, original behaviour) and once without (grouped gemm), and
// compare the dumped outputs with compare_dumps.py. Shapes use model K/M dimensions (2560/640) and
// 10 selected experts; the expert count defaults to 16 for quick boundary tests and is configurable.
//
// build (against a configured llama.cpp build dir B):
//   g++ -O2 -std=c++17 test_mmid_repack.cpp -I$S/ggml/include -L$B/bin -lggml -lggml-base -lggml-cpu -o test_mmid_repack
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const int n_tokens = argc > 1 ? atoi(argv[1]) : 64;
    const char * dump  = argc > 2 ? argv[2] : "mmid_out.bin";
    const int n_embd = 2560, n_ff = 640, n_used = 10;
    const int n_threads = argc > 3 ? atoi(argv[3]) : 8;
    const bool use_repack = !(argc > 4 && atoi(argv[4]) == 0);
    const int n_expert = argc > 5 ? atoi(argv[5]) : 16;
    const int repeats = argc > 6 ? atoi(argv[6]) : 0;
    const int input_rows = argc > 7 ? atoi(argv[7]) : 1;
    const std::string routing = argc > 8 ? argv[8] : "hotspot";

    if (n_tokens <= 0 || n_tokens > 1024 || n_threads <= 0 || n_threads > 38 ||
        n_expert < n_used || n_expert > 512 || repeats < 0 || repeats > 20 ||
        (input_rows != 1 && input_rows != n_used) ||
        (routing != "hotspot" && routing != "balanced")) {
        fprintf(stderr, "invalid test dimensions: tokens=%d threads=%d experts=%d (minimum experts=%d)\n",
                n_tokens, n_threads, n_expert, n_used);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    // weight buffer type: the CPU repack extra buffer (what llama.cpp uses for Q4_K on x86) or plain CPU
    ggml_backend_buffer_type_t wbuft = ggml_backend_cpu_buffer_type();
    if (use_repack) {
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        ggml_backend_buffer_type_t * extra = fn ? fn(dev) : nullptr;
        for (ggml_backend_buffer_type_t * candidate = extra; candidate && *candidate; ++candidate) {
            const char * name = ggml_backend_buft_name(*candidate);
            if (name && strcmp(name, "CPU_REPACK") == 0) {
                wbuft = *candidate;
                break;
            }
        }
        if (wbuft == ggml_backend_cpu_buffer_type()) {
            fprintf(stderr, "ERROR: CPU_REPACK buffer type is unavailable; refusing a non-repack fallback\n");
            ggml_backend_free(backend);
            return 2;
        }
        printf("weights in extra buffer type: %s\n", ggml_backend_buft_name(wbuft));
    }

    // ---- weights: Q4_K [n_embd, n_ff, n_expert], deterministic pseudo-random ----
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, n_embd, n_ff, n_expert);
    ggml_set_name(w, "ffn_up_exps");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, wbuft);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 0.5f);
        const int64_t n = (int64_t) n_embd * n_ff * n_expert;
        std::vector<float> f(n);
        for (auto & v : f) v = nd(rng);
        std::vector<uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), q.data(), 0, n_ff * n_expert, n_embd, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size()); // repack happens inside set_tensor for the repack buffer
    }

    // ---- graph: mul_mat_id(w, x[n_embd,1,n_tokens], ids[n_used,n_tokens]) ----
    ggml_init_params gp = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * x   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, input_rows, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * out = ggml_mul_mat_id(gctx, w, x, ids); // [n_ff, n_used, n_tokens]
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, backend);
    (void) gbuf;

    {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xf((size_t) n_embd * input_rows * n_tokens);
        for (auto & v : xf) v = nd(rng);
        ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
        // Routing forces experts 0..3 to receive every token, covering 1/2/3-row tails and 4-row groups;
        // remaining selections are spread over the other experts.
        std::vector<int32_t> id((size_t) n_used * n_tokens);
        std::uniform_int_distribution<int> ud(4, n_expert - 1);
        for (int t = 0; t < n_tokens; ++t) {
            for (int k = 0; k < n_used; ++k) {
                int e = routing == "balanced" ? (t * n_used + k) % n_expert : ((k < 4) ? k : ud(rng));
                for (int j = 0; j < k; ++j) if (id[(size_t) t * n_used + j] == e) { e = (e + 1) % n_expert; j = -1; }
                id[(size_t) t * n_used + k] = e;
            }
        }
        ggml_backend_tensor_set(ids, id.data(), 0, id.size() * sizeof(int32_t));
        std::vector<int> counts(n_expert, 0);
        for (int e : id) ++counts[e];
        int groups = 0, tails = 0, maximum = 0;
        for (int count : counts) {
            groups += count / 4;
            tails += count % 4;
            if (count > maximum) maximum = count;
        }
        printf("ROUTING {\"kind\":\"%s\",\"input_rows\":%d,\"groups4\":%d,\"tail_rows\":%d,\"maximum_rows_per_expert\":%d}\n",
               routing.c_str(), input_rows, groups, tails, maximum);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
    ggml_backend_synchronize(backend);
    std::vector<int64_t> samples_ns;
    for (int rep = 0; rep < repeats; ++rep) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_synchronize(backend);
        const auto end = std::chrono::steady_clock::now();
        samples_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }
    if (repeats) {
        printf("TIMING {\"scope\":\"resident_synthetic_graph_compute_and_sync\",\"warmup\":1,\"threads\":%d,\"tokens\":%d,\"experts\":%d,\"samples_ns\":[",
               n_threads, n_tokens, n_expert);
        for (size_t i = 0; i < samples_ns.size(); ++i) printf("%s%lld", i ? "," : "", (long long) samples_ns[i]);
        printf("]}\n");
    }

    std::vector<float> o(ggml_nelements(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size() * sizeof(float));
    double sum = 0, asum = 0; int nans = 0;
    for (float v : o) { if (!std::isfinite(v)) nans++; sum += v; asum += fabs(v); }
    printf("n_tokens=%d n_expert=%d threads=%d gemv_forced=%s elems=%zu sum=%.6f abs_sum=%.6f non_finite=%d first=[%g %g %g %g]\n",
           n_tokens, n_expert, n_threads, getenv("GGML_REPACK_MMID_GEMV") ? getenv("GGML_REPACK_MMID_GEMV") : "unset",
           o.size(), sum, asum, nans, o[0], o[1], o[2], o[3]);
    FILE * fp = fopen(dump, "wb");
    if (!fp) { fprintf(stderr, "failed opening output dump %s\n", dump); return 3; }
    const size_t written = fwrite(o.data(), sizeof(float), o.size(), fp);
    const int close_status = fclose(fp);
    if (written != o.size() || close_status != 0) {
        fprintf(stderr, "failed writing complete output dump %s\n", dump);
        return 3;
    }
    printf("dumped %s\n", dump);

    ggml_backend_buffer_free(gbuf); ggml_backend_buffer_free(wbuf);
    ggml_free(wctx); ggml_free(gctx); ggml_backend_free(backend);
    return nans ? 2 : 0;
}
