/*
 * cpu_inference.c — Bit-packed AVX-512 scaffold + state-cached Mamba-2
 *
 * Key design choices:
 *   - Weights stay bit-packed (uint32). No int8 expansion → 8× less memory.
 *   - AVX-512 mask_add/mask_sub: 2 instructions per 16 elements, no quantization.
 *   - Scaffold weights: 3.2 MB/layer → fits in L3 cache.
 *   - State cached: h_state and conv_state persist between tokens.
 *   - OpenMP for row parallelism in scaffold and LoRA A.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC \
 *       -o libcpu_inference.so cpu_inference.c -lm
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#define D_MODEL     2048
#define D_INNER     4096
#define D_XBC       4352
#define D_STATE     128
#define NHEADS      64
#define HEADDIM     32
#define D_CONV      4
#define LORA_RANK   16
#define IN_PROJ_OUT 8512

/* ─── RMSNorm ─────────────────────────────────────────────────── */

static void rms_norm(const float* x, const float* w, float* out, int dim) {
    __m512 sum_sq = _mm512_setzero_ps();
    int i;
    for (i = 0; i + 16 <= dim; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        sum_sq = _mm512_fmadd_ps(v, v, sum_sq);
    }
    float ss = _mm512_reduce_add_ps(sum_sq);
    for (; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / dim + 1e-5f);

    __m512 vr = _mm512_set1_ps(inv_rms);
    for (i = 0; i + 16 <= dim; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        __m512 wv = _mm512_loadu_ps(&w[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(_mm512_mul_ps(v, wv), vr));
    }
    for (; i < dim; i++) out[i] = x[i] * w[i] * inv_rms;
}

/* ─── Bit-packed scaffold matvec (on-the-fly VNNI expansion) ──── */
/*
 * Weights stay bit-packed (256 bytes/row). For each group of 64 weights:
 *   1. Load 64 bits (8 bytes) of packed weights
 *   2. mask_blend → 64 int8 values (+1/-1) in 1 instruction
 *   3. VPDPBUSD: 64 uint8×int8 MACs in 1 instruction
 *
 * = VNNI speed + bit-packed memory. Best of both.
 * Input quantized to uint8 once per call (shared across all rows).
 */

static void quantize_input_uint8(
    const float* x, uint8_t* x_q, float* scale_out, int N
) {
    __m512 vmax = _mm512_setzero_ps();
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    int i;
    for (i = 0; i + 16 <= N; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign_mask, v));
    }
    float max_abs = _mm512_reduce_max_ps(vmax);
    for (; i < N; i++) { float a = fabsf(x[i]); if (a > max_abs) max_abs = a; }
    if (max_abs < 1e-8f) max_abs = 1e-8f;

    float s = 255.0f / (2.0f * max_abs);
    *scale_out = 1.0f / s;

    __m512 vs = _mm512_set1_ps(s);
    __m512 voff = _mm512_set1_ps(128.0f);
    __m512 vlo = _mm512_setzero_ps();
    __m512 vhi = _mm512_set1_ps(255.0f);
    for (i = 0; i + 16 <= N; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        __m512 q = _mm512_min_ps(vhi, _mm512_max_ps(vlo,
                   _mm512_add_ps(_mm512_mul_ps(v, vs), voff)));
        _mm_storeu_si128((__m128i*)&x_q[i], _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q)));
    }
    for (; i < N; i++) {
        float q = x[i] * s + 128.0f;
        if (q < 0) q = 0; if (q > 255) q = 255;
        x_q[i] = (uint8_t)(q + 0.5f);
    }
}

static void scaffold_matvec_avx512(
    const uint32_t* packed_w,   /* (out_features, packed_cols) bit-packed */
    const float* x,             /* (in_features,) float32 */
    const float* scale,         /* (out_features,) */
    const int* row_sums,        /* (out_features,) precomputed 2*popcount - in_features */
    float* output,              /* (out_features,) */
    int out_features,
    int packed_cols             /* in_features / 32 */
) {
    int in_features = packed_cols * 32;

    /* Quantize input once (shared by all rows) */
    uint8_t x_q[D_INNER > D_MODEL ? D_INNER : D_MODEL]
        __attribute__((aligned(64)));
    float x_scale;
    quantize_input_uint8(x, x_q, &x_scale, in_features);

    __m512i ones = _mm512_set1_epi8(1);
    __m512i neg_ones = _mm512_set1_epi8(-1);
    int vnni_pairs = packed_cols / 2;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < out_features; r++) {
        const uint32_t* row = packed_w + r * packed_cols;
        __m512i acc = _mm512_setzero_si512();

        for (int g = 0; g < vnni_pairs; g++) {
            uint64_t bits = *(const uint64_t*)(row + g * 2);
            __mmask64 mask = (__mmask64)bits;
            __m512i w_int8 = _mm512_mask_blend_epi8(mask, neg_ones, ones);
            __m512i xq = _mm512_loadu_si512((__m512i*)&x_q[g * 64]);
            acc = _mm512_dpbusd_epi32(acc, xq, w_int8);
        }

        int32_t tmp[16];
        _mm512_storeu_si512(tmp, acc);
        int dot = 0;
        for (int k = 0; k < 16; k++) dot += tmp[k];

        if (packed_cols & 1) {
            int g_last = packed_cols - 1;
            uint32_t bits = row[g_last];
            int base = g_last * 32;
            for (int b = 0; b < 32; b++) {
                int w = (bits & (1u << b)) ? +1 : -1;
                dot += (int)x_q[base + b] * w;
            }
        }

        float corrected = (float)(dot - 128 * row_sums[r]);
        output[r] = corrected * x_scale * scale[r];
    }
}

/* ─── Int8 LoRA matvec ────────────────────────────────────────── */

static void lora_matvec(
    const int8_t* A,        /* (out_features, rank) */
    const float* A_scale,   /* (out_features,) */
    const int8_t* B,        /* (rank, in_features) */
    const float* B_scale,   /* (rank,) */
    const float* x,         /* (in_features,) */
    float* output,          /* (out_features,) — ADDED to existing */
    int out_features,
    int in_features,
    int rank
) {
    /* Step 1: h = B @ x → (rank,) — small, single-threaded */
    float h[LORA_RANK];
    for (int r = 0; r < rank; r++) {
        const int8_t* Brow = B + r * in_features;
        __m512 acc = _mm512_setzero_ps();
        int j;
        for (j = 0; j + 16 <= in_features; j += 16) {
            /* Convert 16 int8 → 16 float */
            __m128i bytes = _mm_loadu_si128((__m128i*)&Brow[j]);
            __m512i ints = _mm512_cvtepi8_epi32(bytes);
            __m512 floats = _mm512_cvtepi32_ps(ints);
            __m512 xv = _mm512_loadu_ps(&x[j]);
            acc = _mm512_fmadd_ps(floats, xv, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; j < in_features; j++) sum += (float)Brow[j] * x[j];
        h[r] = sum * B_scale[r];
    }

    /* Step 2: output += A @ h → (out_features,) — parallelized */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < out_features; r++) {
        const int8_t* Arow = A + r * rank;
        float acc = 0;
        for (int j = 0; j < rank; j++)
            acc += (float)Arow[j] * h[j];
        output[r] += acc * A_scale[r];
    }
}

/* ─── SiLU ────────────────────────────────────────────────────── */

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* ─── Conv1d shift register ───────────────────────────────────── */

static void conv_step(
    float* conv_state,      /* (D_XBC, 3) */
    float* xBC,             /* (D_XBC,) in/out */
    const float* conv_w,    /* (D_XBC, 4) */
    const float* conv_b     /* (D_XBC,) */
) {
    for (int ch = 0; ch < D_XBC; ch++) {
        float s0 = conv_state[ch * 3 + 0];
        float s1 = conv_state[ch * 3 + 1];
        float s2 = conv_state[ch * 3 + 2];
        float xn = xBC[ch];

        conv_state[ch * 3 + 0] = s1;
        conv_state[ch * 3 + 1] = s2;
        conv_state[ch * 3 + 2] = xn;

        float out = s0 * conv_w[ch * D_CONV + 0]
                  + s1 * conv_w[ch * D_CONV + 1]
                  + s2 * conv_w[ch * D_CONV + 2]
                  + xn * conv_w[ch * D_CONV + 3]
                  + conv_b[ch];
        xBC[ch] = silu_f(out);
    }
}

/* ─── SSM step ────────────────────────────────────────────────── */

static void ssm_step(
    float* h_state,         /* (NHEADS, HEADDIM, D_STATE) */
    const float* x_ssm,     /* (D_INNER,) */
    const float* B_param,   /* (D_STATE,) */
    const float* C_param,   /* (D_STATE,) */
    const float* dt,        /* (NHEADS,) */
    const float* A_log,     /* (NHEADS,) */
    const float* D_param,   /* (NHEADS,) */
    float* y_out            /* (D_INNER,) */
) {
    #pragma omp parallel for schedule(static)
    for (int head = 0; head < NHEADS; head++) {
        float A = -expf(A_log[head]);
        float dt_v = dt[head];
        float A_bar = expf(dt_v * A);

        for (int p = 0; p < HEADDIM; p++) {
            float xval = x_ssm[head * HEADDIM + p];
            float X = xval * dt_v;
            float* h_ptr = h_state + (head * HEADDIM + p) * D_STATE;
            float y_acc = 0;

            for (int n = 0; n < D_STATE; n++) {
                float h = h_ptr[n];
                h = A_bar * h + X * B_param[n];
                h_ptr[n] = h;
                y_acc += C_param[n] * h;
            }
            y_out[head * HEADDIM + p] = y_acc + xval * D_param[head];
        }
    }
}

/* ─── Gate + RMSNorm ──────────────────────────────────────────── */

static void gate_norm(
    const float* y, const float* z, const float* w, float* out, int dim
) {
    float ss = 0;
    for (int i = 0; i < dim; i++) {
        float g = y[i] * silu_f(z[i]);
        out[i] = g;
        ss += g * g;
    }
    float inv_rms = 1.0f / sqrtf(ss / dim + 1e-5f);
    for (int i = 0; i < dim; i++)
        out[i] = out[i] * inv_rms * w[i];
}

/* ─── Softplus ────────────────────────────────────────────────── */

static inline float softplus_f(float x) { return log1pf(expf(x)); }

/* ─── Precompute row sums for VNNI bias correction ────────────── */

void precompute_row_sums(
    const uint32_t* packed_w,
    int* row_sums,
    int out_features,
    int packed_cols
) {
    for (int r = 0; r < out_features; r++) {
        const uint32_t* row = packed_w + r * packed_cols;
        int popcnt = 0;
        for (int g = 0; g < packed_cols; g++)
            popcnt += __builtin_popcount(row[g]);
        row_sums[r] = 2 * popcnt - packed_cols * 32;
    }
}

/* ─── Fused layer forward ─────────────────────────────────────── */

void layer_forward(
    float* x,
    const uint32_t* in_packed, const float* in_scale, const int* in_row_sums,
    const uint32_t* out_packed, const float* out_scale, const int* out_row_sums,
    const int8_t* in_lora_A,       /* (IN_PROJ_OUT, LORA_RANK) */
    const float* in_lora_A_s,      /* (IN_PROJ_OUT,) */
    const int8_t* in_lora_B,       /* (LORA_RANK, D_MODEL) */
    const float* in_lora_B_s,      /* (LORA_RANK,) */
    const int8_t* out_lora_A,      /* (D_MODEL, LORA_RANK) */
    const float* out_lora_A_s,     /* (D_MODEL,) */
    const int8_t* out_lora_B,      /* (LORA_RANK, D_INNER) */
    const float* out_lora_B_s,     /* (LORA_RANK,) */
    const float* conv_w,           /* (D_XBC, D_CONV) */
    const float* conv_b,           /* (D_XBC,) */
    const float* A_log,            /* (NHEADS,) */
    const float* D_param,          /* (NHEADS,) */
    const float* dt_bias,          /* (NHEADS,) */
    const float* norm_weight,      /* (D_INNER,) */
    const float* ln_weight,        /* (D_MODEL,) */
    float* h_state,                /* (NHEADS, HEADDIM, D_STATE) */
    float* conv_state,             /* (D_XBC, 3) */
    /* scratch */
    float* x_norm,                 /* (D_MODEL,) */
    float* proj,                   /* (IN_PROJ_OUT,) */
    float* y_buf,                  /* (D_INNER,) */
    float* out_buf                 /* (D_MODEL,) */
) {
    /* 1. RMSNorm */
    rms_norm(x, ln_weight, x_norm, D_MODEL);

    /* 2. in_proj: scaffold + LoRA */
    scaffold_matvec_avx512(in_packed, x_norm, in_scale, in_row_sums, proj,
                           IN_PROJ_OUT, D_MODEL / 32);
    lora_matvec(in_lora_A, in_lora_A_s, in_lora_B, in_lora_B_s,
                x_norm, proj, IN_PROJ_OUT, D_MODEL, LORA_RANK);

    /* 3. Split */
    float* z_ptr = proj;
    float* xBC_ptr = proj + D_INNER;
    float* dt_raw = proj + D_INNER + D_XBC;

    /* 4. Conv step */
    conv_step(conv_state, xBC_ptr, conv_w, conv_b);

    /* 5. Split xBC */
    float* x_ssm = xBC_ptr;
    float* B_param = xBC_ptr + D_INNER;
    float* C_param = xBC_ptr + D_INNER + D_STATE;

    /* 6. Softplus dt */
    float dt[NHEADS];
    for (int i = 0; i < NHEADS; i++)
        dt[i] = softplus_f(dt_raw[i] + dt_bias[i]);

    /* 7. SSM step */
    ssm_step(h_state, x_ssm, B_param, C_param, dt, A_log, D_param, y_buf);

    /* 8. Gate + norm */
    float gated[D_INNER];
    gate_norm(y_buf, z_ptr, norm_weight, gated, D_INNER);

    /* 9. out_proj: scaffold + LoRA */
    scaffold_matvec_avx512(out_packed, gated, out_scale, out_row_sums, out_buf,
                           D_MODEL, D_INNER / 32);
    lora_matvec(out_lora_A, out_lora_A_s, out_lora_B, out_lora_B_s,
                gated, out_buf, D_MODEL, D_INNER, LORA_RANK);

    /* 10. Residual */
    for (int i = 0; i + 16 <= D_MODEL; i += 16) {
        __m512 xv = _mm512_loadu_ps(&x[i]);
        __m512 ov = _mm512_loadu_ps(&out_buf[i]);
        _mm512_storeu_ps(&x[i], _mm512_add_ps(xv, ov));
    }
}

/* ─── Final norm ──────────────────────────────────────────────── */

void final_rms_norm(const float* x, const float* w, float* out, int dim) {
    rms_norm(x, w, out, dim);
}

/* ─── Embedding lookup ────────────────────────────────────────── */

void embedding_lookup(const float* emb, int tok, float* out, int d) {
    memcpy(out, emb + tok * d, d * sizeof(float));
}
