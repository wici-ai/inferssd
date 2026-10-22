#pragma once

#include <cstdint>
// Task 60 targets the x86-64 production oracle (glibc 2.35, AVX-512).
// Keep these routines opt-in and limited to Kimi KDA shapes.
static inline bool ggml_cuda_kimi_kda_cpu_parity_enabled(const ggml_tensor * node) {
    return node != nullptr && (node->flags & GGML_TENSOR_FLAG_KIMI_KDA_CPU_PARITY) != 0;
}

static inline bool ggml_cuda_kimi_kda_unary_shape(
        const ggml_tensor * node,
        const ggml_tensor * tensor) {
    if (!ggml_cuda_kimi_kda_cpu_parity_enabled(node) || tensor->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t elements = ggml_nelements(tensor);
    return elements == 96 || elements == 128*96 ||
        (tensor->ne[0] == 7168 && elements <= 3*7168);
}

// -use_fast_math enables FTZ for float arithmetic.  Reconstruct subnormal
// results through their IEEE-754 payload so the strict path keeps the x86
// oracle's gradual underflow behavior.
static __device__ __forceinline__ float ggml_cuda_kimi_f32_from_double_rn(double value) {
    const uint64_t bits = (uint64_t) __double_as_longlong(value);
    const uint32_t sign = (uint32_t) (bits >> 32) & 0x80000000U;
    const double magnitude = __longlong_as_double((long long) (bits & 0x7fffffffffffffffULL));

    if (magnitude == 0.0) {
        return __uint_as_float(sign);
    }
    if (magnitude >= 0x1p-126) {
        return __double2float_rn(value);
    }

    const uint32_t mantissa = __double2uint_rn(__dmul_rn(magnitude, 0x1p149));
    if (mantissa >= 0x00800000U) {
        return __uint_as_float(sign | 0x00800000U);
    }
    return __uint_as_float(sign | mantissa);
}

static __device__ __forceinline__ float ggml_cuda_kimi_mul_f32_cpu(float x, float y) {
    const uint32_t xbits = __float_as_uint(x);
    const uint32_t ybits = __float_as_uint(y);
    const uint32_t xexp = (xbits >> 23) & 0xffU;
    const uint32_t yexp = (ybits >> 23) & 0xffU;
    const uint32_t xmantissa = xbits & 0x007fffffU;
    const uint32_t ymantissa = ybits & 0x007fffffU;

    if (xexp == 0xffU || yexp == 0xffU ||
            (xexp == 0 && xmantissa == 0) || (yexp == 0 && ymantissa == 0)) {
        return __fmul_rn(x, y);
    }

    // A product of two normal floats cannot underflow when the sum of their
    // biased exponents is at least 129.  Keep that overwhelmingly common path
    // in float; use exact double arithmetic when either input is subnormal or
    // the correctly rounded result can be subnormal.
    if (xexp != 0 && yexp != 0 && xexp + yexp >= 129U) {
        return __fmul_rn(x, y);
    }

    double xd = (double) x;
    if (xexp == 0) {
        xd = __dmul_rn((double) xmantissa, 0x1p-149);
        if ((xbits & 0x80000000U) != 0) {
            xd = -xd;
        }
    }
    double yd = (double) y;
    if (yexp == 0) {
        yd = __dmul_rn((double) ymantissa, 0x1p-149);
        if ((ybits & 0x80000000U) != 0) {
            yd = -yd;
        }
    }
    return ggml_cuda_kimi_f32_from_double_rn(__dmul_rn(xd, yd));
}

// Values used by the glibc 2.35 x86 FMA expf implementation.  The operation
// sequence below is written explicitly so nvcc cannot contract or reassociate
// it differently from the scalar CPU oracle.
static __device__ __constant__ uint64_t ggml_cuda_kimi_expf_table[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL,
    0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL,
    0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL,
    0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL,
    0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL,
    0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

static __device__ __forceinline__ float ggml_cuda_kimi_expf_cpu(float x) {
    const uint32_t xbits = __float_as_uint(x);
    if (xbits == 0xff800000U) {
        return 0.0f;
    }
    if ((xbits & 0x7fffffffU) >= 0x7f800000U) {
        return x + x;
    }
    if (x > 0x1.62e42ep+6f) {
        return __uint_as_float(0x7f800000U);
    }
    if (x < -0x1.9fe368p+6f) {
        return 0.0f;
    }

    constexpr double inv_ln2_scaled = 0x1.71547652b82fep+5;
    constexpr double shift          = 0x1.8p+52;
    constexpr double c0             = 0x1.c6af84b912394p-20;
    constexpr double c1             = 0x1.ebfce50fac4f3p-13;
    constexpr double c2             = 0x1.62e42ff0c52d6p-6;

    const double xd = (double) x;
    const double shifted = __fma_rn(inv_ln2_scaled, xd, shift);
    const uint64_t ki = (uint64_t) __double_as_longlong(shifted);
    const double kd = __dsub_rn(shifted, shift);
    const double r = __fma_rn(inv_ln2_scaled, xd, -kd);

    uint64_t bits = ggml_cuda_kimi_expf_table[ki & 31U];
    bits += ki << 47;
    const double scale = __longlong_as_double((long long) bits);

    const double z  = __fma_rn(c0, r, c1);
    const double r2 = __dmul_rn(r, r);
    const double y0 = __fma_rn(c2, r, 1.0);
    const double y1 = __fma_rn(z, r2, y0);
    return ggml_cuda_kimi_f32_from_double_rn(__dmul_rn(y1, scale));
}

// Scalar transcription of ggml's AVX-512 exp approximation used by SiLU.
static __device__ __forceinline__ float ggml_cuda_kimi_expf_avx512(float x) {
    constexpr float rounder = 0x1.8p23f;
    const float z = __fmaf_rn(x, 0x1.715476p+0f, rounder);
    const float n = __fsub_rn(z, rounder);
    const float b0 = __fmaf_rn(-n, 0x1.62e4p-1f, x);
    const float b  = __fmaf_rn(-n, 0x1.7f7d1cp-20f, b0);
    const float u  = __fmul_rn(b, b);

    const float p0 = __fmaf_rn(0x1.0e4020p-7f, b, 0x1.573e2ep-5f);
    const float p1 = __fmaf_rn(0x1.555e66p-3f, b, 0x1.fffdb6p-2f);
    const float p2 = __fmaf_rn(p0, u, p1);
    const float p3 = __fmaf_rn(0x1.ffffecp-1f, b, 1.0f);
    const float polynomial = __fmaf_rn(p2, u, p3);

    const int exponent = __float2int_rz(n);
    uint32_t bits = __float_as_uint(polynomial);
    const int scaled_exponent = (int) ((bits >> 23) & 0xffU) + exponent;
    if (scaled_exponent > 0 && scaled_exponent < 0xff) {
        bits = (bits & 0x807fffffU) | ((uint32_t) scaled_exponent << 23);
        return __uint_as_float(bits);
    }
    return scalbnf(polynomial, exponent);
}

static __device__ __forceinline__ float ggml_cuda_kimi_silu_cpu(float x) {
    const float denominator = __fadd_rn(1.0f, ggml_cuda_kimi_expf_avx512(-x));
    return __fdiv_rn(x, denominator);
}

static __device__ __forceinline__ float ggml_cuda_kimi_sigmoid_cpu(float x) {
    const float denominator = __fadd_rn(1.0f, ggml_cuda_kimi_expf_cpu(-x));
    if ((__float_as_uint(denominator) & 0x7fffffffU) == 0x7f800000U) {
        return 0.0f;
    }
    if (denominator > 0x1p126f) {
        return ggml_cuda_kimi_f32_from_double_rn(__ddiv_rn(1.0, (double) denominator));
    }
    return __fdiv_rn(1.0f, denominator);
}

// Match ggml_vec_dot_f32 for n == 128 on the AVX-512 production CPU.  Four
// vector accumulators cover each 64-element step; the final reduction mirrors
// GCC's expansion of _mm512_reduce_add_ps.
static __device__ __forceinline__ float ggml_cuda_kimi_dot_f32_avx512_128(
        const float * x, const float * y) {
    float lanes[16];

#pragma unroll
    for (int lane = 0; lane < 16; ++lane) {
        float sum0 = __fmaf_rn(x[     lane], y[     lane], 0.0f);
        float sum1 = __fmaf_rn(x[16 + lane], y[16 + lane], 0.0f);
        float sum2 = __fmaf_rn(x[32 + lane], y[32 + lane], 0.0f);
        float sum3 = __fmaf_rn(x[48 + lane], y[48 + lane], 0.0f);

        sum0 = __fmaf_rn(x[64 + lane], y[64 + lane], sum0);
        sum1 = __fmaf_rn(x[80 + lane], y[80 + lane], sum1);
        sum2 = __fmaf_rn(x[96 + lane], y[96 + lane], sum2);
        sum3 = __fmaf_rn(x[112 + lane], y[112 + lane], sum3);

        sum0 = __fadd_rn(sum0, sum2);
        sum1 = __fadd_rn(sum1, sum3);
        lanes[lane] = __fadd_rn(sum0, sum1);
    }

#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        lanes[lane] = __fadd_rn(lanes[lane], lanes[lane + 8]);
    }
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        lanes[lane] = __fadd_rn(lanes[lane], lanes[lane + 4]);
    }

    const float pair0 = __fadd_rn(lanes[0], lanes[2]);
    const float pair1 = __fadd_rn(lanes[1], lanes[3]);
    return __fadd_rn(pair0, pair1);
}
