#include "dsv4-hc.cuh"

// Fused DeepSeek-V4 hyper-connection kernels, ported from vcruz305/llama.cpp@5210c7c (dsv4-hc.cu).
// The plain-op graph spends one tiny kernel per add/div/sum_rows on 4x4 matrices; these do a whole
// Sinkhorn or a whole expansion per token (comb) or per output element (post).

#define CUDA_DSV4_HC_BLOCK_SIZE 256

static constexpr int DSV4_HC = 4;

static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_tokens) {
        return;
    }

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;
    if (ir >= nr) {
        return;
    }

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    const int64_t n_tokens = mixes->ne[1];
    const float   eps      = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter   = ggml_get_op_params_i32(dst, 1);

    const dim3 block_dims(CUDA_DSV4_HC_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((unsigned int) ((n_tokens + CUDA_DSV4_HC_BLOCK_SIZE - 1) / CUDA_DSV4_HC_BLOCK_SIZE), 1, 1);

    dsv4_hc_comb_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            mixes->nb[0] / sizeof(float), mixes->nb[1] / sizeof(float),
            scale->nb[0] / sizeof(float),
            base->nb[0] / sizeof(float),
            dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(CUDA_DSV4_HC_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((unsigned int) ((nr + CUDA_DSV4_HC_BLOCK_SIZE - 1) / CUDA_DSV4_HC_BLOCK_SIZE), 1, 1);

    dsv4_hc_post_f32<<<block_nums, block_dims, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            x->nb[0] / sizeof(float), x->nb[1] / sizeof(float),
            residual->nb[0] / sizeof(float), residual->nb[1] / sizeof(float), residual->nb[2] / sizeof(float),
            post->nb[0] / sizeof(float), post->nb[1] / sizeof(float),
            comb->nb[0] / sizeof(float), comb->nb[1] / sizeof(float), comb->nb[2] / sizeof(float),
            dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float));
}
