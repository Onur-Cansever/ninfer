#include "ops/common/host/operands.h"
#include "ninfer/ops/indexing.h"
#include <stdexcept>
#include "ops/common/numeric/math.h"
#include "ops/common/host/launch.h"
#include "core/device.h"
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>

namespace ninfer::ops {
using namespace common;

namespace {

// Packed feature split geometries: source rows are the concatenation of four
// feature segments (q, k, g, v row counts) and each output is viewed as
// [head_dim=256, heads, tokens].
//
//   27B MoE attention: 14336 = 6144 + 1024 + 6144 + 1024 (24/4 heads)
//   9B MTP attention:  10240 = 4096 + 1024 + 4096 + 1024 (16/4 heads)
struct SplitGeometry {
    int a, b, c, d;
};

constexpr SplitGeometry geometry27b = {6144, 1024, 6144, 1024};
constexpr SplitGeometry geometry9b  = {4096, 1024, 4096, 1024};

__global__ void concat_bf16_features_kernel(const __nv_bfloat16* first, const __nv_bfloat16* second,
                                            __nv_bfloat16* out, std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = first[in_idx];
    out[out_base + rows + row]  = second[in_idx];
}

template <int kA, int kB, int kC, int kD>
__global__ void split_bf16_features_kernel(const __nv_bfloat16* source, __nv_bfloat16* first,
                                           __nv_bfloat16* second, __nv_bfloat16* third,
                                           __nv_bfloat16* fourth, std::int32_t tokens) {
    constexpr int kSplitRows = kA + kB + kC + kD;
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(kSplitRows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % kSplitRows);
    const int token           = static_cast<int>(idx / kSplitRows);
    const __nv_bfloat16 value = source[idx];

    if (row < kA) {
        first[static_cast<std::int64_t>(token) * kA + row] = value;
        return;
    }
    if (row < kA + kB) {
        const int local                                             = row - kA;
        second[static_cast<std::int64_t>(token) * kB + local] = value;
        return;
    }
    if (row < kA + kB + kC) {
        const int local                                        = row - kA - kB;
        third[static_cast<std::int64_t>(token) * kC + local] = value;
        return;
    }

    const int local = row - kA - kB - kC;
    fourth[static_cast<std::int64_t>(token) * kD + local] = value;
}

void concat_bf16_features_launch(const Tensor& first, const Tensor& second, Tensor& out,
                                 cudaStream_t stream) {
    constexpr int kBlock = 256;
    for_each_token_slice(first.ne[1], 1, [&](int token_offset, int token_count) {
        const Tensor first_slice  = first.slice(1, token_offset, token_count);
        const Tensor second_slice = second.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(1, token_offset, token_count);
        const dim3 grid(static_cast<unsigned int>(div_up(first.ne[0], kBlock)),
                        static_cast<unsigned int>(token_count));
        concat_bf16_features_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(first_slice.data),
            static_cast<const __nv_bfloat16*>(second_slice.data),
            static_cast<__nv_bfloat16*>(out_slice.data), first.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    });
}

void split_bf16_features_launch(const Tensor& source, Tensor& first, Tensor& second, Tensor& third,
                                Tensor& fourth, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int64_t n = static_cast<std::int64_t>(source.ne[0]) * source.ne[1];
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
    if (source.ne[0] == 14336) {
        split_bf16_features_kernel<6144, 1024, 6144, 1024><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data), static_cast<__nv_bfloat16*>(first.data),
            static_cast<__nv_bfloat16*>(second.data), static_cast<__nv_bfloat16*>(third.data),
            static_cast<__nv_bfloat16*>(fourth.data), source.ne[1]);
    } else if (source.ne[0] == 10240) {
        split_bf16_features_kernel<4096, 1024, 4096, 1024><<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(source.data), static_cast<__nv_bfloat16*>(first.data),
            static_cast<__nv_bfloat16*>(second.data), static_cast<__nv_bfloat16*>(third.data),
            static_cast<__nv_bfloat16*>(fourth.data), source.ne[1]);
    } else {
        throw std::invalid_argument("split_bf16_features: unsupported source rows " +
                                    std::to_string(source.ne[0]));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void concat_bf16_features(const Tensor& first, const Tensor& second, Tensor& out,
                          CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "concat_bf16_features";
    common::operand::dtype(first, DType::BF16, op, "first");
    common::operand::contiguous_nonnull(first, op, "first", execution.device());
    common::operand::dtype(second, DType::BF16, op, "second");
    common::operand::contiguous_nonnull(second, op, "second", execution.device());
    common::operand::dtype(out, DType::BF16, op, "out");
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    const std::int32_t rows   = first.ne[0];
    const std::int32_t tokens = first.ne[1];
    if (rows <= 0) { throw std::invalid_argument("concat_bf16_features: D must be positive"); }
    if (tokens <= 0) { throw std::invalid_argument("concat_bf16_features: T must be positive"); }
    common::operand::shape(first, {rows, tokens, 1, 1}, op, "first");
    common::operand::shape(second, {rows, tokens, 1, 1}, op, "second");
    common::operand::shape(out, {2 * rows, tokens, 1, 1}, op, "out");

    concat_bf16_features_launch(first, second, out, stream);
}

void split_bf16_features(const Tensor& source, Tensor& first, Tensor& second, Tensor& third,
                         Tensor& fourth, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    constexpr const char* op = "split_bf16_features";
    common::operand::dtype(source, DType::BF16, op, "source");
    common::operand::contiguous_nonnull(source, op, "source", execution.device());
    common::operand::dtype(first, DType::BF16, op, "first");
    common::operand::contiguous_nonnull(first, op, "first", execution.device());
    common::operand::dtype(second, DType::BF16, op, "second");
    common::operand::contiguous_nonnull(second, op, "second", execution.device());
    common::operand::dtype(third, DType::BF16, op, "third");
    common::operand::contiguous_nonnull(third, op, "third", execution.device());
    common::operand::dtype(fourth, DType::BF16, op, "fourth");
    common::operand::contiguous_nonnull(fourth, op, "fourth", execution.device());
    const std::int32_t tokens = source.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("split_bf16_features: T must be positive"); }
    if (source.ne[0] != 14336 && source.ne[0] != 10240) {
        throw std::invalid_argument("split_bf16_features: unsupported source rows " +
                                    std::to_string(source.ne[0]));
    }
    const SplitGeometry geometry = (source.ne[0] == 10240) ? geometry9b : geometry27b;
    common::operand::shape(source, {static_cast<std::int32_t>(source.ne[0]), tokens, 1, 1}, op, "source");
    const std::array<Tensor*, 4> outputs = {&first, &second, &third, &fourth};
    const std::array<int, 4> segment_rows = {geometry.a, geometry.b, geometry.c, geometry.d};
    constexpr const char* names[4] = {"first", "second", "third", "fourth"};
    for (int i = 0; i < 4; ++i) {
        const std::int32_t heads = segment_rows[i] / 256;
        std::array<std::int32_t, 4> expected{256, heads, tokens, 1};
        if (outputs[i]->ne[0] != 256 || outputs[i]->ne[1] != heads || outputs[i]->ne[2] != tokens ||
            outputs[i]->ne[3] != 1) {
            throw std::invalid_argument(std::string("split_bf16_features: invalid shape for ") + names[i]);
        }
    }

    split_bf16_features_launch(source, first, second, third, fourth, stream);
}

} // namespace ninfer::ops
