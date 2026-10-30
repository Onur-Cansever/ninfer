#include "ops/common/numeric/math.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "core/weight.h"
#include "ops/linear/swiglu/q4/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// Shape-templated geometry so one kernel set serves every registered gate_up
// shape (27B 34816x5120 and 9B 24576x4096).
template <int N, int K>
struct Q4SwiGluShape {
    static constexpr int kN                     = N;
    static constexpr int kK                     = K;
    static constexpr int kIntermediate          = kN / 2;
    static constexpr int kGroupK                = 64;
    static constexpr int kGroups                = kK / kGroupK;
    static constexpr int kBytesPerGroup         = 32;
    static constexpr int kVecBytes              = 16;
    static constexpr int kGroupsPerWarpTile     = 16;
    static constexpr int kVecsPerWarpTile       = kGroupsPerWarpTile * kBytesPerGroup / kVecBytes;
    static constexpr int kWarpsPerBlock         = 4;
    static constexpr int kBlockThreads          = kWarpsPerBlock * 32;
    static constexpr int kPairsPerBlock         = kWarpsPerBlock;
    static constexpr int kXVecs                 = kK / 8; // x as uint4 (8 bf16 each)
    static constexpr int kTiles                 = kGroups / kGroupsPerWarpTile;
    static_assert(kIntermediate % kPairsPerBlock == 0);
    static_assert(kBytesPerGroup == 2 * kVecBytes);
    static_assert(kGroups % kGroupsPerWarpTile == 0);
    static_assert(kVecsPerWarpTile == 32);
};

using Q4SwiGlu27B = Q4SwiGluShape<34816, 5120>;
using Q4SwiGlu9B  = Q4SwiGluShape<24576, 4096>;

template <int kIntermediate>
struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __host__ __device__ int output_rows(int rows) const { return rows / 2; }

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 projected, int rows,
                                                   int token_end) const {
        if (row >= rows / 2) return;
        if (token < token_end) output.store(row, token, common::silu(projected.x) * projected.z);
        if (token + 1 < token_end)
            output.store(row, token + 1, common::silu(projected.y) * projected.w);
    }
};

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Shape>
struct Q4SwiGluSmallT {
    template <int ActiveCols>
    static void launch_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
        // Wider tiles are shared-memory limited before six CTAs can be resident.
        // Match launch bounds to that limit instead of forcing register spills.
        constexpr int kMinBlocks = ActiveCols <= 8    ? 6
                               : ActiveCols <= 16 ? 4
                               : ActiveCols <= 24 ? 3
                                                  : 2;
        using Schedule =
            Q4A16SlicedKMmaSchedule<16, (ActiveCols + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                    kMinBlocks, Shape::kK, ActiveCols, Q4SlicedKReduction::Pairwise>;
        launch_q4_a16_sliced_k_mma<Schedule>(
            q4_linear_operands(x, w),
            LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Shape::kIntermediate},
            Q4SwiGluSmallTEpilogue{}, stream, Q4SwiGluSmallTRows<Shape::kIntermediate>{});
    }

    template <std::size_t... Offsets>
    static constexpr auto make_launchers(std::index_sequence<Offsets...>) {
        return std::array<SmallTLauncher, sizeof...(Offsets)>{
            &launch_active<8 * (1 + static_cast<int>(Offsets))>...};
    }

    static constexpr auto kLaunchers = make_launchers(std::make_index_sequence<4>{});
};

template <class Shape>
__device__ __forceinline__ void q4_issue_pair_tile(
    uint4 (*__restrict__ s_code)[Shape::kVecsPerWarpTile], uint4 (*__restrict__ s_scale)[2],
    const std::uint8_t* __restrict__ gate_code_row, const std::uint8_t* __restrict__ gate_scale_row,
    const std::uint8_t* __restrict__ up_code_row, const std::uint8_t* __restrict__ up_scale_row,
    int tile, int lane) {
    const int g0 = tile * Shape::kGroupsPerWarpTile;
    common::pipe_copy<16>(&s_code[0][lane],
                          reinterpret_cast<const uint4*>(
                              gate_code_row + g0 * Shape::kBytesPerGroup) +
                              lane);
    common::pipe_copy<16>(&s_code[1][lane],
                          reinterpret_cast<const uint4*>(
                              up_code_row + g0 * Shape::kBytesPerGroup) +
                              lane);
    if (lane < 2) {
        common::pipe_copy<16>(&s_scale[0][lane],
                              reinterpret_cast<const uint4*>(gate_scale_row + g0 * 2) + lane);
        common::pipe_copy<16>(&s_scale[1][lane],
                              reinterpret_cast<const uint4*>(up_scale_row + g0 * 2) + lane);
    }
    common::pipe_commit();
}

template <class Shape>
__global__ void q4_linear_swiglu_gemv_pair_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  __nv_bfloat16* __restrict__ out) {
    constexpr int kStages   = 3;
    constexpr int kPrefetch = kStages - 1;
    __shared__ __align__(16) __nv_bfloat16 x_sh[Shape::kK];
    __shared__ uint4 code_tile[Shape::kWarpsPerBlock][kStages][2][Shape::kVecsPerWarpTile];
    __shared__ uint4 scale_tile[Shape::kWarpsPerBlock][kStages][2][2];

    auto* x_sh_v    = reinterpret_cast<uint4*>(x_sh);
    const auto* x_g = reinterpret_cast<const uint4*>(x);
    for (int i = static_cast<int>(threadIdx.x); i < Shape::kXVecs; i += static_cast<int>(blockDim.x)) {
        x_sh_v[i] = x_g[i];
    }
    __syncthreads();

    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int warp    = static_cast<int>(threadIdx.x) >> 5;
    const int out_row = static_cast<int>(blockIdx.x) * Shape::kPairsPerBlock + warp;

    const std::uint8_t* gate_code_row =
        codes + static_cast<std::int64_t>(out_row) * Shape::kGroups * Shape::kBytesPerGroup;
    const std::uint8_t* gate_scale_row =
        scales + static_cast<std::int64_t>(out_row) * Shape::kGroups * 2;
    const std::uint8_t* up_code_row = codes +
        static_cast<std::int64_t>(out_row + Shape::kIntermediate) * Shape::kGroups *
            Shape::kBytesPerGroup;
    const std::uint8_t* up_scale_row =
        scales +
        static_cast<std::int64_t>(out_row + Shape::kIntermediate) * Shape::kGroups * 2;
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x_sh);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < Shape::kTiles) {
            q4_issue_pair_tile<Shape>(code_tile[warp][p], scale_tile[warp][p], gate_code_row,
                                      gate_scale_row, up_code_row, up_scale_row, p, lane);
        } else {
            common::pipe_commit();
        }
    }

#pragma unroll 1
    for (int tile = 0; tile < Shape::kTiles; ++tile) {
        const int fetch = tile + kPrefetch;
        if (fetch < Shape::kTiles) {
            const int buf = fetch % kStages;
            q4_issue_pair_tile<Shape>(code_tile[warp][buf], scale_tile[warp][buf], gate_code_row,
                                      gate_scale_row, up_code_row, up_scale_row, fetch, lane);
        } else {
            common::pipe_commit();
        }
        common::pipe_wait<kPrefetch>();
        __syncwarp();

        const int buf           = tile % kStages;
        const auto* gate_codes  = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][0]);
        const auto* up_codes    = reinterpret_cast<const std::uint8_t*>(code_tile[warp][buf][1]);
        const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][0]);
        const auto* up_scales   = reinterpret_cast<const std::uint16_t*>(scale_tile[warp][buf][1]);
#pragma unroll
        for (int tile_group = 0; tile_group < Shape::kGroupsPerWarpTile; ++tile_group) {
            const float gate_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(gate_scales[tile_group])));
            const float up_scale =
                __half2float(__ushort_as_half(static_cast<std::uint16_t>(up_scales[tile_group])));

            const int gate_packed =
                static_cast<int>(gate_codes[tile_group * Shape::kBytesPerGroup + lane]);
            const int gate_q0   = common::sign_extend<4>(gate_packed & 0x0f);
            const int gate_q1   = common::sign_extend<4>(gate_packed >> 4);
            const int up_packed =
                static_cast<int>(up_codes[tile_group * Shape::kBytesPerGroup + lane]);
            const int up_q0     = common::sign_extend<4>(up_packed & 0x0f);
            const int up_q1     = common::sign_extend<4>(up_packed >> 4);
            const int k0 =
                (tile * Shape::kGroupsPerWarpTile + tile_group) * Shape::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(x2[k0 >> 1]);
            gate_acc        = fmaf(static_cast<float>(gate_q0) * gate_scale, xv.x, gate_acc);
            gate_acc        = fmaf(static_cast<float>(gate_q1) * gate_scale, xv.y, gate_acc);
            up_acc          = fmaf(static_cast<float>(up_q0) * up_scale, xv.x, up_acc);
            up_acc          = fmaf(static_cast<float>(up_q1) * up_scale, xv.y, up_acc);
        }
        __syncwarp();
    }

    gate_acc = common::warp_reduce_sum(gate_acc);
    up_acc   = common::warp_reduce_sum(up_acc);
    if (lane == 0) { out[out_row] = __float2bfloat16(common::silu(gate_acc) * up_acc); }
}

template <class Shape>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    (void)w;
    const int grid = Shape::kIntermediate / Shape::kPairsPerBlock;
    q4_linear_swiglu_gemv_pair_kernel<Shape><<<grid, Shape::kBlockThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_swiglu_gemv_pair(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    if (w.n == Q4SwiGlu27B::kN && w.k == Q4SwiGlu27B::kK && w.padded_shape[1] == Q4SwiGlu27B::kK)
        return launch_gemv<Q4SwiGlu27B>(x, w, out, stream);
    if (w.n == Q4SwiGlu9B::kN && w.k == Q4SwiGlu9B::kK && w.padded_shape[1] == Q4SwiGlu9B::kK)
        return launch_gemv<Q4SwiGlu9B>(x, w, out, stream);
    throw std::invalid_argument(
        "q4 linear_swiglu GEMV requires weight [34816,5120] or [24576,4096]");
}

void launch_q4_swiglu_small_t_tiled(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    if (w.n == Q4SwiGlu27B::kN && w.k == Q4SwiGlu27B::kK)
        Q4SwiGluSmallT<Q4SwiGlu27B>::kLaunchers[index](x, w, out, stream);
    else if (w.n == Q4SwiGlu9B::kN && w.k == Q4SwiGlu9B::kK)
        Q4SwiGluSmallT<Q4SwiGlu9B>::kLaunchers[index](x, w, out, stream);
    else
        throw std::invalid_argument(
            "q4 linear_swiglu small-T requires weight [34816,5120] or [24576,4096]");
}

} // namespace ninfer::ops::detail
