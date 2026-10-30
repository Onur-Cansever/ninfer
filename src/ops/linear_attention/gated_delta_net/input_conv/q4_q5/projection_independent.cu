#include "ops/linear/kernels/q5/q5_instances.cuh"
#include "ops/linear/kernels/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/q4_q5/launch.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/kernels/q4/q4_simt_launch.cuh"
#include "ops/linear/kernels/q4/q4_gemv_launch.cuh"
#include "ops/linear/kernels/q5/q5_simt_launch.cuh"
#include "ops/linear/kernels/q5/q5_gemv_launch.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Q4GdnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8T8Schedule = Q4A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;

template <int Hidden>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const auto operands = q4_linear_operands(x, weight);
    const auto output =
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0};
    if constexpr (Hidden == 4096)
        launch_q4_a16_gemv<q4_instances::GemvR1W8K4096>(operands, output, LinearIdentityEpilogue{},
                                                         stream);
    else
        launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(operands, output, LinearIdentityEpilogue{},
                                                         stream);
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_simt<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <int Hidden, std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, Hidden, Capacity>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <int Hidden>
void launch_q4_by_columns(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv<Hidden>(x, weight, out, stream);
        break;
    case 2:
    case 3:
    case 4:
        launch_q4_simt<Q4GdnSimtR8T4Schedule>(x, weight, out, stream);
        break;
    case 5:
    case 6:
        launch_q4_simt<Q4GdnSimtR8T8Schedule>(x, weight, out, stream);
        break;
    case 7:
        launch_q4_sliced_exact<Hidden, 7>(x, weight, out, stream);
        break;
    case 8:
        launch_q4_sliced_exact<Hidden, 8>(x, weight, out, stream);
        break;
    case 9:
        launch_q4_sliced_exact<Hidden, 9>(x, weight, out, stream);
        break;
    case 10:
        launch_q4_sliced_exact<Hidden, 10>(x, weight, out, stream);
        break;
    case 11:
        launch_q4_sliced_exact<Hidden, 11>(x, weight, out, stream);
        break;
    case 12:
        launch_q4_sliced_exact<Hidden, 12>(x, weight, out, stream);
        break;
    }
    throw std::logic_error("GDN independent Q4 projection: invalid selected extent");
}

template <int ValueRows>
auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<ValueRows>{
        {static_cast<__nv_bfloat16*>(first.data), std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)),
         0},
        {static_cast<__nv_bfloat16*>(second.data),
         std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)), 0}};
}

template <int Hidden, int ValueRows>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    const auto operands = q5_linear_operands(x, weight);
    const auto output   = q5_projection_output<ValueRows>(value, z);
    if constexpr (Hidden == 4096)
        launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK4096>(operands, output,
                                                                 LinearIdentityEpilogue{}, stream);
    else
        launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(operands, output,
                                                                 LinearIdentityEpilogue{}, stream);
}

template <int Cols, int Hidden, int ValueRows>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, Hidden, true>;
    launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight),
                                        q5_projection_output<ValueRows>(value, z),
                                        LinearIdentityEpilogue{}, stream);
}

template <int kColsPerTile, int ValueRows>
void launch_q5_simt_cols(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    using Schedule = Q5A16SimtSchedule<8, kColsPerTile, 1, 16, 2, Cache::ca, 1>;
    launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight),
                                 q5_projection_output<ValueRows>(value, z), LinearIdentityEpilogue{},
                                 stream);
}

template <int Hidden, int ValueRows>
void launch_q5_by_columns(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q5_gemv<Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 2:
        launch_q5_split4<2, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 3:
        launch_q5_split4<3, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 4:
        launch_q5_split4<4, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 5:
        launch_q5_split4<5, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 6:
        launch_q5_split4<6, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 7:
        launch_q5_split4<7, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 8:
        launch_q5_split4<8, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 9:
        launch_q5_split4<9, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 10:
        launch_q5_split4<10, Hidden, ValueRows>(x, weight, value, z, stream);
        break;
    case 11:
    case 12:
        launch_q5_simt_cols<4, ValueRows>(x, weight, value, z, stream);
        break;
    }
    throw std::logic_error("GDN independent Q5 projection: invalid selected extent");
}

template <int Hidden, int ValueRows>
void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule = Q4GdnSimtR8T4Schedule;
    using Q5Schedule = Q5A16DirectSimtSchedule<1, 4, 4, 4, 10, Hidden, true>;
    // Q5 and Q4 publish disjoint rows; Q4 joins the Q5 producer at kernel exit.
    launch_q5_a16_direct_simt<Q5Schedule, true>(q5_linear_operands(x, value_z_weight),
                                                q5_projection_output<ValueRows>(value, z),
                                                LinearIdentityEpilogue{}, stream);
    launch_q4_a16_simt<Q4Schedule, false, true, true>(
        q4_linear_operands(x, qk_weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(qk.data),
                                static_cast<std::int64_t>(qk.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

template <int Hidden, int ValueRows>
void launch_q4_q5_gdn_input_independent_impl(const Tensor& x, const Weight& qk_weight,
                                             const Weight& value_z_weight, Tensor& qk,
                                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl<Hidden, ValueRows>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4_by_columns<Hidden>(x, qk_weight, qk, stream);
    launch_q5_by_columns<Hidden, ValueRows>(x, value_z_weight, value, z, stream);
}

} // namespace

void launch_q4_q5_gdn_input_independent(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    const int hidden    = qk_weight.k;
    const int value_rows = value_z_weight.n / 2;
    if (hidden == 4096 && value_rows == 4096)
        return launch_q4_q5_gdn_input_independent_impl<4096, 4096>(x, qk_weight, value_z_weight,
                                                                    qk, value, z, stream);
    if (hidden == 5120 && value_rows == 6144)
        return launch_q4_q5_gdn_input_independent_impl<5120, 6144>(x, qk_weight, value_z_weight,
                                                                    qk, value, z, stream);
    throw std::invalid_argument("GDN independent Q4/Q5 projection: unsupported geometry");
}

} // namespace ninfer::ops::detail
