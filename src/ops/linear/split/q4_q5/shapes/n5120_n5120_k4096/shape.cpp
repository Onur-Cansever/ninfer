#include "ops/linear/split/q4_q5/shapes/n5120_n5120_k4096/launch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n5120_n5120_k4096 {
namespace {
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    if (x.ne[1] <= 12)
        launch_small(x, weights.first, *weights.second, outputs, stream);
    else
        launch_matrix(x, weights.first, *weights.second, outputs, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::q4_q5_n5120_n5120_k4096

namespace ninfer::ops::detail::linear_split {
const SplitShape kQ4Q5SplitN5120N5120K4096{
    5120, 4096, kSplitRows<4096, 1024, 4096, 1024>, q4_q5_n5120_n5120_k4096::launch, nullptr, 5120};
} // namespace ninfer::ops::detail::linear_split
