#include "ops/linear/split/q4_q5/shapes/n4096_n8192_k4096/launch.h"

namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n8192_k4096 {
namespace {
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs, MemoryArena*,
            cudaStream_t stream) {
    if (x.ne[1] <= 12)
        launch_small(x, weights.first, *weights.second, outputs, stream);
    else
        launch_matrix(x, weights.first, *weights.second, outputs, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::q4_q5_n4096_n8192_k4096

namespace ninfer::ops::detail::linear_split {
const SplitShape kQ4Q5SplitN4096N8192K4096{
    4096, 4096, kSplitRows<8192, 4096>, q4_q5_n4096_n8192_k4096::launch, nullptr, 8192};
} // namespace ninfer::ops::detail::linear_split
