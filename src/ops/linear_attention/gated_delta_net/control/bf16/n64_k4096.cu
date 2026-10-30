#include "ops/linear_attention/gated_delta_net/control/kernels/mma_launch.cuh"

namespace ninfer::ops::detail::gdn_control {
void launch_n64_k4096(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch scratch,
                      CudaExecutionView e) {
    using Geometry = MmaGeometry<32, 4096, 64>;
    bool launched  = false;
    switch (plan.split_k) {
    case 32:
        launched = launch_mma<Geometry, 32>(in, out, scratch, e);
        break;
    case 16:
        launched = launch_mma<Geometry, 16>(in, out, scratch, e);
        break;
    case 8:
        launched = launch_mma<Geometry, 8>(in, out, scratch, e);
        break;
    case 4:
        launched = launch_mma<Geometry, 4>(in, out, scratch, e);
        break;
    case 2:
        launched = launch_mma<Geometry, 2>(in, out, scratch, e);
        break;
    default:
        throw std::logic_error("gdn_control: invalid 64x4096 plan");
    }
    if (!launched) (void)launch_mma<Geometry, 2>(in, out, scratch, e);
}
} // namespace ninfer::ops::detail::gdn_control
