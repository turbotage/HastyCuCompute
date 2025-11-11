module;

export module interface_typing;

import util;
import tensor;

namespace hasty {
namespace interface {

using DeviceTypes = std::tuple<cuda_t, cpu_t>;
using TensorTypes = std::tuple<
                        f32_t, f64_t, 
                        c64_t, c128_t, 
                        i16_t, i32_t, i64_t,
                        b8_t
                    >;
constexpr std::size_t TensorRanks[] = {1,2,3,4,5};


}
}