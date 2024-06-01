module;

#include "../pch.hpp"

export module min;

import util;
import op;

namespace hasty {



    export template<typename T, typename OUTPUT, typename... INPUT_PACK>
    concept is_anytensor_function = requires(T t, INPUT_PACK&&... ttpack) {
        requires sizeof...(INPUT_PACK) > 0;
        requires (is_anytensor<INPUT_PACK> && ...);
        requires is_anytensor<OUTPUT>;
        { t.operator()(ttpack...) } -> std::same_as<OUTPUT>;
    };

    export template<typename T, typename... INPUT_PACK>
    concept is_anytensor_logical = requires(T t, INPUT_PACK&&... ttpack) {
        requires sizeof...(INPUT_PACK) > 0;
        requires (is_anytensor<INPUT_PACK> && ...);
        { t.operator()(ttpack...) } -> std::same_as<bool>;
    };

    export template<typename FMIN, typename GMIN, typename UGLUE, typename CONVERGER,
                    is_anytensor X, is_anytensor Z, is_anytensor U>
    requires is_anytensor_function<FMIN,  X, X, Z, U> &&
             is_anytensor_function<GMIN,  Z, X, Z, U> &&
             is_anytensor_function<UGLUE, U, X, Z, U> &&
             is_anytensor_logical<CONVERGER, X, Z, U>
    auto admm(FMIN&& fmin, GMIN&& gmin, UGLUE&& uglue, CONVERGER&& converger, X&& x, Z&& z, U&& u) 
    {
        while (!converger(x, z, u)) {
            x = fmin(x, z, u);
            z = gmin(x, z, u);
            u = uglue(u, x, z);
        }
    }


}