module;

#include <matx.h>
#include <xtensor/xarray.hpp>

export module tensor;

namespace hasty {

    export struct tensorview {
    public:

        tensorview() {
            a = 10;
        }
        
    private:

        int64_t a;
    };

}