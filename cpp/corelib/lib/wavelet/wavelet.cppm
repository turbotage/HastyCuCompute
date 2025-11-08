module;

export module wavelet;

import tensor;

namespace hasty {

    export class WaveDec1D {
    public:
        WaveDec1D(std::string wavelet, int level, std::string mode)
            : wavelet_(wavelet), level_(level), mode_(mode)
        {}

    private:
        std::string wavelet_;
        int level_;
        std::string mode_;
    };

}