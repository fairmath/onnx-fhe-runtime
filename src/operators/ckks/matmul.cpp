#include <cstdint>
#include <vector>

#include <openfhe/pke/cryptocontext-ser.h>
#include "openfhe/pke/ciphertext-fwd.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <operators/helper.h>

#include "matmul.h"

namespace operators {
namespace ckks {

CKKSMatMulKernel::CKKSMatMulKernel(OrtApi api,
                                   const OrtKernelInfo* info,
                                   std::shared_ptr<reg::CryptoRegistry> registry)
    : api_(api), reg_(registry) {}

void CKKSMatMulKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
    auto cipher = ctx.GetInput(1).GetTensorData<std::string>();

    auto cc = reg_->context(*cryptoCtxStr);
    auto input = reg_->cipher(*cipher);

    std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> results;
    Ort::ConstValue weights = ctx.GetInput(2);

    std::vector<int64_t> shape;
    GetTypeAndShape(weights, shape);
    auto tensorWeight = weights.GetTensorData<double>();

    if (shape.size() != 2) {
        throw std::logic_error("weight tensor should be a tensor of rank 2");
    }

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res;
    size_t first = 0;

    for (auto i = 0; i < shape[0]; ++i) {  // for every weights array
        auto weightVectorSize = shape[1];
        std::vector<double> wi;
        wi.reserve(weightVectorSize);

        std::copy(&tensorWeight[first], &tensorWeight[first + weightVectorSize],
                  std::back_inserter(wi));
        first += weightVectorSize;

        auto weightedInput = cc->EvalMult(input, cc->MakeCKKSPackedPlaintext(wi));
        weightedInput = cc->EvalSum(weightedInput, wi.size());
        std::vector<double> one(wi.size());
        one[0] = 1.0;
        weightedInput = cc->EvalMult(weightedInput, cc->MakeCKKSPackedPlaintext(one));

        if (i == 0) {
            res = weightedInput;

            continue;
        }

        weightedInput = stepRotation(cc, weightedInput, -i);
        cc->EvalAddInPlace(res, weightedInput);
    }

    auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
    std::string* out = r.GetTensorMutableData<std::string>();
    *out = reg_->loadCipher(res);
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> CKKSMatMulKernel::stepRotation(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext,
    int index) {
    int sign = index >= 0 ? 1 : -1;
    index = std::abs(index);
    int step = 1;

    while (index > 0) {
        if (index & 1) {
            ciphertext = cc->EvalRotate(ciphertext, sign * step);
        }

        index >>= 1;
        step *= 2;
    }

    return ciphertext;
}

}  // namespace ckks
}  // namespace operators