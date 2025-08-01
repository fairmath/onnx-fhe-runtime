#include <cstdint>
#include <iostream>
#include <vector>

#include <openfhe/pke/ciphertext-fwd.h>
#include <openfhe/pke/cryptocontext-ser.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <operators/helper.h>
#include "svm.h"

#include "utils/serial.h"
#include "utils/sertype.h"

namespace operators {
namespace ckks {

const char* RBF_KERNEL = "rbf";
const char* LIN_KERNEL = "linear";
}  // namespace ckks
}  // namespace operators

namespace {

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> sum(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& in,
                                             size_t row_size) {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> out = in;
    for (size_t i = 0; i < static_cast<size_t>(std::log2(row_size)); ++i) {
        auto tmp = cc->EvalRotate(out, 1 << i);
        out = cc->EvalAdd(out, tmp);
    }
    return out;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> expand(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct,
                                                size_t svSize,
                                                size_t copyCount) {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> out = ct;
    auto rCount = static_cast<int>(std::ceil(std::log2(copyCount)));
    for (size_t i = 0; i < rCount; ++i) {
        auto tmp = cc->EvalRotate(out, svSize * (1 << i));
        out = cc->EvalAdd(out, tmp);
    }
    return out;
}

class LinKernel : public operators::ckks::Kernel {
   public:
    LinKernel(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, int batchSize, int features)
        : cc_(cc), batchSize_(batchSize), features_(features) {}

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> compute(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> input,
                                                     lbcrypto::Plaintext svPlaintext) override {
        return sum(cc_, cc_->EvalMult(input, svPlaintext), features_);
    }

   private:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
    int batchSize_;
    int features_;
};

class RBFKernel : public operators::ckks::Kernel {
   public:
    RBFKernel(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
              std::vector<double> kernelParams,
              int batchSize,
              int features)
        : cc_(cc), batchSize_(batchSize), features_(features) {
        std::vector<double> gammaVector(batchSize, 0);
        for (size_t i = 0; i < batchSize; i += features) {
            gammaVector[i] = kernelParams[0];
        }

        gammaPlaintext_ = cc->MakeCKKSPackedPlaintext(gammaVector);
    }

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> compute(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> input,
                                                     lbcrypto::Plaintext svPlaintext) override {
        auto sub = cc_->EvalSub(input, svPlaintext);
        auto subPow2 = cc_->EvalMult(sub, sub);
        auto sqDistance = sum(cc_, subPow2, features_);
        auto arg = cc_->EvalNegate(cc_->EvalMult(sqDistance, gammaPlaintext_));

        return cc_->EvalChebyshevFunction([](double x) { return std::exp(x); }, arg, -30000, 0,
                                          1024);
    }

   private:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
    lbcrypto::Plaintext gammaPlaintext_;
    int batchSize_;
    int features_;
};
}  // namespace

namespace operators {
namespace ckks {

CKKSSvmKernel::CKKSSvmKernel(const OrtApi& api,
                             const OrtKernelInfo* info,
                             std::shared_ptr<reg::CryptoRegistry> registry)
    : api_(api), reg_(registry) {
    Ort::ConstKernelInfo kinfo(info);
    kernelType_ = kinfo.GetAttribute<std::string>("kernel_type");
}

void CKKSSvmKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
    auto cipher = ctx.GetInput(1).GetTensorData<std::string>();
    auto pk = ctx.GetInput(2).GetTensorData<std::string>();

    auto cc = reg_->context(*cryptoCtxStr);
    auto input = reg_->cipher(*cipher);
    auto pubKey = reg_->pk(*pk);

    Ort::ConstValue supportVectors = ctx.GetInput(3);

    std::vector<int64_t> shape;
    GetTypeAndShape(supportVectors, shape);
    if (shape.size() != 2) {
        throw std::logic_error("suport vectors tensor should be a tensor of rank 2!");
    }

    std::vector<double> flattenSupportVectors;
    auto svData = supportVectors.GetTensorData<double>();
    flattenSupportVectors.assign(svData, svData + shape[0] * shape[1]);

    Ort::ConstValue kernelSettingsParam = ctx.GetInput(4);
    auto kernelParams = kernelSettingsParam.GetTensorData<double>();

    auto coefficientsParam = ctx.GetInput(5);
    GetTypeAndShape(coefficientsParam, shape);
    if (shape.size() != 1) {
        throw std::logic_error("coefficients tensor should be a tensor of rank 1!");
    }

    std::vector<double> coefficients;
    auto dataCoef = coefficientsParam.GetTensorData<double>();
    coefficients.assign(dataCoef, dataCoef + shape[0]);

    Ort::ConstValue biasParam = ctx.GetInput(6);
    auto bias = *biasParam.GetTensorData<double>();

    auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
    std::string* out = r.GetTensorMutableData<std::string>();

    auto features = flattenSupportVectors.size() / coefficients.size();
    auto nearPow2Features = 1 << static_cast<int>(std::ceil(std::log2(features)));

    if (kernelType_ == RBF_KERNEL) {
        auto gamma = kernelParams[0];
        auto params = std::vector<double>(1, gamma);
        kernel_ = std::make_shared<RBFKernel>(cc, params, cc->GetEncodingParams()->GetBatchSize(),
                                              nearPow2Features);
    }

    if (kernelType_ == LIN_KERNEL) {
        kernel_ = std::make_shared<LinKernel>(cc, cc->GetEncodingParams()->GetBatchSize(),
                                              nearPow2Features);
    }

    auto res = compute(cc, pubKey, input, flattenSupportVectors, coefficients, bias);

    *out = reg_->loadCipher(res);
}

std::vector<double> insertZeros(const std::vector<double>& input, int n, int k) {
    std::vector<double> result;
    result.reserve(input.size() + (input.size() / n + 1) * k);

    int count = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        result.push_back(input[i]);
        count++;
        if (count == n) {
            result.insert(result.end(), k, 0);
            count = 0;
        }
    }
    return result;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> CKKSSvmKernel::compute(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> pk,
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> input,
    std::vector<double> flattenSupportVectors,
    std::vector<double> coefficients,
    double bias) {
    auto batchSize = cc->GetEncodingParams()->GetBatchSize();
    auto features = flattenSupportVectors.size() / coefficients.size();
    auto nearPow2Features = 1 << static_cast<int>(std::ceil(std::log2(features)));
    flattenSupportVectors =
        insertZeros(flattenSupportVectors, features, nearPow2Features - features);

    features = nearPow2Features;  // extend vectors dimension to nearest power of 2

    auto rotate = static_cast<int>(std::log2(batchSize) - std::log2(features));

    auto expandedCt = expand(cc, input, features, batchSize / features);
    std::vector<double> biasVector(batchSize, bias);

    auto biasPlaintext = cc->MakeCKKSPackedPlaintext(biasVector);
    auto svPlaintext = cc->MakeCKKSPackedPlaintext(flattenSupportVectors);

    std::vector<double> expandedCoeffs(batchSize, 0);
    for (auto i = 0; i < coefficients.size(); i++) {
        expandedCoeffs[i * features] = coefficients[i];
    }

    auto coeffs = cc->MakeCKKSPackedPlaintext(expandedCoeffs);
    auto kernelValue = kernel_->compute(expandedCt, svPlaintext);

    // decision
    auto withKoeffs = cc->EvalMult(kernelValue, coeffs);
    auto overall = cc->EvalSum(withKoeffs, features * coefficients.size());

    return cc->EvalAdd(overall, biasPlaintext);
}

}  // namespace ckks
}  // namespace operators
