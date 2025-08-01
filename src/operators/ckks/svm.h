#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/encoding/plaintext-fwd.h"

#include <registry/registry.h>

namespace operators {
namespace ckks {

class Kernel {
   public:
    virtual lbcrypto::Ciphertext<lbcrypto::DCRTPoly> compute(
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> input,
        lbcrypto::Plaintext svPlaintext) = 0;

    virtual ~Kernel(){};
};

struct CKKSSvmKernel {
    CKKSSvmKernel(const OrtApi& api,
                  const OrtKernelInfo* info,
                  std::shared_ptr<reg::CryptoRegistry> registry);

    void Compute(OrtKernelContext* context);

   private:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> compute(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
                                                     lbcrypto::PublicKey<lbcrypto::DCRTPoly> pk,
                                                     lbcrypto::Ciphertext<lbcrypto::DCRTPoly> input,
                                                     std::vector<double> flattenSupportVectors,
                                                     std::vector<double> coefficients,
                                                     double bias);

   private:
    const OrtApi& api_;
    std::shared_ptr<reg::CryptoRegistry> reg_;
    std::shared_ptr<Kernel> kernel_;
    std::string kernelType_;
};

struct CustomOpCKKSSvm : Ort::CustomOpBase<CustomOpCKKSSvm, CKKSSvmKernel> {
    CustomOpCKKSSvm(std::shared_ptr<reg::CryptoRegistry> registry) : registry_(registry){};

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new CKKSSvmKernel(api, info, registry_);
    }
    const char* GetName() const { return "fhe.ckks.svm"; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0 || index == 1 || index == 2) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
        }

        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }
    size_t GetInputTypeCount() const { return 7; }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    size_t GetOutputTypeCount() const { return 1; }

   private:
    std::shared_ptr<reg::CryptoRegistry> registry_;
};

}  // namespace ckks
}  // namespace operators