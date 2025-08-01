#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <registry/registry.h>

namespace operators {
namespace ckks {

struct CKKSSquareKernel {
    CKKSSquareKernel(OrtApi api,
                     const OrtKernelInfo* info,
                     std::shared_ptr<reg::CryptoRegistry> registry);

    void Compute(OrtKernelContext* context);

   private:
    OrtApi api_;
    std::shared_ptr<reg::CryptoRegistry> reg_;
};

struct CustomOpCKKSSquare : Ort::CustomOpBase<CustomOpCKKSSquare, CKKSSquareKernel> {
    CustomOpCKKSSquare(std::shared_ptr<reg::CryptoRegistry> registry) : registry_(registry){};

    void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
        return new CKKSSquareKernel(api, info, registry_);
    }
    const char* GetName() const { return "fhe.ckks.square"; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }
    size_t GetInputTypeCount() const { return 2; }
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