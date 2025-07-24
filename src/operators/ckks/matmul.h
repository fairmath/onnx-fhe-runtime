#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <registry/registry.h>

namespace operators { namespace ckks {

struct CKKSMatMulKernel {
	CKKSMatMulKernel(OrtApi api, const OrtKernelInfo* info, std::shared_ptr<reg::CryptoRegistry> registry);

	void Compute(OrtKernelContext* context);

private:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> stepRotation(
        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, 
        lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext, 
        int index
    );

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
    std::shared_ptr<reg::CryptoRegistry> reg_;
};

struct CustomOpCKKSMatMul : Ort::CustomOpBase<CustomOpCKKSMatMul, CKKSMatMulKernel> {
    CustomOpCKKSMatMul(std::shared_ptr<reg::CryptoRegistry> registry) : registry_(registry) { };

	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSMatMulKernel(api, info, registry_);
	}
	const char* GetName() const {
		return "fhe.ckks.matmul";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		if (index == 2) {
			return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
		}

		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetInputTypeCount() const {
		return 3;
	}
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}
	ONNXTensorElementDataType GetOutputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetOutputTypeCount() const {
		return 1;
	}

private:
    std::shared_ptr<reg::CryptoRegistry> registry_;
};

} }