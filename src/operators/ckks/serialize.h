#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <openfhe/pke/ciphertext-fwd.h>
#include <openfhe/pke/cryptocontext-fwd.h>
#include <openfhe/core/lattice/hal/lat-backend.h>

#include <registry/registry.h>

namespace operators { namespace ckks {

struct CKKSLoaderKernel {
	CKKSLoaderKernel(OrtApi api, const OrtKernelInfo* info, std::shared_ptr<reg::CryptoRegistry> registry);

	void Compute(OrtKernelContext* context);

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
    std::shared_ptr<reg::CryptoRegistry> reg_;
};

struct CKKSSaverKernel {
	CKKSSaverKernel(OrtApi api, const OrtKernelInfo* info, std::shared_ptr<reg::CryptoRegistry> registry);

	void Compute(OrtKernelContext* context);

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
    std::shared_ptr<reg::CryptoRegistry> reg_;
};

struct CustomOpCKKSLoader : Ort::CustomOpBase<CustomOpCKKSLoader, CKKSLoaderKernel> {
    explicit CustomOpCKKSLoader(std::shared_ptr<reg::CryptoRegistry> registry) : registry_(registry) {}

	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSLoaderKernel(api, info, registry_);
	}
	const char* GetName() const {
		return "fhe.ckks.loader";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetInputTypeCount() const {
		return 4;
	}
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}
	ONNXTensorElementDataType GetOutputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetOutputTypeCount() const {
		return 2;
	}

private:
    std::shared_ptr<reg::CryptoRegistry> registry_;
};

struct CustomOpCKKSSaver : Ort::CustomOpBase<CustomOpCKKSSaver, CKKSSaverKernel> {
    explicit CustomOpCKKSSaver(std::shared_ptr<reg::CryptoRegistry> registry) : registry_(registry) {}

	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSSaverKernel(api, info, registry_);
	}
	const char* GetName() const {
		return "fhe.ckks.saver";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetInputTypeCount() const {
		return 1;
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