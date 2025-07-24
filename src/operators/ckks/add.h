#pragma once

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

namespace operators { namespace ckks {

struct CKKSAddKernel {
	CKKSAddKernel(OrtApi api, const OrtKernelInfo* info);

	void Compute(OrtKernelContext* context);

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

} }
