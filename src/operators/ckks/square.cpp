#include <cstdint>
#include <vector>

#include "openfhe/pke/ciphertext-fwd.h"
#include <openfhe/pke/cryptocontext-ser.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include "square.h"

namespace operators { namespace ckks {


CKKSSquareKernel::CKKSSquareKernel(OrtApi api, const OrtKernelInfo* info, std::shared_ptr<reg::CryptoRegistry> registry) 
    : api_(api), kinfo_(info), reg_(registry) {
}

void CKKSSquareKernel::Compute(OrtKernelContext* context) {
	Ort::KernelContext ctx(context);

	auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
	auto cipher = ctx.GetInput(1).GetTensorData<std::string>();

	auto cc = reg_->context(*cryptoCtxStr);
	auto input = reg_->cipher(*cipher);

	auto result = cc->EvalMult(input, input);

	auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
	std::string* out = r.GetTensorMutableData<std::string>();
	
    *out = reg_->loadCipher(result);
}

} }

