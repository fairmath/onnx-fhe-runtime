#include <cstdint>
#include <vector>

#include "openfhe/core/lattice/hal/lat-backend.h"
#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/cryptocontext-fwd.h"
#include <openfhe/pke/cryptocontext-ser.h>

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "add.h"

namespace operators { namespace ckks {


CKKSAddKernel::CKKSAddKernel(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {

}

//this code has to be reworked:
// - get rid of serialization here
// - use crypto registry to get ciphertext and context
void CKKSAddKernel::Compute(OrtKernelContext* context) {
	Ort::KernelContext ctx(context);

	auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
	auto cipherA = ctx.GetInput(1).GetTensorData<std::string>();
	auto cipherB = ctx.GetInput(2).GetTensorData<std::string>();

	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
	std::stringstream ccStream(*cryptoCtxStr);
	lbcrypto::Serial::Deserialize(cc, ccStream, lbcrypto::SerType::JSON);

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encA;
	std::stringstream caStream(*cipherA);
	lbcrypto::Serial::Deserialize(encA, caStream, lbcrypto::SerType::JSON);

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encB;
	std::stringstream cbStream(*cipherB);
	lbcrypto::Serial::Deserialize(encB, cbStream, lbcrypto::SerType::JSON);
		
	auto encRes = cc->EvalAdd(encA, encB);

	auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
	std::string* out = r.GetTensorMutableData<std::string>();

	std::stringstream res;
	lbcrypto::Serial::Serialize(encRes, res, lbcrypto::SerType::JSON);
	*out = res.str();
}



} }

