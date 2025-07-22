#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string_view>
#include <tuple>
#include <mutex>
#include <iostream>

#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/cryptocontext-fwd.h"
#include <openfhe/pke/cryptocontext-ser.h>
#include "openfhe/core/lattice/hal/lat-backend.h"

#include "utils/serial.h"
#include "utils/sertype.h"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

// internal linkage
namespace {

std::string rndstr(size_t length) {
    const std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd; 
    std::mt19937 gen(rd()); 

    std::uniform_int_distribution<> dist(0, charset.size() - 1);

    std::string result;
    result.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        result += charset[dist(gen)];
    }

    return result;
}

class CryptoRegistry {
public:
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context(std::string name) {
		return contexts_[name];
	}

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipher(std::string name) {
		return ciphertexts_[name];
	}

	std::string loadCtx(std::string ctxName, std::string rotKeysName, std::string mulKeys) {
		auto id = rndstr(16);

		lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
		lbcrypto::Serial::DeserializeFromFile(ctxName, cc, lbcrypto::SerType::BINARY);

		if (!rotKeysName.empty()) {
			std::fstream rkStream(rotKeysName);
			cc->DeserializeEvalAutomorphismKey(rkStream, lbcrypto::SerType::BINARY);
		}

		if (!mulKeys.empty()) {
			std::fstream mkStream(mulKeys);
			cc->DeserializeEvalMultKey(mkStream, lbcrypto::SerType::BINARY);
		}

		contexts_[id] = cc;

		return id;
	}

	std::string loadCipher(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> c) {
		auto id = rndstr(16);
	
		ciphertexts_[id] = c;

		return id;
	}

	std::string loadCipher(std::string name) {
		auto id = rndstr(16);
		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cphr;
		lbcrypto::Serial::DeserializeFromFile(name, cphr, lbcrypto::SerType::BINARY);

		ciphertexts_[id] = cphr;

		return id;
	}

private:
	std::unordered_map<std::string, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> contexts_;
	std::unordered_map<std::string, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts_;
};

CryptoRegistry reg_;

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue& input, std::vector<int64_t>& shape, bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();

  if (swap) {
    std::swap(shape[0], shape[1]);
  }
  return t.GetElementType();
}

// kernel of FMA operation
struct CKKSAddKernel {
	CKKSAddKernel(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		assert(ctx.GetInputCount() == 3); //context, cipher1, cipher2

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

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

struct CKKSSquare {
	CKKSSquare(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		assert(ctx.GetInputCount() == 2); //context, cipher1

		auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
		auto cipher = ctx.GetInput(1).GetTensorData<std::string>();

		auto cc = reg_.context(*cryptoCtxStr);
		auto input = reg_.cipher(*cipher);

		auto result = cc->EvalMult(input, input);

		auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
		std::string* out = r.GetTensorMutableData<std::string>();
		*out = reg_.loadCipher(result);
	}

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

struct CKKSLoader {
	CKKSLoader(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
		auto rotKeyStr = ctx.GetInput(1).GetTensorData<std::string>();
		auto mulKeyStr = ctx.GetInput(2).GetTensorData<std::string>();

		auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
		std::string* outCtx = r.GetTensorMutableData<std::string>();
		*outCtx = reg_.loadCtx(*cryptoCtxStr, *rotKeyStr, *mulKeyStr);
		
		r = ctx.GetOutput(1, std::vector<int64_t>{1});
		std::string* outCipher = r.GetTensorMutableData<std::string>();
		*outCipher = reg_.loadCipher(*ctx.GetInput(3).GetTensorData<std::string>());
	}

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

struct CKKSSaver {
	CKKSSaver(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		assert(ctx.GetInputCount() == 1); //ciphertext

		auto cipher = ctx.GetInput(0).GetTensorData<std::string>();

		auto cphr = reg_.cipher(*cipher);
		
		auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
		std::string* outCtx = r.GetTensorMutableData<std::string>();
		*outCtx = "mres.bin";
		
		lbcrypto::Serial::SerializeToFile(*outCtx, cphr, lbcrypto::SerType::BINARY);
	}

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

// kernel of FMA operation
struct CKKSMatMul {
	CKKSMatMul(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		assert(ctx.GetInputCount() == 3); //context, cipher1, weights

		auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
		auto cipher = ctx.GetInput(1).GetTensorData<std::string>();

		auto cc = reg_.context(*cryptoCtxStr);
		auto input = reg_.cipher(*cipher);

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> results;
  		Ort::ConstValue weights = ctx.GetInput(2);

  		std::vector<int64_t> shape;
  		GetTypeAndShape(weights, shape);
		auto tensorWeight = weights.GetTensorData<double>();
		
		assert(shape.size() == 2);
		if (shape.size() != 2) {
			throw std::logic_error("weight tensor should be a tensor of rank 2");
		}

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res;
		size_t first = 0;

		for (auto i = 0; i < shape[0]; ++i) { //for every weights array
			auto weightVectorSize = shape[1];
			std::vector<double> wi;
			wi.reserve(weightVectorSize);			
			
			std::copy(&tensorWeight[first], &tensorWeight[first + weightVectorSize], std::back_inserter(wi));
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
		*out = reg_.loadCipher(res);
	}

private:
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> stepRotation(
		lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc, 
		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext, 
		int index
	) {
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
private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

struct CustomOpCKKSMatMulSum : Ort::CustomOpBase<CustomOpCKKSMatMulSum, CKKSMatMul> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSMatMul(api, info);
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
};

struct CustomOpCKKSSquare : Ort::CustomOpBase<CustomOpCKKSSquare, CKKSSquare> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSSquare(api, info);
	}
	const char* GetName() const {
		return "fhe.ckks.square";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	}
	size_t GetInputTypeCount() const {
		return 2;
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
};

struct CustomOpCKKSAdd : Ort::CustomOpBase<CustomOpCKKSAdd, CKKSAddKernel> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSAddKernel(api, info);
	}
	const char* GetName() const {
		return "fhe.ckks.add";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
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
};

// kernel of FMA operation
template <typename T>
struct FmaKernel {
	FmaKernel(OrtApi api, const OrtKernelInfo* info):api_(api), kinfo_(info) {}

	void Compute(OrtKernelContext* context) {
		Ort::KernelContext ctx(context);

		assert(ctx.GetInputCount() >= 2);

  		Ort::ConstValue input_A = ctx.GetInput(0);
  		Ort::ConstValue input_B = ctx.GetInput(1);

  		std::vector<int64_t> shape_A, shape_B;
  		GetTypeAndShape(input_A, shape_A);
  		GetTypeAndShape(input_B, shape_B);

		std::vector<int64_t> dimensions{shape_A[0], shape_B[0]};
		Ort::UnownedValue Y = ctx.GetOutput(0, shape_A);
		T* out = Y.GetTensorMutableData<T>();

		if (ctx.GetInputCount() > 2) {
  			Ort::ConstValue input_C = ctx.GetInput(2);

			std::vector<int64_t> shape_C;
			GetTypeAndShape(input_C, shape_C);

			auto a = input_A.GetTensorData<T>();
			auto b = input_B.GetTensorData<T>();
			auto c = input_C.GetTensorData<T>();

			for (auto i = 0; i < shape_A[0]; i++) {
				std::cout << a[i] << " * " << b[i] << " + " << c[i] << std::endl;
				*out++ = a[i] * b[i] + c[i];
			}
		} else {
			auto a = input_A.GetTensorData<T>();
			auto b = input_B.GetTensorData<T>();

			for (auto i = 0; i < shape_A[0]; i++) {
				std::cout << a[i] << " * " << b[i] << std::endl;
				*out++ = a[i] * b[i];
			}
		}
	}

private:
	OrtApi api_;
	Ort::ConstKernelInfo kinfo_;
};

struct CustomOpCKKSLoader : Ort::CustomOpBase<CustomOpCKKSLoader, CKKSLoader> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSLoader(api, info);
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
};

struct CustomOpCKKSSaver : Ort::CustomOpBase<CustomOpCKKSSaver, CKKSSaver> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new CKKSLoader(api, info);
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
};

// FMA operator
template <typename T>
struct CustomOpFma : Ort::CustomOpBase<CustomOpFma<T>, FmaKernel<T>> {
	void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
		return new FmaKernel<T>(api, info);
	}
	const char* GetName() const {
		return "Fma";
	}
	ONNXTensorElementDataType GetInputType(size_t index) const {
		return Ort::TypeToTensorType<T>::type;
	}
	size_t GetInputTypeCount() const {
		return 3;
	}
	OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
		if (index > 1) {
			return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
		}
		return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
	}
	ONNXTensorElementDataType GetOutputType(size_t index) const {
		return Ort::TypeToTensorType<T>::type;
	}
	size_t GetOutputTypeCount() const {
		return 1;
	}
};


void register_domain(Ort::CustomOpDomain&& domain) {
	static std::mutex m;
	static std::vector<Ort::CustomOpDomain> domain_registry;

	std::lock_guard<std::mutex> lock(m);
	domain_registry.push_back(std::move(domain));
}


static const char* domain_float = "my_ops.float";
static const char* domain_double = "my_ops.double";
static const char* domain_ckks_add = "fhe.ckks.add";
static const char* domain_ckks_matmul = "fhe.ckks.matmul";
static const char* domain_ckks_square = "fhe.ckks.square";
static const char* domain_ckks_loader = "fhe.ckks.loader";
static const char* domain_ckks_saver = "fhe.ckks.saver";

CustomOpFma<float> op_fma_float;	// float type FMA operator
CustomOpFma<double> op_fma_double;	// double type FMA operator
CustomOpCKKSAdd op_ckks_add;
CustomOpCKKSMatMulSum op_ckks_matmul;
CustomOpCKKSSquare op_ckks_square;
CustomOpCKKSLoader op_ckks_loader;
CustomOpCKKSSaver op_ckks_saver;

// domain name and corresponding operator
std::vector<std::tuple<const char*, OrtCustomOp*>> op_list = {
	std::make_tuple(domain_float, &op_fma_float),
	std::make_tuple(domain_double, &op_fma_double),
	std::make_tuple(domain_ckks_add, &op_ckks_add),
	std::make_tuple(domain_ckks_matmul, &op_ckks_matmul),
	std::make_tuple(domain_ckks_square, &op_ckks_square),
	std::make_tuple(domain_ckks_loader, &op_ckks_loader),
	std::make_tuple(domain_ckks_saver, &op_ckks_saver)
};

}	// namespace


// external linkage

extern "C" {

// call from SessionOptions.register_custom_ops_library()
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
	Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
	OrtStatus* status = nullptr;

#ifndef ORT_NO_EXCEPTIONS
  	try {
#endif
		Ort::UnownedSessionOptions session_options(options);

		Ort::CustomOpDomain domainFloat{domain_float};
		domainFloat.Add(&op_fma_float);
		session_options.Add(domainFloat);
		register_domain(std::move(domainFloat));

		Ort::CustomOpDomain domainDouble{domain_double};
		domainDouble.Add(&op_fma_double);
		session_options.Add(domainDouble);
		register_domain(std::move(domainDouble));

		Ort::CustomOpDomain domainCkksAdd{domain_ckks_add};
		domainCkksAdd.Add(&op_ckks_add);
		session_options.Add(domainCkksAdd);
		register_domain(std::move(domainCkksAdd));

		Ort::CustomOpDomain domainCkksMatMul(domain_ckks_matmul);
		domainCkksMatMul.Add(&op_ckks_matmul);
		session_options.Add(domainCkksMatMul);
		register_domain(std::move(domainCkksMatMul));

		Ort::CustomOpDomain domainCkksSquare(domain_ckks_square);
		domainCkksSquare.Add(&op_ckks_square);
		session_options.Add(domainCkksSquare);
		register_domain(std::move(domainCkksSquare));

		Ort::CustomOpDomain domainCkksLoader(domain_ckks_loader);
		domainCkksLoader.Add(&op_ckks_loader);
		session_options.Add(domainCkksLoader);
		register_domain(std::move(domainCkksLoader));

		Ort::CustomOpDomain domainCkksSaver(domain_ckks_saver);
		domainCkksSaver.Add(&op_ckks_saver);
		session_options.Add(domainCkksSaver);
		register_domain(std::move(domainCkksSaver));

#ifndef ORT_NO_EXCEPTIONS
  	} catch (const std::exception& e) {
    	Ort::Status status{e};

    	return status.release();
#endif
	}

	return status;
}

}	// extern "C"