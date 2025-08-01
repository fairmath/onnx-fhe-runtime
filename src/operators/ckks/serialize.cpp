#include <cstdint>
#include <vector>

#include <openfhe/pke/cryptocontext-ser.h>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <operators/helper.h>
#include <tools/tools.h>

#include "serialize.h"

namespace operators {
namespace ckks {

CKKSLoaderKernel::CKKSLoaderKernel(OrtApi api,
                                   const OrtKernelInfo* info,
                                   std::shared_ptr<reg::CryptoRegistry> registry)
    : api_(api), reg_(registry) {}

void CKKSLoaderKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    auto cryptoCtxStr = ctx.GetInput(0).GetTensorData<std::string>();
    auto rotKeyStr = ctx.GetInput(1).GetTensorData<std::string>();
    auto mulKeyStr = ctx.GetInput(2).GetTensorData<std::string>();

    auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
    std::string* outCtx = r.GetTensorMutableData<std::string>();
    *outCtx = reg_->loadCtx(*cryptoCtxStr, *rotKeyStr, *mulKeyStr);

    r = ctx.GetOutput(1, std::vector<int64_t>{1});
    std::string* outCipher = r.GetTensorMutableData<std::string>();
    auto ciphertext = ctx.GetInput(3).GetTensorData<std::string>();
    *outCipher = reg_->loadCipher(*ciphertext);

    auto pk = ctx.GetInput(4).GetTensorData<std::string>();
    if (!pk->empty()) {
        r = ctx.GetOutput(2, std::vector<int64_t>{1});
        std::string* outPk = r.GetTensorMutableData<std::string>();
        *outPk = reg_->loadPubKey(*pk);
    }
}

CKKSSaverKernel::CKKSSaverKernel(OrtApi api,
                                 const OrtKernelInfo* info,
                                 std::shared_ptr<reg::CryptoRegistry> registry)
    : api_(api), reg_(registry) {}

void CKKSSaverKernel::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);

    auto cipher = ctx.GetInput(0).GetTensorData<std::string>();
    auto cphr = reg_->cipher(*cipher);

    auto r = ctx.GetOutput(0, std::vector<int64_t>{1});
    std::string* outCtx = r.GetTensorMutableData<std::string>();
    *outCtx = tools::rndstr(16) + ".bin";

    lbcrypto::Serial::SerializeToFile(*outCtx, cphr, lbcrypto::SerType::BINARY);
}

}  // namespace ckks
}  // namespace operators