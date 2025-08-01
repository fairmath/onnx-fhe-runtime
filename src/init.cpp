
#include <tuple>
#include <vector>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <operators/operators.h>
#include <registry/registry.h>
#include <tools/tools.h>

namespace {

std::shared_ptr<reg::CryptoRegistry> reg_ = std::make_shared<reg::CryptoRegistry>();

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
static const char* domain_ckks_svm = "fhe.ckks.svm";

operators::ckks::CustomOpCKKSMatMul op_ckks_matmul(reg_);
operators::ckks::CustomOpCKKSSquare op_ckks_square(reg_);
operators::ckks::CustomOpCKKSLoader op_ckks_loader(reg_);
operators::ckks::CustomOpCKKSSaver op_ckks_saver(reg_);
operators::ckks::CustomOpCKKSSvm op_ckks_svm(reg_);

std::vector<std::tuple<const char*, OrtCustomOp*>> op_list = {
    std::make_tuple(domain_ckks_matmul, &op_ckks_matmul),
    std::make_tuple(domain_ckks_square, &op_ckks_square),
    std::make_tuple(domain_ckks_loader, &op_ckks_loader),
    std::make_tuple(domain_ckks_saver, &op_ckks_saver),
    std::make_tuple(domain_ckks_svm, &op_ckks_svm)};

}  // namespace

extern "C" {

// call from SessionOptions.register_custom_ops_library()
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
    OrtStatus* status = nullptr;

#ifndef ORT_NO_EXCEPTIONS
    try {
#endif
        Ort::UnownedSessionOptions session_options(options);

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

        Ort::CustomOpDomain domainCkksSvm(domain_ckks_svm);
        domainCkksSvm.Add(&op_ckks_svm);
        session_options.Add(domainCkksSvm);
        register_domain(std::move(domainCkksSvm));

#ifndef ORT_NO_EXCEPTIONS
    } catch (const std::exception& e) {
        Ort::Status status{e};

        return status.release();
#endif
    }

    return status;
}

}  // extern "C"