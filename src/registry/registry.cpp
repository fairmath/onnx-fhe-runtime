
#include <unordered_map>

#include <openfhe/pke/cryptocontext-ser.h>
#include "openfhe/core/lattice/hal/lat-backend.h"
#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/cryptocontext-fwd.h"

#include "utils/serial.h"
#include "utils/sertype.h"

#include <tools/tools.h>

#include "registry.h"

namespace reg {

lbcrypto::CryptoContext<lbcrypto::DCRTPoly> CryptoRegistry::context(const std::string& name) const {
    auto res = contexts_.find(name);

    if (res != contexts_.end()) {
        return res->second;
    }

    return lbcrypto::CryptoContext<lbcrypto::DCRTPoly>();
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> CryptoRegistry::cipher(const std::string& name) const {
    auto res = ciphertexts_.find(name);

    if (res != ciphertexts_.end()) {
        return res->second;
    }

    return lbcrypto::Ciphertext<lbcrypto::DCRTPoly>();
}

lbcrypto::PublicKey<lbcrypto::DCRTPoly> CryptoRegistry::pk(const std::string& name) const {
    auto res = pks_.find(name);

    if (res != pks_.end()) {
        return res->second;
    }

    return lbcrypto::PublicKey<lbcrypto::DCRTPoly>();
}

std::string CryptoRegistry::loadCtx(const std::string& ctxName,
                                    const std::string& rotKeysName,
                                    const std::string& mulKeys) {
    auto id = tools::rndstr(16);

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

std::string CryptoRegistry::loadPubKey(const std::string& keyName) {
    auto id = tools::rndstr(16);

    lbcrypto::PublicKey<lbcrypto::DCRTPoly> key;
    lbcrypto::Serial::DeserializeFromFile(keyName, key, lbcrypto::SerType::BINARY);

    pks_[id] = key;

    return id;
}

std::string CryptoRegistry::loadCipher(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> c) {
    auto id = tools::rndstr(16);

    ciphertexts_[id] = c;

    return id;
}

std::string CryptoRegistry::loadCipher(const std::string& name) {
    auto id = tools::rndstr(16);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cphr;
    lbcrypto::Serial::DeserializeFromFile(name, cphr, lbcrypto::SerType::BINARY);

    ciphertexts_[id] = cphr;

    return id;
}

}  // namespace reg