
#pragma once

#include <string>
#include <unordered_map>

#include "openfhe/core/lattice/hal/lat-backend.h"
#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/cryptocontext-fwd.h"
#include "openfhe/pke/key/publickey-fwd.h"

namespace reg {

class CryptoRegistry final {
   public:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context(const std::string& name) const;
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipher(const std::string& name) const;
    lbcrypto::PublicKey<lbcrypto::DCRTPoly> pk(const std::string& name) const;

    std::string loadCtx(const std::string& ctxName,
                        const std::string& rotKeysName,
                        const std::string& mulKeys);
    std::string loadPubKey(const std::string& keyName);
    std::string loadCipher(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> c);
    std::string loadCipher(const std::string& name);

   private:
    std::unordered_map<std::string, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> contexts_;
    std::unordered_map<std::string, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts_;
    std::unordered_map<std::string, lbcrypto::PublicKey<lbcrypto::DCRTPoly>> pks_;
};

}  // namespace reg