
#pragma once

#include <string>
#include <unordered_map>

#include "openfhe/pke/ciphertext-fwd.h"
#include "openfhe/pke/cryptocontext-fwd.h"
#include "openfhe/core/lattice/hal/lat-backend.h"


namespace reg {

class CryptoRegistry final {
public:
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context(std::string name) const;
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cipher(std::string name) const;

    std::string loadCtx(std::string ctxName, std::string rotKeysName, std::string mulKeys);
	std::string loadCipher(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> c);
    std::string loadCipher(std::string name);

private:
	std::unordered_map<std::string, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>> contexts_;
	std::unordered_map<std::string, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ciphertexts_;
};

}