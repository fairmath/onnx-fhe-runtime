#include <string>
#include <random>

#include "tools.h"

namespace tools {

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

}