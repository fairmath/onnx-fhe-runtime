#pragma once

#include <cstdint>
#include <vector>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

template <typename TValue>
ONNXTensorElementDataType GetTypeAndShape(const TValue& input, std::vector<int64_t>& shape, bool swap = false) {
  auto t = input.GetTensorTypeAndShapeInfo();
  shape = t.GetShape();

  if (swap) {
    std::swap(shape[0], shape[1]);
  }

  return t.GetElementType();
}