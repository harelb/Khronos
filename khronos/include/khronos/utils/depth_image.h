#pragma once

#include <cmath>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace khronos {

// PNG cannot represent float metres. Zero denotes invalid/unrepresentable depth;
// do not saturate far surfaces into a fabricated surface at 65.535 m.
inline cv::Mat depthMillimetres(const cv::Mat& depth) {
  if (depth.type() == CV_16UC1) return depth.clone();
  if (depth.type() != CV_32FC1) {
    throw std::invalid_argument("Depth must be float metres or uint16 millimetres");
  }
  cv::Mat result = cv::Mat::zeros(depth.size(), CV_16UC1);
  for (int row = 0; row < depth.rows; ++row) {
    for (int col = 0; col < depth.cols; ++col) {
      const double mm = std::round(static_cast<double>(depth.at<float>(row, col)) * 1000.0);
      if (std::isfinite(mm) && mm > 0 && mm <= 65535) {
        result.at<uint16_t>(row, col) = static_cast<uint16_t>(mm);
      }
    }
  }
  return result;
}

}  // namespace khronos
