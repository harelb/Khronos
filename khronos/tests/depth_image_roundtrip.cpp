// Standalone regression: compile with pkg-config --cflags --libs opencv4.
#include "khronos/utils/depth_image.h"
#include <opencv2/imgcodecs.hpp>
#include <cassert>
#include <limits>
#include <vector>

int main() {
  cv::Mat metres = (cv::Mat_<float>(1, 8) << 0.123f, 1.234f, 45.0f, 70.0f,
                    -1.0f, 0.0f, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN());
  const auto mm = khronos::depthMillimetres(metres);
  std::vector<unsigned char> bytes;
  assert(cv::imencode(".png", mm, bytes));
  const auto decoded = cv::imdecode(bytes, cv::IMREAD_UNCHANGED);
  assert(decoded.type() == CV_16UC1);
  assert(decoded.at<uint16_t>(0, 0) == 123);
  assert(decoded.at<uint16_t>(0, 1) == 1234);
  assert(decoded.at<uint16_t>(0, 2) == 45000);
  for (int col = 3; col < 8; ++col) assert(decoded.at<uint16_t>(0, col) == 0);
  assert(cv::countNonZero(khronos::depthMillimetres(decoded) != decoded) == 0);
  bool rejected = false;
  try { khronos::depthMillimetres(cv::Mat::zeros(1, 1, CV_8UC1)); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
