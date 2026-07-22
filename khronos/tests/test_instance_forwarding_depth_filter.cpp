/** -----------------------------------------------------------------------------
 * Tests for the InstanceForwarding per-cluster depth-mode consistency filter.
 * Added 2026-07 as part of the perception-depth-filter spec: mask pixels that
 * bleed onto farther background carry farther depths into an object's TSDF
 * extent, dragging its DSG centroid outward. This filter rejects, per
 * semantic cluster, any pixel whose range deviates from the cluster's median
 * range by more than max(depth_mad_k * MAD, depth_mad_floor_m).
 * -------------------------------------------------------------------------- */

#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hydra/input/camera.h>
#include <hydra/input/sensor_extrinsics.h>
#include <hydra/reconstruction/volumetric_map.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/object_detection/instance_forwarding.h"

namespace khronos {
namespace {

constexpr int kWidth = 32;
constexpr int kHeight = 32;

// Builds a minimal valid Camera sensor purely so InputData has something to
// hold. Its projection math is never exercised by extractSemanticClusters.
std::shared_ptr<hydra::Camera> makeCamera() {
  hydra::Camera::Config config;
  config.min_range = 0.1;
  config.max_range = 20.0;
  config.width = kWidth;
  config.height = kHeight;
  config.cx = kWidth / 2.0f;
  config.cy = kHeight / 2.0f;
  config.fx = kWidth / 2.0f;
  config.fy = kHeight / 2.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, "test_camera");
}

// A pixel with a value to write into label_image (id) and range_image
// (range, possibly NaN or <=0 to model invalid returns).
struct LabeledPixel {
  int u;
  int v;
  int id;
  float range;
};

hydra::InputData makeInputData(const std::shared_ptr<hydra::Camera>& camera,
                                const std::vector<LabeledPixel>& pixels) {
  hydra::InputData input(camera);
  input.timestamp_ns = 0;
  input.world_T_body.setIdentity();
  input.label_image = cv::Mat::zeros(kHeight, kWidth, CV_32SC1);
  input.range_image = cv::Mat::zeros(kHeight, kWidth, CV_32FC1);
  input.vertex_map = cv::Mat::zeros(kHeight, kWidth, CV_32FC3);
  for (const auto& px : pixels) {
    input.label_image.at<hydra::InputData::LabelType>(px.v, px.u) = px.id;
    input.range_image.at<hydra::InputData::RangeType>(px.v, px.u) = px.range;
  }
  return input;
}

// Places `num_clean` pixels at `clean_range` and `num_contaminated` pixels at
// `contaminated_range`, all under id 1, in raster order starting at (0, 0).
std::vector<LabeledPixel> makeContaminatedCluster(int num_clean,
                                                   float clean_range,
                                                   int num_contaminated,
                                                   float contaminated_range) {
  std::vector<LabeledPixel> pixels;
  int idx = 0;
  for (int i = 0; i < num_clean; ++i, ++idx) {
    pixels.push_back({idx % kWidth, idx / kWidth, 1, clean_range});
  }
  for (int i = 0; i < num_contaminated; ++i, ++idx) {
    pixels.push_back({idx % kWidth, idx / kWidth, 1, contaminated_range});
  }
  return pixels;
}

std::set<std::pair<int, int>> pixelSet(const Pixels& pixels) {
  std::set<std::pair<int, int>> out;
  for (const auto& px : pixels) {
    out.emplace(px.u, px.v);
  }
  return out;
}

const MeasurementCluster* findCluster(const std::vector<MeasurementCluster>& clusters, int id) {
  for (const auto& cluster : clusters) {
    if (cluster.id == id) {
      return &cluster;
    }
  }
  return nullptr;
}

InstanceForwarding::Config makeConfig(bool enable_depth_mode_filter = true) {
  InstanceForwarding::Config config;
  config.enable_depth_mode_filter = enable_depth_mode_filter;
  config.depth_mad_k = 3.0f;
  config.depth_mad_floor_m = 0.15f;
  config.depth_filter_min_pixels = 10;
  return config;
}

}  // namespace

// Case 1: 70% of a cluster's pixels at range 3.0, 30% at 6.0 (contamination).
// With the filter on, the 6.0 m pixels are rejected: the surviving cluster
// contains only the 3.0 m pixels, and object_image is 0 at every rejected
// pixel.
TEST(InstanceForwardingDepthFilter, RejectsContaminationKeepsMode) {
  const auto camera = makeCamera();
  const auto pixels = makeContaminatedCluster(7, 3.0f, 3, 6.0f);
  auto input = makeInputData(camera, pixels);
  FrameData data(input);

  InstanceForwarding detector(makeConfig(/*enable_depth_mode_filter=*/true));
  const hydra::VolumetricMap::Config map_config;
  hydra::VolumetricMap map(map_config);
  detector.processInput(map, data);

  const auto* cluster = findCluster(data.semantic_clusters, 1);
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->pixels.size(), 7u);

  const auto kept = pixelSet(cluster->pixels);
  for (const auto& px : pixels) {
    const bool should_keep = px.range == 3.0f;
    EXPECT_EQ(kept.count({px.u, px.v}) > 0, should_keep)
        << "pixel (" << px.u << "," << px.v << ") range=" << px.range;
    EXPECT_EQ(data.object_image.at<FrameData::ObjectImageType>(px.v, px.u),
              should_keep ? 1 : 0)
        << "object_image at (" << px.u << "," << px.v << ")";
  }
}

// Case 2: same input, enable_depth_mode_filter=false: cluster keeps ALL
// pixels and object_image matches today's (shallow-copy) behavior.
TEST(InstanceForwardingDepthFilter, FlagOffIsIdentical) {
  const auto camera = makeCamera();
  const auto pixels = makeContaminatedCluster(7, 3.0f, 3, 6.0f);
  auto input = makeInputData(camera, pixels);
  FrameData data(input);

  InstanceForwarding detector(makeConfig(/*enable_depth_mode_filter=*/false));
  const hydra::VolumetricMap::Config map_config;
  hydra::VolumetricMap map(map_config);
  detector.processInput(map, data);

  const auto* cluster = findCluster(data.semantic_clusters, 1);
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->pixels.size(), 10u);

  // Pre-existing behavior: object_image is a SHALLOW copy of label_image, so
  // they must be the exact same underlying buffer (same data pointer).
  EXPECT_EQ(data.object_image.data, data.input.label_image.data);

  for (const auto& px : pixels) {
    EXPECT_EQ(data.object_image.at<FrameData::ObjectImageType>(px.v, px.u), 1);
  }
}

// Case 3: thick object -- uniform ramp 3.0..3.6 m: the floor term keeps
// every pixel (MAD ~0.15 -> k*MAD ~0.45 > spread/2). Assert no rejection.
TEST(InstanceForwardingDepthFilter, FloorTermProtectsThickObjects) {
  const auto camera = makeCamera();
  std::vector<LabeledPixel> pixels;
  int idx = 0;
  for (int i = 0; i < 13; ++i, ++idx) {
    const float range = 3.0f + 0.05f * static_cast<float>(i);  // 3.00 .. 3.60
    pixels.push_back({idx % kWidth, idx / kWidth, 1, range});
  }
  auto input = makeInputData(camera, pixels);
  FrameData data(input);

  InstanceForwarding detector(makeConfig(/*enable_depth_mode_filter=*/true));
  const hydra::VolumetricMap::Config map_config;
  hydra::VolumetricMap map(map_config);
  detector.processInput(map, data);

  const auto* cluster = findCluster(data.semantic_clusters, 1);
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->pixels.size(), pixels.size());

  for (const auto& px : pixels) {
    EXPECT_EQ(data.object_image.at<FrameData::ObjectImageType>(px.v, px.u), 1)
        << "pixel (" << px.u << "," << px.v << ") range=" << px.range
        << " should have been kept by the floor term";
  }
}

// Case 4: cluster smaller than depth_filter_min_pixels (e.g. 6 px, half
// contaminated): filter skipped, cluster unchanged.
TEST(InstanceForwardingDepthFilter, SmallClustersSkipFilter) {
  const auto camera = makeCamera();
  const auto pixels = makeContaminatedCluster(3, 3.0f, 3, 6.0f);
  ASSERT_EQ(pixels.size(), 6u);
  auto input = makeInputData(camera, pixels);
  FrameData data(input);

  auto config = makeConfig(/*enable_depth_mode_filter=*/true);
  ASSERT_LT(static_cast<int>(pixels.size()), config.depth_filter_min_pixels);
  InstanceForwarding detector(config);
  const hydra::VolumetricMap::Config map_config;
  hydra::VolumetricMap map(map_config);
  detector.processInput(map, data);

  const auto* cluster = findCluster(data.semantic_clusters, 1);
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->pixels.size(), 6u);

  for (const auto& px : pixels) {
    EXPECT_EQ(data.object_image.at<FrameData::ObjectImageType>(px.v, px.u), 1)
        << "small cluster should be unfiltered regardless of range";
  }
}

// Case 5: invalid ranges (0.0 / NaN) inside a cluster: excluded from the
// median AND rejected from the cluster/object_image when the filter is on.
TEST(InstanceForwardingDepthFilter, InvalidRangesRejected) {
  const auto camera = makeCamera();
  std::vector<LabeledPixel> pixels = makeContaminatedCluster(10, 3.0f, 0, 0.0f);
  int idx = static_cast<int>(pixels.size());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  pixels.push_back({idx % kWidth, idx / kWidth, 1, 0.0f});
  ++idx;
  pixels.push_back({idx % kWidth, idx / kWidth, 1, 0.0f});
  ++idx;
  pixels.push_back({idx % kWidth, idx / kWidth, 1, nan});
  ++idx;
  pixels.push_back({idx % kWidth, idx / kWidth, 1, nan});
  ASSERT_EQ(pixels.size(), 14u);

  auto input = makeInputData(camera, pixels);
  FrameData data(input);

  InstanceForwarding detector(makeConfig(/*enable_depth_mode_filter=*/true));
  const hydra::VolumetricMap::Config map_config;
  hydra::VolumetricMap map(map_config);
  detector.processInput(map, data);

  const auto* cluster = findCluster(data.semantic_clusters, 1);
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->pixels.size(), 10u);

  for (const auto& px : pixels) {
    const bool should_keep = px.range == 3.0f;
    EXPECT_EQ(data.object_image.at<FrameData::ObjectImageType>(px.v, px.u),
              should_keep ? 1 : 0)
        << "pixel (" << px.u << "," << px.v << ") range=" << px.range;
  }
}

}  // namespace khronos
