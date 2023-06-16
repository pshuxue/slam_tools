#pragma once
#include "common/util/image_info.h"
#include "common/util/types.h"
#include "hdmap/util/timer.h"
#include <Eigen/Geometry>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

namespace mr_engine {
namespace psx_hdmap {

struct GridInfo {
  typedef std::shared_ptr<GridInfo> Ptr;
  uint32_t image_id_;
  float x_, pitch_, roll_;
  unsigned int yaw_idx_;
  Eigen::Matrix3f Rwc;
  Eigen::Vector3f twc;
  explicit GridInfo(const uint32_t& image_id, const Eigen::Matrix4f& Twc, float x, float pitch,
                    float roll, unsigned int yaw_idx);
};

class Node {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Node(const Eigen::Vector2f& min_b, const Eigen::Vector2f& max_b);
  Node() {}

  float Size();
  Eigen::AlignedBox2f top_box;
  std::vector<uchar> angle_grids;
  std::vector<GridInfo::Ptr> infos;

  std::vector<Node*> nodes;
};

class QuadTree {
 public:
  DEFINE_POINTER_TYPE(QuadTree);
  explicit QuadTree(const float grid_len, const float yaw_angle_th, const float pitch_angle_th,
                    const float roll_angle_th, const float vertical_offset_th);
  ~QuadTree();

  void AddElement(const uint32_t& id, const Eigen::Matrix4f& Twc);
  int IsOccupied(const Eigen::Matrix4f& Twc);
  std::vector<uint32_t> GetClosetImages(const Eigen::Matrix4f& Twc, const int num_images);
  std::vector<Node*> Traverse();

  void Init(const Eigen::Matrix3f& K);

 private:
  float Score(float d_angle, float d_dis);

  std::unordered_set<uint32_t> image_set;

  Node* root;

  int num_grid_angle;

  float horizon_grid_len_;
  float yaw_angle_th_;
  float vertical_offset_th_;
  float pitch_angle_th_, roll_angle_th_;

  const float scene_length = 1024;
  float fov;
  float fx, fy, cx, cy;
};

class PoseGridManager {
 private:
  QuadTree::Ptr tree_;

  // 空间格子大小
  const float grid_len = 1;
  const float grid_angle = 30;
  const float pitch_angle_th = 45;
  const float roll_angle_th = 45;
  const float vertical_offset_th = 1;

  const float kf_t_th = 0.3;
  const float kf_theta_th = 10;

  // debug
  const int cell_size = 50;

 private:
  void AddToGrids(const Eigen::Matrix4f& Twb);

 public:
  DEFINE_POINTER_TYPE(PoseGridManager);
  explicit PoseGridManager(const std::string& map_folder, bool is_new_map = true);

  ~PoseGridManager();

  int CheckKeyFrame(const mr_engine::common::ImageInfo& image_info);
  int CheckKeyFrame(const Eigen::Matrix4f& Twc);
  void RegisterImage(const uint32_t image_id, const Eigen::Matrix4f& Twc);
  std::vector<uint32_t> GetClosetImages(const Eigen::Matrix4f& Twc, const int num_images);

  void SaveGrid();
  void SaveGridPicture();

  void Init(const Eigen::Matrix3f& K);
  void Init(const std::vector<float>& K_arr);

 private:
  bool is_new_map_;
  Eigen::Matrix4f Twb_last_;
  float angle_tf_, position_tf_;
  std::string map_folder_;
  bool is_initialized_;
};

}  // namespace psx_hdmap
}  // namespace mr_engine