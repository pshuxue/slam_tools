#include "psx_hdmap/estimators/pose_grid_manager.h"

namespace mr_engine {
namespace psx_hdmap {

void TransPoseToYZYaw(const Eigen::Matrix4f& Twb, float& y, float& z, float& yaw) {
  y = Twb(1, 3);
  z = Twb(2, 3);

  yaw = atan2(Twb(2, 1), Twb(2, 2));
  yaw = yaw * 180 / M_PI;
  if (yaw < 0) yaw += 360;
}

void TransPoseToXYZAngle(const Eigen::Matrix4f& Twb, float& x, float& y, float& z, float& yaw,
                         float& pitch, float& roll) {
  x = Twb(0, 3);
  y = Twb(1, 3);
  z = Twb(2, 3);
  Eigen::Matrix3f Rwb = Twb.topLeftCorner(3, 3);

  roll = atan2(Twb(1, 0), Twb(0, 0));
  pitch = atan2(-Twb(2, 0), sqrt(Twb(2, 1) * Twb(2, 1) + Twb(2, 2) * Twb(2, 2)));
  yaw = atan2(Twb(2, 1), Twb(2, 2));
  yaw = yaw * 180 / M_PI;
  roll = roll * 180 / M_PI;
  pitch = pitch * 180 / M_PI;
  if (yaw < 0) yaw += 360;
  if (roll < 0) roll += 360;
  if (pitch < 0) pitch += 360;
}

GridInfo::GridInfo(const uint32_t& image_id, const Eigen::Matrix4f& Twc, float x, float pitch,
                   float roll, unsigned int yaw_idx) {
  image_id_ = image_id;
  x_ = x;
  pitch_ = pitch;
  roll_ = roll;
  yaw_idx_ = yaw_idx;
  Rwc = Twc.topLeftCorner(3, 3);
  twc = Twc.topRightCorner(3, 1);
}

Node::Node(const Eigen::Vector2f& min_b, const Eigen::Vector2f& max_b) {
  top_box = Eigen::AlignedBox2f(min_b, max_b);

  Eigen::Vector2f cen_b = (min_b + max_b) / 2;
  Node* node0 = new Node();
  node0->top_box = Eigen::AlignedBox2f(min_b, cen_b);
  Node* node1 = new Node();
  node1->top_box = Eigen::AlignedBox2f(Eigen::Vector2f(min_b.x(), cen_b.y()),
                                       Eigen::Vector2f(cen_b.x(), max_b.y()));
  Node* node2 = new Node();
  node2->top_box = Eigen::AlignedBox2f(Eigen::Vector2f(cen_b.x(), min_b.y()),
                                       Eigen::Vector2f(max_b.x(), cen_b.y()));
  Node* node3 = new Node();
  node3->top_box = Eigen::AlignedBox2f(cen_b, max_b);
  nodes.push_back(node0);
  nodes.push_back(node1);
  nodes.push_back(node2);
  nodes.push_back(node3);
  angle_grids.clear();
  infos.clear();
}

float Node::Size() { return top_box.max().x() - top_box.min().x(); }

void QuadTree::Init(const Eigen::Matrix3f& K) {
  fx = K(0, 0);
  fy = K(1, 1);
  cx = K(0, 2);
  cy = K(1, 2);
  fov = atan(cx / fx);
  fov = fov * 180 / M_PI;
  fov *= 2;
}

float QuadTree::Score(float d_angle, float d_dis) {
  float angle_rest_area = 1 - d_angle / fov;
  float dis_rest_area = 1 - d_dis / (cx / fx * 10);

  return angle_rest_area * dis_rest_area;
}

std::vector<uint32_t> QuadTree::GetClosetImages(const Eigen::Matrix4f& Twc, const int num_images) {
  Eigen::Matrix3f cur_Rwc = Twc.topLeftCorner(3, 3);
  Eigen::Vector3f cur_twc = Twc.topRightCorner(3, 1);
  float y, z, yaw;
  TransPoseToYZYaw(Twc, y, z, yaw);

  std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> points;
  points.emplace_back(y, z);
  points.emplace_back(y - horizon_grid_len_, z);
  points.emplace_back(y + horizon_grid_len_, z);
  points.emplace_back(y, z - horizon_grid_len_);
  points.emplace_back(y, z + horizon_grid_len_);
  points.emplace_back(y - horizon_grid_len_, z + horizon_grid_len_);
  points.emplace_back(y - horizon_grid_len_, z - horizon_grid_len_);
  points.emplace_back(y + horizon_grid_len_, z + horizon_grid_len_);
  points.emplace_back(y + horizon_grid_len_, z - horizon_grid_len_);

  std::vector<Node*> nodes;
  for (auto& point : points) {
    Node* node_iter = root;
    while (node_iter->Size() > horizon_grid_len_) {
      bool flag = true;
      for (auto& n : node_iter->nodes) {
        if (n->top_box.contains(point)) {
          node_iter = n;
          flag = false;
          break;
        }
      }

      if (flag) break;

      if (node_iter->infos.size() > 0) {
        nodes.push_back(node_iter);
        break;
      }
    }
  }

  std::vector<std::pair<float, uint32_t>> score_images;
  for (auto& node : nodes) {
    for (auto& info : node->infos) {
      Eigen::Matrix3f dR = info->Rwc.transpose() * cur_Rwc;
      Eigen::Vector3f dt = cur_Rwc.transpose() * (info->twc - cur_twc);
      float theta = std::acos((dR.trace() - 1) / 2) * 180 / M_PI;  // 弧度制转角度制
      float d_dis = dt.head(2).norm();
      float s = Score(theta, d_dis);
      if (s > 0.4) score_images.emplace_back(s, info->image_id_);
    }
  }

  if (score_images.size() == 0) return std::vector<uint32_t>();

  std::sort(score_images.begin(), score_images.end(),
            [&](std::pair<float, uint32_t> a, std::pair<float, uint32_t> b) {
              return a.first > b.first;
            });

  size_t size = std::min(score_images.size(), size_t(num_images));
  std::vector<uint32_t> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(score_images[i].second);
  }
  return result;
}

QuadTree::QuadTree(const float grid_len, const float yaw_angle_th, const float pitch_angle_th,
                   const float roll_angle_th, const float vertical_offset_th) {
  horizon_grid_len_ = grid_len;
  yaw_angle_th_ = yaw_angle_th;
  float edge = scene_length / 2.0;
  num_grid_angle = roundf(360.0 / yaw_angle_th);
  root = new Node(Eigen::Vector2f(-edge, -edge), Eigen::Vector2f(edge, edge));
  pitch_angle_th_ = pitch_angle_th;
  roll_angle_th_ = roll_angle_th;
  vertical_offset_th_ = vertical_offset_th;
}

std::vector<Node*> QuadTree::Traverse() {
  std::queue<Node*> queue;
  std::vector<Node*> result;
  queue.push(root);

  while (!queue.empty()) {
    Node* node = queue.front();
    queue.pop();
    if (node->angle_grids.size() > 0) result.push_back(node);
    for (auto& son : node->nodes) {
      if (son) queue.push(son);
    }
  }
  return result;
}

void QuadTree::AddElement(const uint32_t& id, const Eigen::Matrix4f& Twc) {
  if (image_set.count(id) == 1) {
    std::cout << "element " << id << " already added" << std::endl;
    return;
  }

  float x, y, z, yaw, pitch, roll;
  TransPoseToXYZAngle(Twc, x, y, z, yaw, pitch, roll);

  Node* node_iter = root;
  unsigned int angle_idx = floor(yaw / yaw_angle_th_);

  Eigen::Vector2f point(y, z);
  while (node_iter->Size() > horizon_grid_len_) {
    for (auto& n : node_iter->nodes) {
      if (n->top_box.contains(point)) {
        if (n->nodes.size() == 0) {
          Eigen::Vector2f min_b = n->top_box.min();
          Eigen::Vector2f max_b = n->top_box.max();
          Eigen::Vector2f cen_b = (min_b + max_b) / 2;
          Node* node0 = new Node();
          node0->top_box = Eigen::AlignedBox2f(min_b, cen_b);
          Node* node1 = new Node();
          node1->top_box = Eigen::AlignedBox2f(Eigen::Vector2f(min_b.x(), cen_b.y()),
                                               Eigen::Vector2f(cen_b.x(), max_b.y()));
          Node* node2 = new Node();
          node2->top_box = Eigen::AlignedBox2f(Eigen::Vector2f(cen_b.x(), min_b.y()),
                                               Eigen::Vector2f(max_b.x(), cen_b.y()));
          Node* node3 = new Node();
          node3->top_box = Eigen::AlignedBox2f(cen_b, max_b);
          n->nodes.push_back(node0);
          n->nodes.push_back(node1);
          n->nodes.push_back(node2);
          n->nodes.push_back(node3);
          n->angle_grids.clear();
          n->infos.clear();
        }
        node_iter = n;
        break;
      }
    }

    if (node_iter->angle_grids.size() > 0) {
      if (angle_idx >= node_iter->angle_grids.size()) throw std::runtime_error("node idx error");
      node_iter->infos.emplace_back(std::make_shared<GridInfo>(id, Twc, x, pitch, roll, angle_idx));
      node_iter->angle_grids[angle_idx] += 1;
      break;
    } else if (node_iter->Size() <= horizon_grid_len_) {
      if (node_iter->nodes.size() == 0) {
      }
      node_iter->infos.emplace_back(std::make_shared<GridInfo>(id, Twc, x, pitch, roll, angle_idx));
      node_iter->angle_grids.resize(num_grid_angle, 0);
      node_iter->angle_grids[angle_idx] += 1;
      break;
    }
  }

  image_set.emplace(id);
}

int QuadTree::IsOccupied(const Eigen::Matrix4f& Twc) {
  int result = 0;
  Node* node_iter = root;
  float x, y, z, yaw, pitch, roll;
  TransPoseToXYZAngle(Twc, x, y, z, yaw, pitch, roll);
  unsigned int angle_idx = floor(yaw / yaw_angle_th_);

  Eigen::Vector2f point(y, z);
  while (node_iter->Size() > horizon_grid_len_) {
    for (auto& n : node_iter->nodes) {
      if (n->top_box.contains(point)) {
        node_iter = n;
        break;
      }
    }

    if (node_iter->nodes.size() == 0) {
      return 0;
    }

    if (node_iter->angle_grids.size() > 0) {
      if (angle_idx >= node_iter->angle_grids.size()) throw std::runtime_error("node idx error");
      result = node_iter->angle_grids[angle_idx];

      bool is_roll_gap_per_grid = true, is_pitch_gap_per_grid = true, is_x_gap_per_grid = true;
      for (auto& info : node_iter->infos) {
        if (info->yaw_idx_ != angle_idx) continue;
        if (abs(info->x_ - x) < vertical_offset_th_) is_x_gap_per_grid = false;
        if (abs(info->roll_ - roll) < roll_angle_th_) is_roll_gap_per_grid = false;
        if (abs(info->pitch_ - pitch) < pitch_angle_th_) is_pitch_gap_per_grid = false;
      }
      if (is_roll_gap_per_grid || is_pitch_gap_per_grid || is_x_gap_per_grid) result = 0;
      break;
    }
  }
  return result;
}

QuadTree::~QuadTree() {}

void PoseGridManager::Init(const Eigen::Matrix3f& K) {
  if (is_initialized_) return;
  tree_->Init(K);
  is_initialized_ = true;
}

void PoseGridManager::Init(const std::vector<float>& K_arr) {
  if (is_initialized_) return;
  Eigen::Matrix3f KK;
  KK << K_arr[0], 0, K_arr[2], 0, K_arr[1], K_arr[3], 0, 0, 1;
  tree_->Init(KK);
  is_initialized_ = true;
}

std::vector<uint32_t> PoseGridManager::GetClosetImages(const Eigen::Matrix4f& Twc,
                                                       const int num_images) {
  return tree_->GetClosetImages(Twc, num_images);
}

PoseGridManager::PoseGridManager(const std::string& map_folder, bool is_new_map) {
  is_new_map_ = is_new_map;
  is_initialized_ = false;
  map_folder_ = map_folder;

  tree_ = std::make_shared<QuadTree>(grid_len, grid_angle, pitch_angle_th, roll_angle_th,
                                     vertical_offset_th);

  if (!is_new_map) {
    std::ifstream fin(map_folder_ + "/grid.bin", std::ios::binary);
    int size;
    fin.read(reinterpret_cast<char*>(&size), 4);
    for (int i = 0; i < size; ++i) {
      Eigen::Matrix3f R;
      Eigen::Vector3f t;
      uint32_t id;
      fin.read(reinterpret_cast<char*>(&id), 4);
      fin.read(reinterpret_cast<char*>(R.data()), 4 * 9);
      fin.read(reinterpret_cast<char*>(t.data()), 4 * 3);
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.topLeftCorner(3, 3) = R;
      T.topRightCorner(3, 1) = t;
      tree_->AddElement(id, T);
    }
  }
}

PoseGridManager::~PoseGridManager() {}

int PoseGridManager::CheckKeyFrame(const mr_engine::common::ImageInfo& image_info) {
  int keyframe_flag = 0;
  Eigen::Matrix4f Tlast_now = Twb_last_.inverse() * image_info.Twb;
  Eigen::Matrix3f R = Tlast_now.block<3, 3>(0, 0);
  Eigen::Vector3f delta_T = Tlast_now.block<3, 1>(0, 3);

  double theta = std::acos((R.trace() - 1) / 2) * 180 / M_PI;  // 弧度制转角度制
  if (theta < kf_theta_th && delta_T.norm() < kf_t_th) {
    return 0;
  }

  for (int i = 0; i < image_info.Tbcs.size(); i++) {
    Eigen::Matrix4f Twc = image_info.Twb * image_info.Tbcs[i];
    if (!tree_->IsOccupied(Twc)) {
      return 1;
    }
  }
  return 0;
}

int PoseGridManager::CheckKeyFrame(const Eigen::Matrix4f& Twc) {
  if (!tree_->IsOccupied(Twc)) {
    return 1;
  }
  return 0;
}

void PoseGridManager::RegisterImage(const uint32_t image_id, const Eigen::Matrix4f& Twc) {
  tree_->AddElement(image_id, Twc);
}

void PoseGridManager::SaveGrid() {
  const std::string path = map_folder_ + "/grid.bin";
  std::ofstream fout(path, std::ios::binary);
  std::vector<GridInfo::Ptr> grid_infos;

  auto nodes = tree_->Traverse();
  for (auto& node : nodes) {
    for (auto& info : node->infos) {
      grid_infos.push_back(info);
    }
  }

  int size = grid_infos.size();
  fout.write(reinterpret_cast<const char*>(&size), 4);
  for (auto& info : grid_infos) {
    fout.write(reinterpret_cast<const char*>(&info->image_id_), 4);
    fout.write(reinterpret_cast<const char*>(info->Rwc.data()), 4 * 9);
    fout.write(reinterpret_cast<const char*>(info->twc.data()), 4 * 3);
  }
  fout.close();
}

void PoseGridManager::SaveGridPicture() {
  cv::Vec3b color1(10, 10, 200);
  cv::Vec3b color2(0, 200, 10);
  cv::Vec3b color3(200, 20, 10);
  cv::Vec3b color4(0, 0, 30);

  const std::string path = map_folder_ + "/grid.png";

  std::vector<Eigen::Vector2f> directions;
  std::vector<cv::Mat> cells;
  for (float an = 0; an < 359; an += grid_angle) {
    float an_rad = an / 180.0 * M_PI;
    directions.push_back(Eigen::Vector2f(cosf(an_rad), sin(an_rad)));
    cv::Mat img = cv::Mat::zeros(cell_size, cell_size, CV_8UC1);
    cells.push_back(img);
  }
  directions.push_back(directions[0]);

  Eigen::Vector2f center(float(cell_size - 1) / 2.0, float(cell_size - 1) / 2.0);
  for (int x = 0; x < cell_size; ++x) {
    for (int y = 0; y < cell_size; ++y) {
      Eigen::Vector2f cur_d(float(x) - center.x(), float(y) - center.y());
      if (cur_d.norm() > cell_size / 2.0) continue;
      float weight = cur_d.norm() / (cell_size / 2);
      cur_d = cur_d.normalized();
      for (unsigned int i = 0; i < directions.size() - 1; i++) {
        Eigen::Vector2f& d1 = directions[i];
        Eigen::Vector2f& d2 = directions[i + 1];
        float cos_m = d1.dot(d2);
        float cos_1 = d1.dot(cur_d);
        float cos_2 = d2.dot(cur_d);
        if (cos_1 >= cos_m && cos_2 >= cos_m) {
          // cells[i].at<uchar>(y, x) = 200 - 200 * weight;
          cells[i].at<uchar>(y, x) = 1;
          break;
        }
      }
    }
  }

  std::vector<int> xs, ys;
  for (int x = 0; x < cell_size; ++x) {
    xs.push_back(x);
    ys.push_back(0);
  }
  for (int y = 0; y < cell_size; ++y) {
    xs.push_back(cell_size - 1);
    ys.push_back(y);
  }
  for (int x = cell_size - 1; x >= 0; --x) {
    xs.push_back(x);
    ys.push_back(cell_size - 1);
  }
  for (int y = cell_size - 1; y >= 0; --y) {
    xs.push_back(0);
    ys.push_back(y);
  }

  for (unsigned int i = 0; i < directions.size() - 1; i++) {
    for (unsigned int j = 0; j < xs.size(); ++j) {
      int x = xs[j];
      int y = ys[j];
      Eigen::Vector2f cur_d(float(x) - center.x(), float(y) - center.y());
      cur_d = cur_d.normalized();
      Eigen::Vector2f& d1 = directions[i];
      float cos_1 = d1.dot(cur_d);
      if (1.0 - cos_1 < 0.001) {
        cv::line(cells[i], cv::Point2i(center.x(), center.y()), cv::Point2i(x, y), 250);
        break;
      }
    }
  }

  for (auto& c : cells) {
    cv::rotate(c, c, cv::ROTATE_90_COUNTERCLOCKWISE);
  }

  std::vector<Node*> nodes = tree_->Traverse();
  Eigen::AlignedBox2f boundary(Eigen::Vector2f::Zero(), Eigen::Vector2f::Zero());
  for (auto& node : nodes) {
    boundary = boundary.merged(node->top_box);
  }

  float x_min = boundary.min().x();
  float y_min = boundary.min().y();
  float x_max = boundary.max().x();
  float y_max = boundary.max().y();
  int width = cell_size * (x_max - x_min) / grid_len;
  int height = cell_size * (y_max - y_min) / grid_len;
  if (width < 4) width = 4;

  cv::Mat traj_image(height, width, CV_8UC3);
  cv::Mat flag_image(height, width, CV_8UC1);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      traj_image.at<cv::Vec3b>(i, j)[0] = 0;
      traj_image.at<cv::Vec3b>(i, j)[1] = 0;
      traj_image.at<cv::Vec3b>(i, j)[2] = 0;
      flag_image.at<uchar>(i, j) = 0;
    }

  for (auto& node : nodes) {
    cv::Mat cell = cv::Mat::zeros(cell_size, cell_size, CV_8UC1);
    for (unsigned int i = 0; i < node->angle_grids.size(); ++i) {
      if (node->angle_grids[i] == 0) continue;
      for (int row = 0; row < cell_size; ++row)
        for (int col = 0; col < cell_size; ++col) {
          if (cells[i].at<uchar>(row, col) == 250) {
            cell.at<uchar>(row, col) =
                std::max(cell.at<uchar>(row, col), cells[i].at<uchar>(row, col));
          } else {
            uchar c = node->angle_grids[i] * cells[i].at<uchar>(row, col);
            cell.at<uchar>(row, col) = std::max(cell.at<uchar>(row, col), c);
          }
        }
    }

    int begin_x = cell_size * (node->top_box.min().x() - x_min) / grid_len;
    int begin_y = cell_size * (node->top_box.min().y() - y_min) / grid_len;
    for (int i = 0; i < cell_size; i++)
      for (int j = 0; j < cell_size; j++) {
        traj_image.at<cv::Vec3b>(i + begin_y, j + begin_x)[2] = cell.at<uchar>(i, j);
        flag_image.at<uchar>(i + begin_y, j + begin_x) = cell.at<uchar>(i, j);
        if (cell.at<uchar>(i, j) == 250) {
          traj_image.at<cv::Vec3b>(i + begin_y, j + begin_x)[1] = 80;
          traj_image.at<cv::Vec3b>(i + begin_y, j + begin_x)[2] = 80;
          traj_image.at<cv::Vec3b>(i + begin_y, j + begin_x)[0] = 80;
        }
      }
  }

  for (int i = cell_size; i < traj_image.rows; i += cell_size)
    cv::line(traj_image, cv::Point(0, i), cv::Point(traj_image.cols - 1, i), cv::Scalar(0, 10, 0));
  for (int i = cell_size; i < traj_image.cols; i += cell_size)
    cv::line(traj_image, cv::Point(i, 0), cv::Point(i, traj_image.rows - 1), cv::Scalar(0, 10, 0));

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      if (traj_image.at<cv::Vec3b>(i, j)[0] == 0 && traj_image.at<cv::Vec3b>(i, j)[1] == 0 &&
          traj_image.at<cv::Vec3b>(i, j)[2] == 0) {
        traj_image.at<cv::Vec3b>(i, j)[0] = 250;
        traj_image.at<cv::Vec3b>(i, j)[1] = 250;
        traj_image.at<cv::Vec3b>(i, j)[2] = 250;
      }

      if (flag_image.at<uchar>(i, j) == 1) {
        traj_image.at<cv::Vec3b>(i, j) = color1;
      } else if (flag_image.at<uchar>(i, j) == 2) {
        traj_image.at<cv::Vec3b>(i, j) = color2;
      } else if (flag_image.at<uchar>(i, j) == 3) {
        traj_image.at<cv::Vec3b>(i, j) = color3;
      } else if (flag_image.at<uchar>(i, j) > 3 && flag_image.at<uchar>(i, j) < 100) {
        traj_image.at<cv::Vec3b>(i, j) = color4;
      }
    }

  cv::Mat label(cell_size * 1.5, traj_image.cols, CV_8UC3);
  label = cv::Vec3b(250, 250, 250);

  int row_begin = cell_size / 4;
  int col_begin = cell_size / 4;
  for (int i = 0; i < cell_size / 2; i++) {
    for (int j = 0; j < cell_size / 2; ++j) {
      label.at<cv::Vec3b>(row_begin + i, col_begin + j) = color1;
    }
  }

  row_begin = cell_size / 4;
  col_begin = cell_size;
  for (int i = 0; i < cell_size / 2; i++) {
    for (int j = 0; j < cell_size / 2; ++j) {
      label.at<cv::Vec3b>(row_begin + i, col_begin + j) = color2;
    }
  }

  row_begin = cell_size / 4;
  col_begin = cell_size / 4 * 7;
  for (int i = 0; i < cell_size / 2; i++) {
    for (int j = 0; j < cell_size / 2; ++j) {
      label.at<cv::Vec3b>(row_begin + i, col_begin + j) = color3;
    }
  }

  row_begin = cell_size / 4;
  col_begin = cell_size / 4 * 10;
  for (int i = 0; i < cell_size / 2; i++) {
    for (int j = 0; j < cell_size / 2; ++j) {
      label.at<cv::Vec3b>(row_begin + i, col_begin + j) = color4;
    }
  }
  cv::putText(label, "1", cv::Point(cell_size / 4 + 5, cell_size / 4 * 4.5), 1, 1.5,
              cv::Scalar(30, 30, 30));
  cv::putText(label, "2", cv::Point(cell_size + 5, cell_size / 4 * 4.5), 1, 1.5,
              cv::Scalar(30, 30, 30));
  cv::putText(label, "3", cv::Point(cell_size / 4 * 7 + 5, cell_size / 4 * 4.5), 1, 1.5,
              cv::Scalar(30, 30, 30));
  cv::putText(label, ">3", cv::Point(cell_size / 4 * 10, cell_size / 4 * 4.5), 1, 1.5,
              cv::Scalar(30, 30, 30));
  cv::vconcat(traj_image, label, label);
  cv::imwrite(path, label);
}
}  // namespace psx_hdmap
}  // namespace mr_engine