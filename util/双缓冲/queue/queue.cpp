#include "feature_generated.h"
#include <fstream>
#include <iostream>
#include <memory.h>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef uint32_t image_t;

class BufferDataBase {
 private:
  uint8_t* data;

 public:
  int a, b;

  BufferDataBase(/* args */);
  ~BufferDataBase() {
    if (data) free(data);
    data = NULL;
  }

  void Serialize(uint8_t*& buffer, int& length) {
    length = 8;
    if (!data) data = (uint8_t*)malloc(8);

    buffer = data;
    memcpy(buffer, &a, 4);
    memcpy(buffer + 4, &b, 4);
  }

  void Parse(const uint8_t* buffer, int length) {
    length = 8;
    memcpy(&a, buffer, 4);
    memcpy(&b, buffer + 4, 4);
  }

  static std::string name;
  static int mean_size_per_element;
};

std::string BufferDataBase::name = "data";
int BufferDataBase::mean_size_per_element = 8;

BufferDataBase::BufferDataBase() { data = nullptr; }

struct CacheInfo {
  image_t begin_idx, end_idx;
  bool is_complete;
  int idx;
};

template <typename DataType>
class DoubleBufferQueue {
 private:
  DataType data;

  uint8_t* pool_[2];
  int read_buffer_id_ = 0;
  int write_buffer_id_ = 0;

  std::vector<CacheInfo> cache_infos;

  std::unordered_map<image_t, uint32_t> feature_start_[2];
  std::unordered_map<image_t, uint32_t> feature_length_[2];

  uint32_t current_bytes_[2];
  uint32_t cache_byte;

  std::string root_path_;
  std::mutex mtx_kf;

 public:
  DoubleBufferQueue(const std::string& file_path, uint32_t cache_size) {
    cache_byte = cache_size;
    for (int i = 0; i < 2; ++i) {
      feature_start_[i].reserve(cache_byte / DataType::mean_size_per_element);
      feature_length_[i].reserve(cache_byte / DataType::mean_size_per_element);
      current_bytes_[i] = 0;
      pool_[i] = (uint8_t*)malloc(cache_byte);
    }

    if (file_path.back() == '/')
      root_path_ = file_path;
    else
      root_path_ = file_path + "/";

    read_buffer_id_ = 1;
    write_buffer_id_ = 0;

    std::ifstream fin(root_path_ + DataType::name + "_list.txt");
    if (fin.is_open()) {
      int incomplete_count = 0;
      int incomplete_idx = 0;
      while (!fin.eof()) {
        std::string str;
        getline(fin, str);
        if (str.length() < 5) continue;
        std::stringstream ss(str);
        CacheInfo info;
        ss >> info.idx >> info.is_complete >> info.begin_idx >> info.end_idx;
        if (!info.is_complete) {
          std::cout << info.idx << std::endl;
          incomplete_idx = cache_infos.size();
          incomplete_count++;
        } else {
          cache_infos.push_back(info);
        }
      }
      if (incomplete_count > 1)
        throw std::runtime_error("too many incomplete pool:" + std::to_string(incomplete_count));
      else if (incomplete_count == 1) {
        std::string path =
            root_path_ + std::to_string(incomplete_idx) + "." + DataType::name + "-bin";
        ReadFileToCache(path, write_buffer_id_);
      }
    } else {
      std::cout << ("[psx]:init DoubleBufferQueue") << std::endl;
    }
    fin.close();
  }

  ~DoubleBufferQueue() {
    Flush();
    free(pool_[0]);
    free(pool_[1]);
    ClearCache(write_buffer_id_);
    ClearCache(read_buffer_id_);
  }

  bool ReadElement(const image_t& image_id, DataType& data) {
    std::unique_lock<std::mutex> lck(mtx_kf);

    // 先尝试读写的两个队列里有没有
    if (feature_start_[write_buffer_id_].count(image_id) == 1) {
      ReadElement(write_buffer_id_, image_id, data);
      return true;
    }
    if (feature_start_[read_buffer_id_].count(image_id) == 1) {
      ReadElement(read_buffer_id_, image_id, data);
      return true;
    }

    // 在查看过去的数据，如果有那就load进内存
    int cch_idx = 0;
    bool is_hit = false;
    for (auto& info : cache_infos) {
      if (image_id >= info.begin_idx && image_id <= info.end_idx) {
        is_hit = true;
        cch_idx = info.idx;
        break;
      }
    }
    if (!is_hit) {
      std::string message = "can not find this feature : " + std::to_string(image_id);
      std::cout << message << std::endl;
    }
    ClearCache(read_buffer_id_);
    std::string path = root_path_ + std::to_string(cch_idx) + "." + DataType::name + "-bin";
    ReadFileToCache(path, read_buffer_id_);
    if (feature_start_[read_buffer_id_].count(image_id) == 1) {
      ReadElement(read_buffer_id_, image_id, data);
      return true;
    }

    std::cout << ("IO Error : impossible file path logic") << std::endl;
    return false;
  }

  void AddElement(image_t id, DataType& data) {
    uint8_t* buf;
    int cur_size;
    data.Serialize(buf, cur_size);

    if (cur_size + current_bytes_[write_buffer_id_] > cache_byte) {
      std::string path =
          root_path_ + std::to_string(cache_infos.size()) + "." + DataType::name + "-bin";
      WriteBufferToFile(path, write_buffer_id_);
      ClearCache(read_buffer_id_);
      std::swap(write_buffer_id_, read_buffer_id_);
    }
    feature_start_[write_buffer_id_].insert({id, current_bytes_[write_buffer_id_]});
    feature_length_[write_buffer_id_].insert({id, cur_size});
    memcpy(pool_[write_buffer_id_] + current_bytes_[write_buffer_id_], buf, cur_size);
    current_bytes_[write_buffer_id_] += cur_size;
  }

  void Flush() {
    std::ofstream fout(root_path_ + DataType::name + "_list.txt");
    for (auto& info : cache_infos)
      fout << info.idx << " " << info.is_complete << " " << info.begin_idx << " " << info.end_idx
           << std::endl;

    CacheInfo info;
    info.idx = cache_infos.size();
    info.is_complete = false;
    info.begin_idx = 999999;
    info.end_idx = 0;
    for (auto p : feature_start_[write_buffer_id_]) {
      info.begin_idx = std::min(info.begin_idx, p.first);
      info.end_idx = std::max(info.end_idx, p.first);
    }
    fout << info.idx << " " << info.is_complete << " " << info.begin_idx << " " << info.end_idx
         << std::endl;
    fout.close();

    std::string path =
        root_path_ + std::to_string(cache_infos.size()) + "." + DataType::name + "-bin";
    WriteBufferToFile(path, write_buffer_id_, false);
  }

 private:
  void WriteBufferToFile(const std::string& file_path, int pool_id, bool is_complete = true) {
    if (feature_start_[pool_id].empty()) return;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<image_t> image_ids;
    std::vector<uint32_t> feature_starts;
    std::vector<uint32_t> feature_lengths;
    for (auto p : feature_start_[pool_id]) {
      image_ids.push_back(p.first);
      feature_starts.push_back(p.second);
      feature_lengths.push_back(feature_length_[pool_id][p.first]);
    }

    auto image_id__ = builder.CreateVector<uint32_t>(image_ids);
    auto feature_start__ = builder.CreateVector<uint32_t>(feature_starts);
    auto feature_length__ = builder.CreateVector<uint32_t>(feature_lengths);

    auto feature_data = builder.CreateVector(pool_[pool_id], current_bytes_[pool_id]);
    auto buffer = pico_MR::flat::CreateFeatureBundle(builder, image_id__, feature_start__,
                                                     feature_length__, feature_data);

    builder.Finish(buffer);
    std::ofstream file(file_path);
    file.write((char*)builder.GetBufferPointer(), builder.GetSize());
    file.close();

    CacheInfo info;
    info.idx = cache_infos.size();
    info.is_complete = is_complete;
    info.begin_idx = 999999;
    info.end_idx = 0;
    for (auto p : feature_start_[pool_id]) {
      info.begin_idx = std::min(info.begin_idx, p.first);
      info.end_idx = std::max(info.end_idx, p.first);
    }
    if (is_complete) cache_infos.push_back(info);
  }

  void ClearCache(int hit_id) {
    feature_start_[hit_id].clear();
    current_bytes_[hit_id] = 0;
    feature_length_[hit_id].clear();
  }

  uint32_t ReadFileToCache(const std::string& file_path, int hit_cache_id) {
    std::ifstream infile(file_path, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    char* data = new char[length];
    infile.read(data, length);
    infile.close();
    const pico_MR::flat::FeatureBundle* feature_bundle = pico_MR::flat::GetFeatureBundle(data);
    const uint8_t* feature_data_bunlde = feature_bundle->feature_data()->data();
    uint32_t feature_data_bundle_size = feature_bundle->feature_data()->size();
    memcpy(pool_[hit_cache_id], feature_data_bunlde, feature_data_bundle_size);
    feature_start_[hit_cache_id].clear();
    feature_length_[hit_cache_id].clear();

    for (unsigned int i = 0; i < feature_bundle->image_id()->size(); ++i) {
      uint32_t begin = feature_bundle->feature_start()->Get(i);
      uint32_t len = feature_bundle->feature_length()->Get(i);
      image_t id = feature_bundle->image_id()->Get(i);
      feature_start_[hit_cache_id][id] = begin;
      feature_length_[hit_cache_id][id] = len;
    }

    current_bytes_[hit_cache_id] = feature_bundle->feature_data()->size();
    delete[] data;
    return feature_data_bundle_size;
  }

  bool ReadElement(int buffer_type, image_t image_id, DataType& data) {
    int begin = feature_start_[buffer_type][image_id];
    int len = feature_length_[buffer_type][image_id];
    const pico_MR::flat::KeyFrame* kf = pico_MR::flat::GetKeyFrame(pool_[buffer_type] + begin);
    data.Parse(pool_[buffer_type] + begin, len);
    return true;
  }
};

int main(int argc, char** argv) {
  DoubleBufferQueue<BufferDataBase> q("/home/psx/data/optitrack/psx_data/eps_small/tmp", 1024);

  for (int i = 0; i < 2000; i++) {
    BufferDataBase data;
    data.a = i;
    data.b = i + 1;
    q.AddElement(i, data);

    if (i < 100) continue;
    int query_idx = rand() % i;
    BufferDataBase query_data;
    q.ReadElement(query_idx, query_data);
    std::cout << "query=" << query_idx << " : " << query_data.a << " " << query_data.b << std::endl;
  }

  q.Flush();
  q.Flush();
  q.Flush();
  q.Flush();
  q.Flush();
  return 0;
}