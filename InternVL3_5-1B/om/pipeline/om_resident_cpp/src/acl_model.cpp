#include "acl_model.h"

#include "acl/acl.h"

#include <dirent.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("missing input bin: " + path);
  }
  in.seekg(0, std::ios::end);
  const std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("read failed: " + path);
  }
  return data;
}

void write_file(const std::string& path, const void* data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("write failed: " + path);
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

void ensure_dir(const std::string& path) {
  mkdir(path.c_str(), 0755);
}

bool has_output_bins(const std::string& output_dir) {
  DIR* dir = opendir(output_dir.c_str());
  if (!dir) {
    return false;
  }
  bool found = false;
  for (dirent* ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
    const std::string name = ent->d_name;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
      found = true;
      break;
    }
  }
  closedir(dir);
  return found;
}

void check_acl(aclError ret, const char* msg) {
  if (ret != ACL_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " ret=" + std::to_string(ret));
  }
}

}  // namespace

struct AclResidentModel::BufferSet {
  aclmdlDataset* dataset = nullptr;
  std::vector<void*> dev_ptrs;
};

AclResidentModel::AclResidentModel() { init_runtime(); }

AclResidentModel::~AclResidentModel() {
  if (model_id_ != 0) {
    aclmdlUnload(model_id_);
    model_id_ = 0;
  }
  if (model_desc_ != nullptr) {
    aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  finalize_runtime();
}

void AclResidentModel::init_runtime() {
  check_acl(aclInit(nullptr), "aclInit");
  check_acl(aclrtSetDevice(0), "aclrtSetDevice");
  runtime_ready_ = true;
}

void AclResidentModel::finalize_runtime() {
  if (!runtime_ready_) {
    return;
  }
  aclrtResetDevice(0);
  aclFinalize();
  runtime_ready_ = false;
}

void AclResidentModel::load(const std::string& om_path) {
  check_acl(aclmdlLoadFromFile(om_path.c_str(), &model_id_), "aclmdlLoadFromFile");
  model_desc_ = aclmdlCreateDesc();
  check_acl(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

  const size_t n_out = aclmdlGetNumOutputs(model_desc_);
  output_sizes_.clear();
  for (size_t i = 0; i < n_out; ++i) {
    output_sizes_.push_back(aclmdlGetOutputSizeByIndex(model_desc_, i));
  }
}

AclResidentModel::BufferSet* AclResidentModel::create_input_dataset(
    const std::vector<std::vector<uint8_t>>& blobs) {
  auto* set = new BufferSet();
  set->dataset = aclmdlCreateDataset();
  for (const auto& blob : blobs) {
    void* dev_ptr = nullptr;
    check_acl(aclrtMalloc(&dev_ptr, blob.size(), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc input");
    check_acl(aclrtMemcpy(dev_ptr, blob.size(), blob.data(), blob.size(), ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy H2D");
    aclDataBuffer* buf = aclCreateDataBuffer(dev_ptr, blob.size());
    check_acl(aclmdlAddDatasetBuffer(set->dataset, buf), "aclmdlAddDatasetBuffer input");
    set->dev_ptrs.push_back(dev_ptr);
  }
  return set;
}

AclResidentModel::BufferSet* AclResidentModel::create_output_dataset() {
  auto* set = new BufferSet();
  set->dataset = aclmdlCreateDataset();
  for (size_t size : output_sizes_) {
    void* dev_ptr = nullptr;
    check_acl(aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc output");
    aclDataBuffer* buf = aclCreateDataBuffer(dev_ptr, size);
    check_acl(aclmdlAddDatasetBuffer(set->dataset, buf), "aclmdlAddDatasetBuffer output");
    set->dev_ptrs.push_back(dev_ptr);
  }
  return set;
}

void AclResidentModel::destroy_dataset(BufferSet* set) {
  if (set == nullptr) {
    return;
  }
  if (set->dataset != nullptr) {
    const size_t n = aclmdlGetDatasetNumBuffers(set->dataset);
    for (size_t i = 0; i < n; ++i) {
      aclDataBuffer* buf = aclmdlGetDatasetBuffer(set->dataset, i);
      void* dev_ptr = aclGetDataBufferAddr(buf);
      aclrtFree(dev_ptr);
      aclDestroyDataBuffer(buf);
    }
    aclmdlDestroyDataset(set->dataset);
  }
  delete set;
}

void AclResidentModel::write_outputs(const std::string& output_dir, BufferSet* out_set) {
  ensure_dir(output_dir);
  const size_t n = aclmdlGetDatasetNumBuffers(out_set->dataset);
  for (size_t i = 0; i < n; ++i) {
    aclDataBuffer* buf = aclmdlGetDatasetBuffer(out_set->dataset, i);
    void* dev_ptr = aclGetDataBufferAddr(buf);
    const size_t size = aclGetDataBufferSizeV2(buf);
    std::vector<uint8_t> host(size);
    check_acl(aclrtMemcpy(host.data(), size, dev_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    write_file(output_dir + "/" + std::to_string(i) + ".bin", host.data(), size);
  }
}

void AclResidentModel::execute(const std::string& input_dir, const std::string& output_dir,
                               int num_inputs) {
  std::vector<std::vector<uint8_t>> blobs;
  blobs.reserve(static_cast<size_t>(num_inputs));
  for (int i = 0; i < num_inputs; ++i) {
    blobs.push_back(read_file(input_dir + "/" + std::to_string(i) + ".bin"));
  }

  BufferSet* in_set = create_input_dataset(blobs);
  BufferSet* out_set = create_output_dataset();
  try {
    check_acl(aclmdlExecute(model_id_, in_set->dataset, out_set->dataset), "aclmdlExecute");
    write_outputs(output_dir, out_set);
  } catch (...) {
    destroy_dataset(in_set);
    destroy_dataset(out_set);
    throw;
  }
  destroy_dataset(in_set);
  destroy_dataset(out_set);

  if (!has_output_bins(output_dir)) {
    throw std::runtime_error("no output bins under " + output_dir);
  }
}

// exported for main.cpp dry-run if needed - actually keep dry run in main
