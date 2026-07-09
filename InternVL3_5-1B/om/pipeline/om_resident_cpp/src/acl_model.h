#pragma once

#include <cstdint>
#include <string>
#include <vector>

class AclResidentModel {
 public:
  AclResidentModel();
  ~AclResidentModel();

  AclResidentModel(const AclResidentModel&) = delete;
  AclResidentModel& operator=(const AclResidentModel&) = delete;

  void load(const std::string& om_path);
  void execute(const std::string& input_dir, const std::string& output_dir, int num_inputs);

 private:
  struct BufferSet;

  void init_runtime();
  void finalize_runtime();
  BufferSet* create_input_dataset(const std::vector<std::vector<uint8_t>>& blobs);
  BufferSet* create_output_dataset();
  void destroy_dataset(BufferSet* set);
  void write_outputs(const std::string& output_dir, BufferSet* out_set);

  bool runtime_ready_ = false;
  uint32_t model_id_ = 0;
  void* model_desc_ = nullptr;
  std::vector<size_t> output_sizes_;
};
