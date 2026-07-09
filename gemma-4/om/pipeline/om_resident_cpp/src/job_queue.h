#pragma once

#include <map>
#include <string>
#include <vector>

struct JobSpec {
  std::string job_id;
  std::string tag;
  std::string input_dir;
  std::string output_dir;
  int num_inputs = 1;
  bool run_msame = true;
};

// FIFO protocol compatible with pipeline/worker.sh
class JobQueue {
 public:
  explicit JobQueue(std::string queue_dir);

  void touch_ready() const;
  bool should_exit() const;
  bool pop_next(JobSpec* job);
  void mark_done(const std::string& job_id) const;
  void mark_failed(const std::string& job_id) const;

  const std::string& queue_dir() const { return queue_dir_; }
  const std::string& jobs_dir() const { return jobs_dir_; }

 private:
  std::string queue_dir_;
  std::string jobs_dir_;
};

std::map<std::string, std::string> parse_env_file(const std::string& path);
