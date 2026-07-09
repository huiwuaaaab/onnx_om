#include "job_queue.h"

#include <dirent.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::string trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
    ++b;
  }
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
    --e;
  }
  return s.substr(b, e - b);
}

void touch_file(const std::string& path) {
  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("touch failed: " + path);
  }
}

}  // namespace

std::map<std::string, std::string> parse_env_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("missing job env: " + path);
  }
  std::map<std::string, std::string> env;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    env[trim(line.substr(0, pos))] = trim(line.substr(pos + 1));
  }
  return env;
}

JobQueue::JobQueue(std::string queue_dir)
    : queue_dir_(std::move(queue_dir)), jobs_dir_(queue_dir_ + "/jobs") {
  mkdir(queue_dir_.c_str(), 0755);
  mkdir(jobs_dir_.c_str(), 0755);
}

void JobQueue::touch_ready() const { touch_file(queue_dir_ + "/ready"); }

bool JobQueue::should_exit() const {
  struct stat st {};
  return stat((queue_dir_ + "/exit").c_str(), &st) == 0;
}

bool JobQueue::pop_next(JobSpec* job) {
  DIR* dir = opendir(jobs_dir_.c_str());
  if (!dir) {
    return false;
  }

  std::vector<std::string> pending;
  for (dirent* ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
    const std::string name = ent->d_name;
    if (name.size() > 8 && name.substr(name.size() - 8) == ".pending") {
      pending.push_back(name.substr(0, name.size() - 8));
    }
  }
  closedir(dir);

  if (pending.empty()) {
    return false;
  }
  std::sort(pending.begin(), pending.end());
  job->job_id = pending.front();

  const std::string pending_path = jobs_dir_ + "/" + job->job_id + ".pending";
  if (unlink(pending_path.c_str()) != 0) {
    return false;
  }

  const auto env = parse_env_file(jobs_dir_ + "/" + job->job_id + ".env");
  auto get = [&](const std::string& key) -> std::string {
    const auto it = env.find(key);
    if (it == env.end()) {
      throw std::runtime_error("job env missing " + key);
    }
    return it->second;
  };

  job->tag = env.count("TAG") ? env.at("TAG") : job->job_id;
  job->input_dir = get("INPUT_DIR");
  job->output_dir = get("OUTPUT_DIR");
  job->num_inputs = env.count("NUM_INPUTS") ? std::stoi(env.at("NUM_INPUTS")) : 1;
  job->run_msame = !env.count("RUN_MSAME") || env.at("RUN_MSAME") != "0";
  return true;
}

void JobQueue::mark_done(const std::string& job_id) const {
  touch_file(jobs_dir_ + "/" + job_id + ".done");
}

void JobQueue::mark_failed(const std::string& job_id) const {
  touch_file(jobs_dir_ + "/" + job_id + ".failed");
}
