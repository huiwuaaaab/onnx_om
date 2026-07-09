#include "acl_model.h"
#include "job_queue.h"

#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace {

std::string g_worker;

void log_msg(const std::string& msg) {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t t = clock::to_time_t(now);
  std::tm tm_buf {};
  localtime_r(&t, &tm_buf);
  std::ostringstream oss;
  oss << std::put_time(&tm_buf, "%H:%M:%S") << "[cpp:" << g_worker << "] " << msg;
  std::cout << oss.str() << std::endl;
}

void dry_run_outputs(const std::string& worker, const std::string& output_dir,
                     const std::string& profile) {
  const int max_seq_len = (profile == "256_256") ? 256 : 512;
  const int num_image_tokens = (profile == "256_256") ? 64 : 196;
  const int h = 2048;
  const size_t image_embeds = static_cast<size_t>(num_image_tokens) * h * 2;
  const size_t block_hidden = static_cast<size_t>(max_seq_len) * h * 2;
  const size_t deepstack = static_cast<size_t>(num_image_tokens) * h * 2;
  const size_t mask = static_cast<size_t>(max_seq_len) * max_seq_len * 2;
  const size_t logits = 151936 * 2;

  auto write = [&](const std::string& name, size_t n) {
    std::ofstream out(output_dir + "/" + name, std::ios::binary);
    std::vector<char> zeros(n, '\0');
    out.write(zeros.data(), static_cast<std::streamsize>(n));
  };

  mkdir(output_dir.c_str(), 0755);
  if (worker == "vision") {
    write("merged_hidden_states.bin", image_embeds);
    write("deepstack_feat_5.bin", deepstack);
    write("deepstack_feat_11.bin", deepstack);
    write("deepstack_feat_17.bin", deepstack);
  } else if (worker == "preblock") {
    write("inputs_embeds_out.bin", block_hidden);
    write("attention_mask_out.bin", mask);
  } else if (worker == "block1" || worker == "block2" || worker == "block3") {
    write("hidden_states_out.bin", block_hidden);
  } else if (worker == "lm_head") {
    write("logits.bin", logits);
  } else {
    write("0.bin", 16);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: om_resident_daemon <worker_name> <om_path> <queue_dir>\n";
    return 2;
  }

  g_worker = argv[1];
  const std::string om_path = argv[2];
  const std::string queue_dir = argv[3];
  const char* profile = std::getenv("EXPORT_PROFILE");
  if (profile == nullptr) {
    profile = std::getenv("QWEN3_EXPORT_PROFILE");
  }
  if (profile == nullptr) {
    profile = "448_512";
  }

  std::ifstream om_file(om_path, std::ios::binary);
  if (!om_file) {
    log_msg("ERROR: OM not found: " + om_path);
    return 1;
  }

  JobQueue queue(queue_dir);
  AclResidentModel model;

  try {
    log_msg("loading OM via AscendCL ...");
    const auto t0 = std::chrono::steady_clock::now();
    model.load(om_path);
    const auto sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log_msg("model resident loaded om=" + om_path + " elapsed=" + std::to_string(sec) + "s");
  } catch (const std::exception& ex) {
    log_msg(std::string("ERROR: load failed: ") + ex.what());
    return 1;
  }

  queue.touch_ready();
  log_msg(std::string("ready  om=") + om_path + "  resident=cpp  jobs=" + queue.jobs_dir());

  while (!queue.should_exit()) {
    JobSpec job;
    if (!queue.pop_next(&job)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    log_msg("job " + job.job_id + " tag=" + job.tag);
    log_msg("  input : " + job.input_dir);
    log_msg("  output: " + job.output_dir);

    try {
      if (job.run_msame) {
        const auto t0 = std::chrono::steady_clock::now();
        log_msg("acl infer start tag=" + job.tag);
        model.execute(job.input_dir, job.output_dir, job.num_inputs);
        const auto sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        log_msg("acl infer done tag=" + job.tag + " elapsed=" + std::to_string(sec) + "s");
      } else {
        log_msg("  [dry-run]");
        dry_run_outputs(g_worker, job.output_dir, profile);
      }
      queue.mark_done(job.job_id);
    } catch (const std::exception& ex) {
      log_msg(std::string("ERROR: ") + ex.what());
      queue.mark_failed(job.job_id);
    }
  }

  log_msg("exit");
  return 0;
}
