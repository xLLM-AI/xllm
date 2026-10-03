/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/npu/npu_cpu_topology.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "core/platform/numa_utils.h"

extern char** environ;

namespace xllm::npu {
namespace {

std::optional<int32_t> parse_id(std::string_view text) {
  int32_t value = -1;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size() ||
      value < 0) {
    return std::nullopt;
  }
  return value;
}

// No shell and a bounded timeout: optional placement discovery must not hang
// server startup when the driver management interface is unavailable.
std::optional<std::string> run_command(std::vector<std::string> arguments) {
  int pipe_fds[2];
  if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
    return std::nullopt;
  }
  arguments.insert(arguments.begin(), {"env", "LC_ALL=C"});
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (auto& argument : arguments) {
    argv.emplace_back(argument.data());
  }
  argv.emplace_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  posix_spawn_file_actions_addopen(
      &actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  pid_t child = -1;
  const int32_t status =
      posix_spawnp(&child, "env", &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pipe_fds[1]);
  if (status != 0) {
    close(pipe_fds[0]);
    return std::nullopt;
  }
  if (fcntl(pipe_fds[0], F_SETFL, O_NONBLOCK) < 0) {
    kill(child, SIGKILL);
    while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
    }
    close(pipe_fds[0]);
    return std::nullopt;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::string output;
  std::array<char, 4096> buffer;
  bool finished = false;
  int child_status = 0;
  while (std::chrono::steady_clock::now() < deadline &&
         output.size() < 1024 * 1024) {
    pollfd descriptor{pipe_fds[0], POLLIN, 0};
    poll(&descriptor, 1, 50);
    ssize_t count = 0;
    while ((count = read(pipe_fds[0], buffer.data(), buffer.size())) > 0) {
      output.append(buffer.data(), static_cast<size_t>(count));
    }
    if (waitpid(child, &child_status, WNOHANG) == child) {
      // The child may have written between the last read and waitpid.
      while ((count = read(pipe_fds[0], buffer.data(), buffer.size())) > 0) {
        output.append(buffer.data(), static_cast<size_t>(count));
      }
      finished = true;
      break;
    }
  }
  if (!finished) {
    kill(child, SIGKILL);
    while (waitpid(child, &child_status, 0) < 0 && errno == EINTR) {
    }
  }
  close(pipe_fds[0]);
  if (!finished || !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    return std::nullopt;
  }
  return output;
}

std::string cpu_list(const std::vector<int32_t>& cpus) {
  std::string result;
  for (int32_t cpu : cpus) {
    if (!result.empty()) {
      result += ',';
    }
    result += std::to_string(cpu);
  }
  return result;
}

}  // namespace

void bind_npu_irqs(const std::vector<int32_t>& cpus,
                   const NpuIdentity& device) {
  if (cpus.empty()) {
    return;
  }
  const auto board = run_command({"npu-smi",
                                  "info",
                                  "-t",
                                  "board",
                                  "-i",
                                  std::to_string(device.card_id),
                                  "-c",
                                  std::to_string(device.chip_id)});
  if (!board) {
    LOG(WARNING) << "NPU CPU binding: IRQ board discovery unavailable";
    return;
  }
  std::istringstream lines(*board);
  std::string line;
  std::string pci;
  while (std::getline(lines, line)) {
    if (line.find("PCIe Bus Info") == std::string::npos) {
      continue;
    }
    std::istringstream fields(line);
    while (fields >> pci) {
    }
    break;
  }
  std::transform(pci.begin(), pci.end(), pci.begin(), [](unsigned char value) {
    return std::tolower(value);
  });
  if (pci.find_first_not_of("0123456789abcdef:.") != std::string::npos ||
      pci.empty()) {
    LOG(WARNING) << "NPU CPU binding: no usable PCI address for IRQ binding";
    return;
  }
  const std::filesystem::path irq_directory =
      "/sys/bus/pci/devices/" + pci + "/msi_irqs";
  std::ifstream interrupts("/proc/interrupts");
  std::array<std::string, 2> irqs;
  while (std::getline(interrupts, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::istringstream number(line.substr(0, colon));
    std::string irq;
    number >> irq;
    std::error_code error;
    if (!std::filesystem::exists(irq_directory / irq, error) || error) {
      continue;
    }
    if (line.find("sq_send_trigger_irq") != std::string::npos) {
      irqs[0] = irq;
    } else if (line.find("cq_update_irq") != std::string::npos) {
      irqs[1] = irq;
    }
  }
  if (irqs[0].empty() || irqs[1].empty()) {
    LOG(WARNING) << "NPU CPU binding: SQ/CQ IRQs could not be resolved for PCI "
                 << pci;
    return;
  }
  std::array<std::string, 2> previous;
  for (size_t index = 0; index < irqs.size(); ++index) {
    const std::string path = "/proc/irq/" + irqs[index] + "/smp_affinity_list";
    std::ifstream input(path);
    if (access(path.c_str(), W_OK) != 0 ||
        !std::getline(input, previous[index])) {
      LOG(WARNING) << "NPU CPU binding: IRQ affinity is not writable";
      return;
    }
  }
  for (size_t index = 0; index < irqs.size(); ++index) {
    std::ofstream output("/proc/irq/" + irqs[index] + "/smp_affinity_list");
    output << cpus[index] << std::flush;
    if (!output) {
      for (size_t restore = 0; restore < index; ++restore) {
        std::ofstream rollback("/proc/irq/" + irqs[restore] +
                               "/smp_affinity_list");
        rollback << previous[restore];
      }
      LOG(WARNING) << "NPU CPU binding: IRQ affinity write failed";
      return;
    }
  }
  LOG(INFO) << "NPU CPU binding: IRQ CPUs=" << cpu_list(cpus)
            << " SQ=" << irqs[0] << " CQ=" << irqs[1];
}

std::vector<NpuIdentity> parse_npu_inventory(const std::string& text) {
  std::vector<NpuIdentity> devices;
  devices.reserve(16);
  // Header positions matter: never treat an MCU or a physical ID as a logical
  // ID.
  const std::regex separator(R"(\s{2,})");
  std::istringstream stream(text);
  std::string line;
  if (!std::getline(stream, line)) {
    return {};
  }
  const size_t first = line.find_first_not_of(" \t\r");
  if (first == std::string::npos) {
    return {};
  }
  line = line.substr(first);
  const std::vector<std::string> headers{
      std::sregex_token_iterator(line.begin(), line.end(), separator, -1),
      std::sregex_token_iterator()};
  const auto column = [&headers](const std::string& name) -> size_t {
    const auto it = std::find(headers.begin(), headers.end(), name);
    return static_cast<size_t>(std::distance(headers.begin(), it));
  };
  const size_t card = column("NPU ID");
  const size_t chip = column("Chip ID");
  const size_t logic = column("Chip Logic ID");
  const size_t physical = column("Chip Phy-ID");
  if (card == headers.size() || chip == headers.size() ||
      logic == headers.size()) {
    return {};
  }
  std::unordered_set<int32_t> seen;
  while (std::getline(stream, line)) {
    std::istringstream fields(line);
    const std::vector<std::string> values{
        std::istream_iterator<std::string>(fields),
        std::istream_iterator<std::string>()};
    if (values.size() <= std::max({card, chip, logic})) {
      continue;
    }
    const auto card_id = parse_id(values[card]);
    const auto chip_id = parse_id(values[chip]);
    const auto logical_id = parse_id(values[logic]);
    if (!logical_id) {
      continue;
    }
    if (!card_id || !chip_id || !seen.insert(*logical_id).second) {
      return {};
    }
    const auto physical_id =
        physical < values.size() ? parse_id(values[physical]) : logical_id;
    if (!physical_id) {
      return {};
    }
    devices.emplace_back(
        NpuIdentity{*card_id, *chip_id, *logical_id, *physical_id});
  }
  std::sort(
      devices.begin(), devices.end(), [](const auto& left, const auto& right) {
        return left.logical_id < right.logical_id;
      });
  return devices;
}

std::unordered_map<int32_t, std::vector<int32_t>> parse_npu_affinity(
    const std::string& text,
    const std::vector<NpuIdentity>& devices) {
  std::unordered_map<int32_t, std::vector<int32_t>> result;
  // A2 reports NPU<n>; newer drivers can label rows with physical IDs.
  const std::regex row(
      R"(^\s*(NPU|Phy-ID)([0-9]+)\s+.*?\s+([0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)\s*$)");
  if (text.find("Affinity") == std::string::npos) {
    return result;
  }
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    std::smatch match;
    if (!std::regex_match(line, match, row)) {
      continue;
    }
    const auto id = parse_id(match[2].str());
    const auto cpus = parse_cpu_list(match[3].str());
    if (!id || !cpus) {
      continue;
    }
    const bool physical = match[1].str() == "Phy-ID";
    const auto device =
        std::find_if(devices.begin(), devices.end(), [&](const auto& item) {
          return (physical ? item.physical_id : item.logical_id) == *id;
        });
    if (device != devices.end()) {
      result.emplace(device->logical_id, *cpus);
    }
  }
  return result;
}

std::optional<int32_t> resolve_npu_logical_id(
    int32_t device_index,
    const std::string& visible_devices,
    const std::vector<NpuIdentity>& devices) {
  if (device_index < 0) {
    return std::nullopt;
  }
  int32_t logical_id = device_index;
  if (!visible_devices.empty()) {
    std::istringstream stream(visible_devices);
    std::string field;
    std::vector<int32_t> ids;
    ids.reserve(devices.size());
    std::unordered_set<int32_t> seen;
    while (std::getline(stream, field, ',')) {
      const auto id = parse_id(field);
      if (!id || !seen.insert(*id).second) {
        return std::nullopt;
      }
      ids.emplace_back(*id);
    }
    if (visible_devices.back() == ',' ||
        static_cast<size_t>(device_index) >= ids.size()) {
      return std::nullopt;
    }
    logical_id = ids[device_index];
  }
  const auto found = std::find_if(
      devices.begin(), devices.end(), [logical_id](const auto& device) {
        return device.logical_id == logical_id;
      });
  return found == devices.end() ? std::nullopt
                                : std::optional<int32_t>(logical_id);
}

CpuBindingOptions npu_cpu_binding_options(bool global_slice) {
  CpuBindingOptions options;
  options.mode = global_slice ? CpuBindingMode::GLOBAL_SLICE
                              : CpuBindingMode::TOPO_AFFINITY;
  options.extend_numa_pool = true;
  options.topology_remainder_to_last = true;
  options.reserved_cpu_count = 2;
  options.dedicated_threads = {"acl_thread", "release_thread"};
  options.memory_mode = CpuBindingMemoryMode::MIGRATE_AFTER_WARMUP;
  return options;
}

std::optional<NpuCpuBindingInfo> get_npu_cpu_binding(
    int32_t device_index,
    const std::string& soc_name) {
#if !defined(__aarch64__)
  LOG(WARNING) << "NPU CPU binding is supported on aarch64 hosts only";
  return std::nullopt;
#endif
  // Ascend 950 uses a different cluster/UVB policy; do not silently apply the
  // A2/A3 thread-role layout to it.
  if (soc_name.find("950") != std::string::npos) {
    LOG(WARNING)
        << "NPU CPU binding: Ascend 950 cluster placement is not supported";
    return std::nullopt;
  }
  CpuBindingTopology topology;
  topology.allowed_cpus = numa::get_thread_cpus();
  if (topology.allowed_cpus.empty()) {
    LOG(WARNING) << "NPU CPU binding: cannot read startup cpuset";
    return std::nullopt;
  }
  const auto inventory = run_command({"npu-smi", "info", "-m"});
  if (!inventory) {
    LOG(WARNING) << "NPU CPU binding: NPU inventory query failed";
    return std::nullopt;
  }
  const auto devices = parse_npu_inventory(*inventory);
  topology.device_ids.reserve(devices.size());
  for (const auto& device : devices) {
    topology.device_ids.emplace_back(device.logical_id);
  }
  const char* visible = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  const auto id = resolve_npu_logical_id(
      device_index, visible == nullptr ? "" : visible, devices);
  if (!id) {
    LOG(WARNING) << "NPU CPU binding: could not resolve runtime device "
                 << device_index;
    return std::nullopt;
  }
  const bool global_slice = soc_name.find("910_93") != std::string::npos ||
                            soc_name.find("910C") != std::string::npos;
  if (!global_slice) {
    const auto affinity = run_command({"npu-smi", "info", "-t", "topo"});
    if (affinity) {
      topology.affinity = parse_npu_affinity(*affinity, devices);
    }
    if (topology.affinity.empty()) {
      LOG(INFO) << "NPU CPU binding: topology affinity unavailable; using "
                   "global_slice";
    }
  }
  topology.cpu_nodes = numa::get_cpu_numa_nodes();
  std::string error;
  auto plan = make_cpu_binding_plan(
      topology, *id, npu_cpu_binding_options(global_slice), &error);
  if (!plan) {
    LOG(WARNING) << "NPU CPU binding skipped: " << error;
    return std::nullopt;
  }
  const auto device =
      std::find_if(devices.begin(), devices.end(), [&id](const auto& item) {
        return item.logical_id == *id;
      });
  return NpuCpuBindingInfo{std::move(*plan), *device};
}

}  // namespace xllm::npu
