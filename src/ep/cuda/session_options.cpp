// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"

#include <filesystem>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "models/env_utils.h"
#include "models/session_options.h"

namespace Generators::CUDAExecutionProvider {

namespace {

void EnsureCudaPluginEpRegistered(bool force_Adjacent_probe = false) {
  if (!FindRegisteredEpDevices("CUDAExecutionProvider").empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::path ep_path = GetEnv("ORT_CUDA_PLUGIN_EP_PATH");
#if !defined(_WIN32)
  if (ep_path.empty() && force_Adjacent_probe) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(Ort::api->RegisterExecutionProviderLibrary), &info) != 0 &&
        info.dli_fname != nullptr) {
      auto candidate = std::filesystem::path(info.dli_fname).parent_path() / "libonnxruntime_providers_cuda.so";
      if (std::filesystem::exists(candidate, ec)) {
        ep_path = std::move(candidate);
      }
    }
  }
  if (!ep_path.empty() && std::filesystem::exists(ep_path, ec)) {
    try {
      Ort::RegisterExecutionProviderLibrary(&GetOrtEnv(), "CUDAExecutionProvider", ep_path.c_str());
    } catch (const Ort::Exception&) {
    }
  }
#else
  (void)force_Adjacent_probe;
#endif
}

void AppendProviderBridgeExecutionProvider(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    DeviceInterface*& device) {
  auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
  std::vector<const char*> keys, values;

  // Memory management settings
  const char* arena_keys[] = {
      "max_mem",
      "arena_extend_strategy",
      "initial_chunk_size_bytes",
      "max_dead_bytes_per_chunk",
      "initial_growth_chunk_size_bytes"};
  size_t arena_values[] = {
      static_cast<size_t>(0),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1),
      static_cast<size_t>(-1)};
  bool use_arena_management = false;

  for (auto& option : provider_options.options) {
    auto it = std::find(std::begin(arena_keys), std::end(arena_keys), option.first);

    if (it == std::end(arena_keys)) {
      keys.emplace_back(option.first.c_str());
      values.emplace_back(option.second.c_str());
    } else {
      const size_t idx = std::distance(std::begin(arena_keys), it);
      long long parsed_value = std::stoll(option.second);
      if (parsed_value < -1) {
        throw std::out_of_range("Arena configuration option value is out of range");
      }
      arena_values[idx] = (parsed_value == -1)
                              ? static_cast<size_t>(-1)
                              : static_cast<size_t>(parsed_value);
      use_arena_management = true;
    }
  }
  ort_provider_options->Update(keys.data(), values.data(), keys.size());

  // Device type determines the scoring device.
  // Create and set our cudaStream_t
  ort_provider_options->UpdateValue("user_compute_stream", device->GetCudaStream());

  // Use fine-grained memory management of BFC Arena.
  // The arena_cfg must outlive the AppendExecutionProvider_CUDA_V2 call below,
  // so it is declared outside the if block.
  std::unique_ptr<OrtArenaCfg> arena_cfg;
  if (use_arena_management) {
    arena_cfg = OrtArenaCfg::Create(arena_keys, arena_values, std::size(arena_keys));
    ort_provider_options->UpdateValue("default_memory_arena_cfg", arena_cfg.get());
  }

  session_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);
}

}  // namespace

void AddCudaStreamConfig(OrtSessionOptions& session_options, DeviceInterface* device,
                         const std::string& config_key) {
  if (device) {
    void* stream_ptr = device->GetCudaStream();
    std::stringstream stream_value;
    stream_value << reinterpret_cast<uintptr_t>(stream_ptr);
    session_options.AddConfigEntry(config_key.c_str(), stream_value.str().c_str());
  }
}

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& /*config*/,
                                         bool /*disable_graph_capture*/) {
  auto device = GetDeviceInterface(DeviceType::CUDA);
  AddCudaStreamConfig(session_options, device);

  // CUDA EP honors enable_cuda_graph from the config as-is. Non-decoder
  // sessions (vision, embedding, speech) should set enable_cuda_graph=0
  // in their own session_options in genai_config.json.

  // Try pre-registered plugin path first
  EnsureCudaPluginEpRegistered(false);
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CUDA, "CUDAExecutionProvider")) {
    try {
      // Register the CUDA execution provider as a provider-bridge provider.
      CUDAExecutionProvider::AppendProviderBridgeExecutionProvider(
          session_options, provider_options, device);
    } catch (const Ort::Exception&) {
      EnsureCudaPluginEpRegistered(true);
      if (!AppendExecutionProviderV2(session_options, provider_options,
                                     DeviceType::CUDA, "CUDAExecutionProvider")) {
        throw;
      }
    }
  }

  return device;
}

}  // namespace Generators::CUDAExecutionProvider
