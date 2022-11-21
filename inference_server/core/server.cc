// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/server.h"

#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "src/core/model_repository_manager.h"
#include "src/core/pinned_memory_manager.h"
#include "src/core/server.h"

#ifdef TRITON_ENABLE_GPU
#include "src/core/cuda_memory_manager.h"
#endif  // TRITON_ENABLE_GPU

namespace nvidia { namespace inferenceserver {

namespace {

// Scoped increment / decrement of atomic
class ScopedAtomicIncrement {
 public:
  explicit ScopedAtomicIncrement(std::atomic<uint64_t>& counter)
      : counter_(counter)
  {
    counter_++;
  }

  ~ScopedAtomicIncrement() { counter_--; }

 private:
  std::atomic<uint64_t>& counter_;
};

}  // namespace

//
// InferenceServer
//
InferenceServer::InferenceServer()
    : version_(TRITON_VERSION), ready_state_(ServerReadyState::SERVER_INVALID)
{
  id_ = "triton";
  extensions_.push_back("classification");
  extensions_.push_back("sequence");
  extensions_.push_back("model_repository");
  extensions_.push_back("schedule_policy");
  extensions_.push_back("model_configuration");
  extensions_.push_back("system_shared_memory");
  extensions_.push_back("cuda_shared_memory");
#ifdef TRITON_ENABLE_HTTP
  extensions_.push_back("binary_tensor_data");
#endif  // TRITON_ENABLE_HTTP
#ifdef TRITON_ENABLE_STATS
  extensions_.push_back("statistics");
#endif  // TRITON_ENABLE_STATS

  strict_model_config_ = true;
  strict_readiness_ = true;
  exit_timeout_secs_ = 30;
  pinned_memory_pool_size_ = 1 << 28;
#ifdef TRITON_ENABLE_GPU
  min_supported_compute_capability_ = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  min_supported_compute_capability_ = 0.0;
#endif  // TRITON_ENABLE_GPU

  tf_soft_placement_enabled_ = true;
  tf_gpu_memory_fraction_ = 0.0;
  tf_vgpu_memory_limits_ = {};

  inflight_request_counter_ = 0;
}

Status
InferenceServer::Init()
{
  Status status;

  ready_state_ = ServerReadyState::SERVER_INITIALIZING;

  LOG_INFO << "Initializing Triton Inference Server";

  if (model_repository_paths_.empty()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return Status(
        Status::Code::INVALID_ARG, "--model-repository must be specified");
  }

  PinnedMemoryManager::Options options(pinned_memory_pool_size_);
  status = PinnedMemoryManager::Create(options);
  if (!status.IsOk()) {
    ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    return status;
  }

#ifdef TRITON_ENABLE_GPU
  // Defer the setting of default CUDA memory pool value here as
  // 'min_supported_compute_capability_' is finalized
  std::set<int> supported_gpus;
  if (GetSupportedGPUs(&supported_gpus, min_supported_compute_capability_)
          .IsOk()) {
    for (const auto gpu : supported_gpus) {
      if (cuda_memory_pool_size_.find(gpu) == cuda_memory_pool_size_.end()) {
        cuda_memory_pool_size_[gpu] = 1 << 26;
      }
    }
  }
  CudaMemoryManager::Options cuda_options(
      min_supported_compute_capability_, cuda_memory_pool_size_);
  status = CudaMemoryManager::Create(cuda_options);
  // If CUDA memory manager can't be created, just log error as the
  // server can still function properly
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }
#endif  // TRITON_ENABLE_GPU


  status = EnablePeerAccess(min_supported_compute_capability_);
  if (!status.IsOk()) {
    // failed to enable peer access is not critical, just inefficient.
    LOG_WARNING << status.Message();
  }

  // Create the model manager for the repository. Unless model control
  // is disabled, all models are eagerly loaded when the manager is created.
  bool polling_enabled = (model_control_mode_ == ModelControlMode::MODE_POLL);
  bool model_control_enabled =
      (model_control_mode_ == ModelControlMode::MODE_EXPLICIT);
  status = ModelRepositoryManager::Create(
      this, version_, model_repository_paths_, startup_models_,
      strict_model_config_, tf_gpu_memory_fraction_, tf_soft_placement_enabled_,
      tf_vgpu_memory_limits_, polling_enabled, model_control_enabled,
      min_supported_compute_capability_, &model_repository_manager_);
  if (!status.IsOk()) {
    if (model_repository_manager_ == nullptr) {
      ready_state_ = ServerReadyState::SERVER_FAILED_TO_INITIALIZE;
    } else {
      // If error is returned while the manager is set, we assume the
      // failure is due to a model not loading correctly so we just
      // continue if not exiting on error.
      ready_state_ = ServerReadyState::SERVER_READY;
    }
    return status;
  }

  ready_state_ = ServerReadyState::SERVER_READY;
  return Status::Success;
}

Status
InferenceServer::Stop()
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status::Success;
  }

  ready_state_ = ServerReadyState::SERVER_EXITING;

  if (model_repository_manager_ == nullptr) {
    LOG_INFO << "No server context available. Exiting immediately.";
    return Status::Success;
  } else {
    LOG_INFO << "Waiting for in-flight requests to complete.";
  }

  Status status = model_repository_manager_->UnloadAllModels();
  if (!status.IsOk()) {
    LOG_ERROR << status.Message();
  }

  // Wait for all in-flight non-inference requests to complete and all
  // loaded models to unload, or for the exit timeout to expire.
  uint32_t exit_timeout_iters = exit_timeout_secs_;

  while (true) {
    const auto& live_models = model_repository_manager_->LiveBackendStates();

    LOG_INFO << "Timeout " << exit_timeout_iters << ": Found "
             << live_models.size() << " live models and "
             << inflight_request_counter_
             << " in-flight non-inference requests";
    if (LOG_VERBOSE_IS_ON(1)) {
      for (const auto& m : live_models) {
        for (const auto& v : m.second) {
          LOG_VERBOSE(1) << m.first << " v" << v.first << ": "
                         << ModelReadyStateString(v.second.first);
        }
      }
    }

    if ((live_models.size() == 0) && (inflight_request_counter_ == 0)) {
      return Status::Success;
    }
    if (exit_timeout_iters <= 0) {
      break;
    }

    exit_timeout_iters--;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status(
      Status::Code::INTERNAL, "Exit timeout expired. Exiting immediately.");
}

Status
InferenceServer::PollModelRepository()
{
  LOG_VERBOSE(1) << "Polling model repository";

  // Look for changes and update the loaded model configurations
  // appropriately.
  if (ready_state_ == ServerReadyState::SERVER_READY) {
    ScopedAtomicIncrement inflight(inflight_request_counter_);
    RETURN_IF_ERROR(model_repository_manager_->PollAndUpdate());
  }

  return Status::Success;
}

Status
InferenceServer::IsLive(bool* live)
{
  *live = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(Status::Code::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered live if it can respond to this health
  // request and it was able to initialize.
  *live =
      ((ready_state_ != ServerReadyState::SERVER_INVALID) &&
       (ready_state_ != ServerReadyState::SERVER_INITIALIZING) &&
       (ready_state_ != ServerReadyState::SERVER_FAILED_TO_INITIALIZE));
  return Status::Success;
}

Status
InferenceServer::IsReady(bool* ready)
{
  *ready = false;

  if (ready_state_ == ServerReadyState::SERVER_EXITING) {
    return Status(Status::Code::UNAVAILABLE, "Server exiting");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  // Server is considered ready if it is in the ready state.
  // Additionally can report ready only when all models are ready.
  *ready = (ready_state_ == ServerReadyState::SERVER_READY);
  if (*ready && strict_readiness_) {
    // Strict readiness... get the model status and make sure all
    // models are ready.
    const auto model_versions = model_repository_manager_->BackendStates();

    for (const auto& mv : model_versions) {
      // If a model status is present but no version status,
      // the model is not ready as there is no proper version to be served
      if (mv.second.size() == 0) {
        *ready = false;
        goto strict_done;
      }
      for (const auto& vs : mv.second) {
        // Okay if model is not ready due to unload
        if ((vs.second.first != ModelReadyState::READY) &&
            (vs.second.second != "unloaded")) {
          *ready = false;
          goto strict_done;
        }
      }
    }
  strict_done:;
  }

  return Status::Success;
}

Status
InferenceServer::ModelIsReady(
    const std::string& model_name, const int64_t model_version, bool* ready)
{
  *ready = false;

  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  std::shared_ptr<InferenceBackend> backend;
  if (GetInferenceBackend(model_name, model_version, &backend).IsOk()) {
    ModelReadyState state;
    if (model_repository_manager_
            ->ModelState(model_name, backend->Version(), &state)
            .IsOk()) {
      *ready = (state == ModelReadyState::READY);
    }
  }

  return Status::Success;
}

Status
InferenceServer::ModelReadyVersions(
    const std::string& model_name, std::vector<int64_t>* versions)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  const ModelRepositoryManager::VersionStateMap version_states =
      model_repository_manager_->VersionStates(model_name);
  for (const auto& pr : version_states) {
    if (pr.second.first == ModelReadyState::READY) {
      versions->push_back(pr.first);
    }
  }

  return Status::Success;
}

Status
InferenceServer::ModelReadyVersions(
    std::map<std::string, std::vector<int64_t>>* ready_model_versions)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  const auto model_versions =
      model_repository_manager_->LiveBackendStates(true /* strict_readiness */);

  ready_model_versions->clear();
  std::vector<int64_t> versions;
  for (const auto& mv_pair : model_versions) {
    for (const auto& vs_pair : mv_pair.second) {
      versions.emplace_back(vs_pair.first);
    }
    ready_model_versions->emplace(mv_pair.first, std::move(versions));
  }

  return Status::Success;
}

Status
InferenceServer::RepositoryIndex(
    const bool ready_only,
    std::vector<ModelRepositoryManager::ModelIndex>* index)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  return model_repository_manager_->RepositoryIndex(ready_only, index);
}

Status
InferenceServer::InferAsync(std::unique_ptr<InferenceRequest>& request)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

#ifdef TRITON_ENABLE_STATS
  INFER_TRACE_ACTIVITY(
      request->Trace(), TRITONSERVER_TRACE_REQUEST_START,
      request->CaptureRequestStartNs());
#endif  // TRITON_ENABLE_STATS

  return InferenceRequest::Run(request);
}

Status
InferenceServer::LoadModel(const std::string& model_name)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ModelRepositoryManager::ActionType::LOAD;
  return model_repository_manager_->LoadUnloadModel(model_name, action_type);
}

Status
InferenceServer::UnloadModel(const std::string& model_name)
{
  if (ready_state_ != ServerReadyState::SERVER_READY) {
    return Status(Status::Code::UNAVAILABLE, "Server not ready");
  }

  ScopedAtomicIncrement inflight(inflight_request_counter_);

  auto action_type = ModelRepositoryManager::ActionType::UNLOAD;
  return model_repository_manager_->LoadUnloadModel(model_name, action_type);
}

}}  // namespace nvidia::inferenceserver
