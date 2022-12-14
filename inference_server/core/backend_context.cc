// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/backend_context.h"

#include "src/core/cuda_utils.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/nvtx.h"

namespace nvidia { namespace inferenceserver {

namespace {

TRITONSERVER_MemoryType
GetUsePinnedMemoryType(TRITONSERVER_MemoryType ref_buffer_type)
{
  // The following matrix is used for both input and output.
  // src   \ dest | non-pinned    | pinned     | device
  // non-pinned   | memcpy        | memcpy     | buffer needed
  // pinned       | memcpy        | memcpy     | cudaMemcpy
  // device       | buffer needed | cudaMemcpy | cudaMemcpy
  if (ref_buffer_type == TRITONSERVER_MEMORY_CPU_PINNED) {
    return TRITONSERVER_MEMORY_CPU_PINNED;
  }

  return (ref_buffer_type == TRITONSERVER_MEMORY_CPU) ? TRITONSERVER_MEMORY_GPU
                                                      : TRITONSERVER_MEMORY_CPU;
}

}  // namespace

//
// BackendContext
//
BackendContext::BackendContext(
    const std::string& name, const int gpu_device, const int max_batch_size,
    const bool enable_pinned_input, const bool enable_pinned_output,
    std::unique_ptr<MetricModelReporter>&& metric_reporter)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      enable_pinned_input_(enable_pinned_input),
      enable_pinned_output_(enable_pinned_output),
      metric_reporter_(std::move(metric_reporter))
{
#ifdef TRITON_ENABLE_GPU
  stream_ = nullptr;
#endif  // TRITON_ENABLE_GPU
}

BackendContext::~BackendContext()
{
#ifdef TRITON_ENABLE_GPU
  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: " << cudaGetErrorString(err);
    }
    stream_ = nullptr;
  }
#endif  // TRITON_ENABLE_GPU
}

Status
BackendContext::CreateCudaStream(
    const int cuda_stream_priority, cudaStream_t* stream)
{
#ifdef TRITON_ENABLE_GPU
  if (gpu_device_ != NO_GPU_DEVICE) {
    // Make sure that correct device is set before creating stream and
    // then restore the device to what was set by the caller.
    int current_device;
    auto cuerr = cudaGetDevice(&current_device);
    bool overridden = false;
    if (cuerr == cudaSuccess) {
      overridden = (current_device != gpu_device_);
      if (overridden) {
        cuerr = cudaSetDevice(gpu_device_);
      }
    }

    if (cuerr == cudaSuccess) {
      cudaStream_t* s = (stream == nullptr) ? &stream_ : stream;
      cuerr = cudaStreamCreateWithPriority(
          s, cudaStreamDefault, cuda_stream_priority);
    }

    if (overridden) {
      cudaSetDevice(current_device);
    }

    if (cuerr != cudaSuccess) {
      return Status(
          Status::Code::INTERNAL, "unable to create stream for " + name_ +
                                      ": " + cudaGetErrorString(cuerr));
    }
  }
#endif  // TRITON_ENABLE_GPU

  return Status::Success;
}

Status
BackendContext::CompareOutputDims(
    const std::string& tensor_name, const std::vector<int64_t>& model_shape,
    const DimsList& dims, const bool supports_batching)
{
  if (supports_batching) {
    DimsList full_dims;
    full_dims.Add(WILDCARD_DIM);
    for (int i = 0; i < dims.size(); ++i) {
      full_dims.Add(dims[i]);
    }

    bool succ = (model_shape.size() == (size_t)full_dims.size());
    if (succ) {
      for (int i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (full_dims[i] != WILDCARD_DIM) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "tensor '" + tensor_name + "': the model expects " +
              std::to_string(model_shape.size()) + " dimensions (shape " +
              DimsListToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(full_dims.size()) +
              " dimensions (an initial batch dimension because max_batch_size "
              "> 0 followed by the explicit tensor shape, making complete "
              "shape " +
              DimsListToString(full_dims) + ")");
    }
  } else {
    // ! supports_batching
    bool succ = (model_shape.size() == (size_t)dims.size());
    if (succ) {
      for (int i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_shape[i];
        if (dims[i] != WILDCARD_DIM) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "tensor '" + tensor_name + "': the model expects " +
              std::to_string(model_shape.size()) + " dimensions (shape " +
              DimsListToString(model_shape) +
              ") but the model configuration specifies " +
              std::to_string(dims.size()) + " dimensions (shape " +
              DimsListToString(dims) + ")");
    }
  }

  return Status::Success;
}

//
// BackendResponder
//
void
BackendResponder::ProcessTensor(
    const std::string& name, const DataType datatype,
    std::vector<int64_t>& batchn_shape, const char* buffer,
    const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }

  size_t tensor_offset = 0;

  for (size_t idx = 0; idx < responses_->size(); idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If then pending copies are from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a
    // new contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (tensor_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
    }

    // Override shape to be correct for this response.
    if (max_batch_size_ != BackendContext::NO_BATCHING) {
      batchn_shape[0] = request->BatchSize();
    }

    const size_t tensor_byte_size = GetByteSize(datatype, batchn_shape);

    InferenceResponse::Output* response_output = nullptr;
    if ((response != nullptr) &&
        (request->ImmutableRequestedOutputs().find(name) !=
         request->ImmutableRequestedOutputs().end())) {
      response->AddOutput(
          name, datatype, batchn_shape, request->BatchSize(), &response_output);
      need_sync_ |= SetFixedSizeOutputBuffer(
          &response, response_output, tensor_byte_size, tensor_offset, buffer,
          memory_type, memory_type_id, use_pinned_memory_type);
    }

    tensor_offset += tensor_byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |= FlushPendingPinned(buffer, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

bool
BackendResponder::Finalize()
{
#ifdef TRITON_ENABLE_GPU
  if ((!deferred_pinned_.empty()) && need_sync_) {
    if (event_ != nullptr) {
      cudaEventSynchronize(event_);
    } else {
      cudaStreamSynchronize(stream_);
    }
    need_sync_ = false;
  }
#endif  // TRITON_ENABLE_GPU

  // After the above sync all the GPU->pinned copies are complete. Any
  // deferred copies of pinned->CPU can now be done.
  for (auto& def : deferred_pinned_) {
    auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    int64_t pinned_memory_id = 0;
    char* pinned_buffer = def.pinned_memory_->MutableBuffer(
        &pinned_memory_type, &pinned_memory_id);

    size_t offset = 0;
    for (auto& pr : def.responses_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      InferenceResponse::Output* response_output = pr.second;

      const void* response_buffer;
      size_t response_byte_size;
      TRITONSERVER_MemoryType response_memory_type;
      int64_t response_memory_type_id;
      void* userp;

      Status status = response_output->DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(std::move(*response), status),
            "error sending TensorFlow response");
      } else {
        bool cuda_used = false;
        status = CopyBuffer(
            response_output->Name(), pinned_memory_type, pinned_memory_id,
            response_memory_type, response_memory_type_id, response_byte_size,
            pinned_buffer + offset, const_cast<void*>(response_buffer), stream_,
            &cuda_used);
        need_sync_ |= cuda_used;

        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(std::move(*response), status),
              "error sending TensorFlow response");
        }
      }

      offset += response_byte_size;
    }
  }

#ifdef TRITON_ENABLE_GPU
  // Record the new event location if deferred copies occur
  if ((!deferred_pinned_.empty()) && need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
  deferred_pinned_.clear();

  return need_sync_;
}

bool
BackendResponder::SetFixedSizeOutputBuffer(
    std::unique_ptr<InferenceResponse>* response,
    InferenceResponse::Output* response_output, const size_t tensor_byte_size,
    const size_t tensor_offset, const char* tensor_buffer,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type)
{
  void* buffer = nullptr;
  bool cuda_copy = false;

  TRITONSERVER_MemoryType actual_memory_type = tensor_memory_type;
  int64_t actual_memory_type_id = tensor_memory_type_id;

  Status status = response_output->AllocateDataBuffer(
      &buffer, tensor_byte_size, &actual_memory_type, &actual_memory_type_id);
  if (!status.IsOk()) {
    LOG_STATUS_ERROR(
        InferenceResponse::SendWithStatus(std::move(*response), status),
        "error sending TensorFlow response");
    return cuda_copy;
  }

  // If the response buffer matches the memory type that should use an
  // intermediate pinned memory buffer for the transfer, then just
  // record the response as pending and increase the size required for
  // the intermediate pinned buffer.
  if ((use_pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) &&
      (actual_memory_type == use_pinned_memory_type)) {
    if (pending_pinned_byte_size_ == 0) {
      pending_pinned_offset_ = tensor_offset;
    }

    pending_pinned_byte_size_ += tensor_byte_size;
    pending_pinned_outputs_.push_back(
        std::make_pair(response, response_output));
  } else {
    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    status = CopyBuffer(
        response_output->Name(), tensor_memory_type, tensor_memory_type_id,
        actual_memory_type, actual_memory_type_id, tensor_byte_size,
        tensor_buffer + tensor_offset, buffer, stream_, &cuda_used);
    cuda_copy |= cuda_used;

    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          InferenceResponse::SendWithStatus(std::move(*response), status),
          "error sending TensorFlow response");
      return cuda_copy;
    }
  }

  return cuda_copy;
}

bool
BackendResponder::FlushPendingPinned(
    const char* tensor_buffer, const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  bool cuda_copy = false;

  // Will be copying from CPU->pinned->GPU or GPU->pinned->CPU

  // Always need a pinned buffer...
  auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t pinned_memory_id = 0;

  std::unique_ptr<AllocatedMemory> pinned_memory(new AllocatedMemory(
      pending_pinned_byte_size_, pinned_memory_type, pinned_memory_id));
  char* pinned_buffer =
      pinned_memory->MutableBuffer(&pinned_memory_type, &pinned_memory_id);

  // If the pinned buffer isn't actually pinned memory then just
  // perform a direct copy. In this case 'pinned_memory' is just
  // deallocated and not used.
  if (pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    pinned_memory.reset();

    size_t offset = 0;
    for (auto& pr : pending_pinned_outputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      InferenceResponse::Output* response_output = pr.second;

      const void* response_buffer;
      size_t response_byte_size;
      TRITONSERVER_MemoryType response_memory_type;
      int64_t response_memory_type_id;
      void* userp;

      Status status = response_output->DataBuffer(
          &response_buffer, &response_byte_size, &response_memory_type,
          &response_memory_type_id, &userp);
      if (!status.IsOk()) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(std::move(*response), status),
            "error sending TensorFlow response");
      } else {
        bool cuda_used = false;
        status = CopyBuffer(
            response_output->Name(), tensor_memory_type, tensor_memory_type_id,
            response_memory_type, response_memory_type_id, response_byte_size,
            tensor_buffer + pending_pinned_offset_ + offset,
            const_cast<void*>(response_buffer), stream_, &cuda_used);
        cuda_copy |= cuda_used;

        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(std::move(*response), status),
              "error sending TensorFlow response");
        }
      }

      offset += response_byte_size;
    }
  }
  // We have a pinned buffer so do a single copy of a block of tensor
  // data to the pinned buffer.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    Status status = CopyBuffer(
        "pinned buffer", tensor_memory_type, tensor_memory_type_id,
        pinned_memory_type, pinned_memory_id, pending_pinned_byte_size_,
        tensor_buffer + pending_pinned_offset_, pinned_buffer, stream_,
        &cuda_used);
    cuda_copy |= cuda_used;

    // If something goes wrong with the copy all the pending
    // responses fail...
    if (!status.IsOk()) {
      for (auto& pr : pending_pinned_outputs_) {
        LOG_STATUS_ERROR(
            InferenceResponse::SendWithStatus(std::move(*(pr.first)), status),
            "error sending TensorFlow response");
      }
    }

    // If the copy was not async (i.e. if tensor was in CPU so a
    // CPU->CPU-PINNED copy was performed above), then the pinned
    // buffer now holds the tensor contents and we can immediately
    // issue the copies from the pinned buffer to the
    // responses.
    //
    // Otherwise the GPU->CPU-PINNED async copies are in flight and we
    // simply remember the pinned buffer and the corresponding
    // response outputs so that we can do the pinned->CPU copies in
    // finalize after we have waited for all async copies to complete.
    if (!cuda_used) {
      size_t offset = 0;
      for (auto& pr : pending_pinned_outputs_) {
        std::unique_ptr<InferenceResponse>* response = pr.first;
        InferenceResponse::Output* response_output = pr.second;

        const void* response_buffer;
        size_t response_byte_size;
        TRITONSERVER_MemoryType response_memory_type;
        int64_t response_memory_type_id;
        void* userp;

        Status status = response_output->DataBuffer(
            &response_buffer, &response_byte_size, &response_memory_type,
            &response_memory_type_id, &userp);
        if (!status.IsOk()) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(std::move(*response), status),
              "error sending TensorFlow response");
        } else {
          bool cuda_used = false;
          status = CopyBuffer(
              response_output->Name(), pinned_memory_type, pinned_memory_id,
              response_memory_type, response_memory_type_id, response_byte_size,
              pinned_buffer + offset, const_cast<void*>(response_buffer),
              stream_, &cuda_used);
          cuda_copy |= cuda_used;

          if (!status.IsOk()) {
            LOG_STATUS_ERROR(
                InferenceResponse::SendWithStatus(std::move(*response), status),
                "error sending TensorFlow response");
          }
        }

        offset += response_byte_size;
      }
    } else {
      deferred_pinned_.emplace_back(
          std::move(pinned_memory), std::move(pending_pinned_outputs_));
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_outputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    pinned_memories_.push_back(std::move(pinned_memory));
  }

  return cuda_copy;
}


//
// BackendInputCollector
//
void
BackendInputCollector::ProcessTensor(
    const std::string& name, const DataType datatype,
    const std::vector<int64_t>& batch1_shape, char* buffer,
    const size_t buffer_byte_size, const TRITONSERVER_MemoryType memory_type,
    const int64_t memory_type_id)
{
  // A value of CPU_PINNED indicates that pinned memory buffer is not
  // needed for this tensor. Any other value indicates that a pinned
  // memory buffer is needed when the target memory type matches
  // 'use_pinned_memory_type'.
  TRITONSERVER_MemoryType use_pinned_memory_type =
      TRITONSERVER_MEMORY_CPU_PINNED;
  if (pinned_enabled_) {
    use_pinned_memory_type = GetUsePinnedMemoryType(memory_type);
  }

  const size_t batch1_byte_size = GetByteSize(datatype, batch1_shape);
  size_t buffer_offset = 0;

  for (size_t idx = 0; idx < requests_.size(); idx++) {
    auto& request = requests_[idx];
    auto& response = (*responses_)[idx];

    // If there are pending copies from tensor buffer that is not
    // contiguous with 'response's part of that buffer, then need to
    // go ahead and perform the pending copies so that can start a new
    // contiguous region if necessary.
    if ((pending_pinned_byte_size_ > 0) &&
        (buffer_offset !=
         (pending_pinned_byte_size_ + pending_pinned_offset_))) {
      need_sync_ |= FlushPendingPinned(
          buffer, buffer_byte_size, memory_type, memory_type_id);
    }

    const size_t request_byte_size =
        batch1_byte_size * std::max(1U, request->BatchSize());

    const InferenceRequest::Input* request_input;
    Status status = request->ImmutableInput(name, &request_input);
    if (!status.IsOk() && (response != nullptr)) {
      InferenceResponse::SendWithStatus(std::move(response), status);
    } else {
      need_sync_ |= SetFixedSizeInputTensor(
          request_input, buffer_offset, buffer, buffer_byte_size, memory_type,
          memory_type_id, use_pinned_memory_type, &response);
    }

    buffer_offset += request_byte_size;
  }

  // Done with the tensor, flush any pending pinned copies.
  need_sync_ |=
      FlushPendingPinned(buffer, buffer_byte_size, memory_type, memory_type_id);
#ifdef TRITON_ENABLE_GPU
  if (need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
}

bool
BackendInputCollector::Finalize()
{
#ifdef TRITON_ENABLE_GPU
  if ((!deferred_pinned_.empty()) && need_sync_) {
    if (event_ != nullptr) {
      cudaEventSynchronize(event_);
    } else {
      cudaStreamSynchronize(stream_);
    }
    need_sync_ = false;
  }
#endif  // TRITON_ENABLE_GPU

  // After the above sync all the GPU->pinned copies are complete. Any
  // deferred copies of pinned->CPU can now be done.
  for (auto& def : deferred_pinned_) {
    auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
    int64_t pinned_memory_id = 0;
    char* pinned_buffer = def.pinned_memory_->MutableBuffer(
        &pinned_memory_type, &pinned_memory_id);

    bool cuda_used = false;
    Status status = CopyBuffer(
        "pinned buffer", pinned_memory_type, pinned_memory_id,
        def.tensor_memory_type_, def.tensor_memory_id_,
        def.pinned_memory_->TotalByteSize(), pinned_buffer,
        def.tensor_buffer_ + def.tensor_buffer_offset_, stream_, &cuda_used);
    need_sync_ |= cuda_used;

    // If something goes wrong with the copy all the pending
    // responses fail...
    if (!status.IsOk()) {
      for (auto& pr : def.requests_) {
        std::unique_ptr<InferenceResponse>* response = pr.first;
        if (*response != nullptr) {
          LOG_STATUS_ERROR(
              InferenceResponse::SendWithStatus(std::move(*response), status),
              "error setting TensorFlow input tensor");
        }
      }
    }
  }

#ifdef TRITON_ENABLE_GPU
  // Record the new event location if deferred copies occur
  if ((!deferred_pinned_.empty()) && need_sync_ && (event_ != nullptr)) {
    cudaEventRecord(event_, stream_);
  }
#endif  // TRITON_ENABLE_GPU
  deferred_pinned_.clear();

  return need_sync_;
}

bool
BackendInputCollector::SetFixedSizeInputTensor(
    const InferenceRequest::Input* request_input,
    const size_t tensor_buffer_offset, char* tensor_buffer,
    const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id,
    const TRITONSERVER_MemoryType use_pinned_memory_type,
    std::unique_ptr<InferenceResponse>* response)
{
  bool cuda_copy = false;
/*
  if ((tensor_buffer_offset + request_input->Data()->TotalByteSize()) >
      tensor_buffer_byte_size) {
    InferenceResponse::SendWithStatus(
        std::move(*response),
        Status(
            Status::Code::INVALID_ARG,
            "unexpected total byte size " +
                std::to_string(
                    tensor_buffer_offset +
                    request_input->Data()->TotalByteSize()) +
                " for input '" + request_input->Name() + "', expecting " +
                std::to_string(tensor_buffer_byte_size)));
    return cuda_copy;
  }
*/
  // Request input tensor data may be in multiple non-contiguous
  // buffers.
  size_t input_offset = 0;
  for (size_t idx = 0; idx < request_input->DataBufferCount(); ++idx) {
    const void* src_buffer;
    size_t src_byte_size;
    TRITONSERVER_MemoryType src_memory_type;
    int64_t src_memory_type_id;

    Status status = request_input->DataBuffer(
        idx, &src_buffer, &src_byte_size, &src_memory_type,
        &src_memory_type_id);
    if (!status.IsOk()) {
      InferenceResponse::SendWithStatus(std::move(*response), status);
      return cuda_copy;
    }

    // If the request buffer matches the memory type that should use an
    // intermediate pinned memory buffer for the transfer, then just
    // record the input as pending and increase the size required for
    // the intermediate pinned buffer. We only do this check for the
    // first buffer of an input and apply the same policy for all
    // buffers. So if an inputs data is split over different memory
    // types this may not be ideal but that should be a very rare
    // situation.
    if ((idx == 0) &&
        (use_pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) &&
        (src_memory_type == use_pinned_memory_type)) {
      if (pending_pinned_byte_size_ == 0) {
        pending_pinned_offset_ = tensor_buffer_offset;
      }

      pending_pinned_byte_size_ += request_input->Data()->TotalByteSize();
      pending_pinned_inputs_.push_back(std::make_pair(response, request_input));
      return cuda_copy;
    }

    // Direct copy without intermediate pinned memory.
    bool cuda_used = false;
    status = CopyBuffer(
        request_input->Name(), src_memory_type, src_memory_type_id,
        tensor_memory_type, tensor_memory_type_id, src_byte_size, src_buffer,
        tensor_buffer + tensor_buffer_offset + input_offset, stream_,
        &cuda_used);
    cuda_copy |= cuda_used;

    if (!status.IsOk()) {
      LOG_STATUS_ERROR(
          InferenceResponse::SendWithStatus(std::move(*response), status),
          "error setting TensorFlow input tensor");
      return cuda_copy;
    }

    input_offset += src_byte_size;
  }

  return cuda_copy;
}

bool
BackendInputCollector::FlushPendingPinned(
    char* tensor_buffer, const size_t tensor_buffer_byte_size,
    const TRITONSERVER_MemoryType tensor_memory_type,
    const int64_t tensor_memory_type_id)
{
  bool cuda_copy = false;

  // Will be copying from CPU->pinned->GPU or GPU->pinned->CPU

  // Always need a pinned buffer...
  auto pinned_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
  int64_t pinned_memory_id = 0;

  std::unique_ptr<AllocatedMemory> pinned_memory(new AllocatedMemory(
      pending_pinned_byte_size_, pinned_memory_type, pinned_memory_id));
  char* pinned_buffer =
      pinned_memory->MutableBuffer(&pinned_memory_type, &pinned_memory_id);

  // If the pinned buffer isn't actually pinned memory then just
  // perform a direct copy. In this case 'pinned_memory' is just
  // deallocated and not used.
  if (pinned_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    pinned_memory.reset();

    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      const InferenceRequest::Input* request_input = pr.second;

      cuda_copy |= SetFixedSizeInputTensor(
          request_input, pending_pinned_offset_ + offset, tensor_buffer,
          tensor_buffer_byte_size, tensor_memory_type, tensor_memory_type_id,
          TRITONSERVER_MEMORY_CPU_PINNED, response);
      offset += request_input->Data()->TotalByteSize();
    }
  }
  // We have a pinned buffer so copy the pending input buffer(s) into
  // the pinned memory.
  else {  // pinned_memory_type == TRITONSERVER_MEMORY_CPU_PINNED
    bool cuda_used = false;
    size_t offset = 0;
    for (auto& pr : pending_pinned_inputs_) {
      std::unique_ptr<InferenceResponse>* response = pr.first;
      const InferenceRequest::Input* request_input = pr.second;

      cuda_used |= SetFixedSizeInputTensor(
          request_input, offset, pinned_buffer, pending_pinned_byte_size_,
          pinned_memory_type, pinned_memory_id, TRITONSERVER_MEMORY_CPU_PINNED,
          response);
      offset += request_input->Data()->TotalByteSize();
    }

    cuda_copy |= cuda_used;

    // If the copy was not async (i.e. if request input was in CPU so
    // a CPU->CPU-PINNED copy was performed above), then the pinned
    // buffer now holds the tensor contents and we can immediately
    // issue the copies from the pinned buffer to the tensor.
    //
    // Otherwise the GPU->CPU-PINNED async copies are in flight and we
    // simply remember the pinned buffer and the corresponding
    // request inputs so that we can do the pinned->CPU copies in
    // finalize after we have waited for all async copies to complete.
    if (!cuda_used) {
      Status status = CopyBuffer(
          "pinned buffer", pinned_memory_type, pinned_memory_id,
          tensor_memory_type, tensor_memory_type_id, pending_pinned_byte_size_,
          pinned_buffer, tensor_buffer + pending_pinned_offset_, stream_,
          &cuda_used);
      cuda_copy |= cuda_used;

      // If something goes wrong with the copy all the pending
      // responses fail...
      if (!status.IsOk()) {
        for (auto& pr : pending_pinned_inputs_) {
          std::unique_ptr<InferenceResponse>* response = pr.first;
          if (*response != nullptr) {
            LOG_STATUS_ERROR(
                InferenceResponse::SendWithStatus(std::move(*response), status),
                "error setting TensorFlow input tensor");
          }
        }
      }
    } else {  // cuda_used
      deferred_pinned_.emplace_back(
          std::move(pinned_memory), tensor_buffer, pending_pinned_offset_,
          tensor_memory_type, tensor_memory_type_id,
          std::move(pending_pinned_inputs_));
    }
  }

  // Pending pinned copies are handled...
  pending_pinned_byte_size_ = 0;
  pending_pinned_offset_ = 0;
  pending_pinned_inputs_.clear();

  // Need to hold on to the allocated pinned buffer as there are still
  // copies in flight. Will delete it in finalize.
  if (pinned_memory != nullptr) {
    pinned_memories_.push_back(std::move(pinned_memory));
  }

  return cuda_copy;
}

}}  // namespace nvidia::inferenceserver
