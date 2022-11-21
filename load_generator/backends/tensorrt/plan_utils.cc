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

#include "src/backends/tensorrt/plan_utils.h"

namespace nvidia { namespace inferenceserver {

DataType
ConvertTrtTypeToDataType(nvinfer1::DataType trt_type)
{
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      return TYPE_FP32;
    case nvinfer1::DataType::kHALF:
      return TYPE_FP16;
    case nvinfer1::DataType::kINT8:
      return TYPE_INT8;
    case nvinfer1::DataType::kINT32:
      return TYPE_INT32;
    case nvinfer1::DataType::kBOOL:
      return TYPE_BOOL;
  }

  return TYPE_INVALID;
}

MemoryFormat
ConvertTrtFmtToFmt(nvinfer1::TensorFormat trt_fmt)
{
  switch (trt_fmt) {
    case nvinfer1::TensorFormat::kLINEAR:
      return MemoryFormat::LINEAR;
    case nvinfer1::TensorFormat::kCHW2:
      return MemoryFormat::CHW2;
    case nvinfer1::TensorFormat::kCHW4:
      return MemoryFormat::CHW4;
    case nvinfer1::TensorFormat::kHWC8:
      return MemoryFormat::HCW8;
    case nvinfer1::TensorFormat::kCHW16:
      return MemoryFormat::CHW16;
    case nvinfer1::TensorFormat::kCHW32:
      return MemoryFormat::CHW32;
  }

  return MemoryFormat::INVALID;
}

const std::string
MemoryFormat_Name(MemoryFormat fmt)
{
  switch (fmt) {
    case MemoryFormat::LINEAR:
      return "LINEAR";
    case MemoryFormat::CHW2:
      return "CHW2";
    case MemoryFormat::CHW4:
      return "CHW4";
    case MemoryFormat::HCW8:
      return "HCW8";
    case MemoryFormat::CHW16:
      return "CHW16";
    case MemoryFormat::CHW32:
      return "CHW32";
    case MemoryFormat::INVALID:
      return "INVALID";
  }

  return "INVALID";
}

std::pair<bool, nvinfer1::DataType>
ConvertDataTypeToTrtType(const DataType& dtype)
{
  nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT;
  switch (dtype) {
    case TYPE_FP32:
      trt_type = nvinfer1::DataType::kFLOAT;
      break;
    case TYPE_FP16:
      trt_type = nvinfer1::DataType::kHALF;
      break;
    case TYPE_INT8:
      trt_type = nvinfer1::DataType::kINT8;
      break;
    case TYPE_INT32:
      trt_type = nvinfer1::DataType::kINT32;
      break;
    case TYPE_BOOL:
      trt_type = nvinfer1::DataType::kBOOL;
      break;
    default:
      return std::make_pair(false, trt_type);
  }
  return std::make_pair(true, trt_type);
}

bool
CompareDims(const nvinfer1::Dims& model_dims, const DimsList& dims)
{
  if (model_dims.nbDims != dims.size()) {
    return false;
  }

  for (int i = 0; i < model_dims.nbDims; ++i) {
    if (model_dims.d[i] != dims[i]) {
      return false;
    }
  }

  return true;
}

Status
CompareDimsSupported(
    const std::string& model_name, const std::string& binding_name,
    const nvinfer1::Dims& model_dims, const DimsList& dims,
    const bool supports_batching, const bool is_dynamic,
    const bool compare_exact)
{
  // If the model configuration expects batching support in the model,
  // then the first dimension must be -1.
  if (supports_batching && is_dynamic) {
    if ((model_dims.nbDims == 0) || (model_dims.d[0] != -1)) {
      return Status(
          Status::Code::INVALID_ARG,
          "model '" + model_name + "', tensor '" + binding_name +
              "': for the model to support batching the shape should have at "
              "least 1 dimension and the first dimension must be -1; but shape "
              "expected by the model is " +
              DimsDebugString(model_dims));
    }

    DimsList full_dims;
    full_dims.Add(-1);
    for (int i = 0; i < dims.size(); ++i) {
      full_dims.Add(dims[i]);
    }

    bool succ = (model_dims.nbDims == full_dims.size());
    if (succ) {
      for (int i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_dims.d[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "model '" + model_name + "', tensor '" + binding_name +
              "': the model expects " + std::to_string(model_dims.nbDims) +
              " dimensions (shape " + DimsDebugString(model_dims) +
              ") but the model configuration specifies " +
              std::to_string(full_dims.size()) +
              " dimensions (an initial batch dimension because max_batch_size "
              "> 0 followed by the explicit tensor shape, making complete "
              "shape " +
              DimsListToString(full_dims) + ")");
    }
  } else {
    // ! supports_batching
    bool succ = (model_dims.nbDims == dims.size());
    if (succ) {
      for (int i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_dims.d[i];
        if (compare_exact || (model_dim != -1)) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return Status(
          Status::Code::INVALID_ARG,
          "model '" + model_name + "', tensor '" + binding_name +
              "': the model expects " + std::to_string(model_dims.nbDims) +
              " dimensions (shape " + DimsDebugString(model_dims) +
              ") but the model configuration specifies " +
              std::to_string(dims.size()) + " dimensions (shape " +
              DimsListToString(dims) + ")");
    }
  }

  return Status::Success;
}

Status
CompareShapeDimsSupported(
    const std::string& model_name, const std::string& binding_name,
    const nvinfer1::Dims& model_dims, const DimsList& dims,
    const bool supports_batching)
{
  const int batch_offset = supports_batching ? 1 : 0;
  bool not_supported = false;
  if (dims.size() != model_dims.nbDims) {
    not_supported = true;
  } else if (model_dims.nbDims == 1) {
    if (((dims[0] + batch_offset) != model_dims.d[0]) ||
        (dims[0] == WILDCARD_DIM)) {
      not_supported = true;
    }
  } else if (model_dims.nbDims > 1) {
    return Status(
        Status::Code::INTERNAL, "unable to load model '" + model_name +
                                    "', shape binding '" + binding_name +
                                    "' can only be 0-d or 1-D tensor, got " +
                                    DimsDebugString(model_dims));
  }


  if (not_supported) {
    return Status(
        Status::Code::INVALID_ARG,
        "unable to load model '" + model_name + "', binding '" + binding_name +
            "' shape expected by framework " + DimsDebugString(model_dims) +
            " doesn't match model configuration shape " +
            DimsListToString(dims));
  }


  return Status::Success;
}

Status
MaximumDims(
    const nvinfer1::Dims& max_profile_dims, const DimsList& dims,
    const bool support_batching, const int max_batch_size,
    std::vector<int64_t>* max_dims)
{
  const int nonbatch_start_idx = (support_batching ? 1 : 0);
  if (max_profile_dims.nbDims != (dims.size() + nonbatch_start_idx)) {
    return Status(
        Status::Code::INVALID_ARG,
        "can not maximize dimension " + DimsListToString(dims) + " to " +
            DimsDebugString(max_profile_dims) + " due to  incompatibility.");
  }

  if (support_batching) {
    if (max_batch_size > max_profile_dims.d[0]) {
      return Status(
          Status::Code::INVALID_ARG,
          "unexpected configuration maximum batch size " +
              std::to_string(max_batch_size) + " binding maximum is " +
              std::to_string(max_profile_dims.d[0]));
    }
    max_dims->emplace_back(max_batch_size);
  }

  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] == WILDCARD_DIM) {
      max_dims->emplace_back(max_profile_dims.d[i + nonbatch_start_idx]);
    } else if (dims[i] <= max_profile_dims.d[i + nonbatch_start_idx]) {
      max_dims->emplace_back(dims[i]);
    } else {
      return Status(
          Status::Code::INVALID_ARG,
          "can not maximize dimension " + DimsListToString(dims) + " to " +
              DimsDebugString(max_profile_dims) + " due to  incompatibility.");
    }
  }
  return Status::Success;
}

Status
ValidateDimension(
    const nvinfer1::Dims& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension)
{
  const int nonbatch_start_idx = (skip_first_dimension ? 1 : 0);
  if ((this_dims.nbDims + nonbatch_start_idx) != max_dims.nbDims) {
    return Status(
        Status::Code::INTERNAL,
        "model expected " +
            std::to_string(max_dims.nbDims - nonbatch_start_idx) +
            " dimensions but received " + std::to_string(this_dims.nbDims) +
            " dimensions");
  }

  for (int i = 0; i < this_dims.nbDims; i++) {
    if (this_dims.d[i] < min_dims.d[i + nonbatch_start_idx] ||
        this_dims.d[i] > max_dims.d[i + nonbatch_start_idx]) {
      return Status(
          Status::Code::INTERNAL,
          "model expected the shape of dimension " + std::to_string(i) +
              " to be between " +
              std::to_string(min_dims.d[i + nonbatch_start_idx]) + " and " +
              std::to_string(max_dims.d[i + nonbatch_start_idx]) +
              " but received " + std::to_string(this_dims.d[i]));
    }
  }
  return Status::Success;
}

Status
ValidateControlDimsDynamic(
    const nvinfer1::Dims& dims, const bool support_batching)
{
  int expected_first_shape = (support_batching ? -1 : 1);
  if (dims.d[0] != expected_first_shape) {
    return Status(
        Status::Code::INTERNAL, "expects the first dimension to be " +
                                    std::to_string(expected_first_shape) +
                                    " but the model specified " +
                                    std::to_string(dims.d[0]));
  }
  for (int i = 1; i < dims.nbDims; i++) {
    if (dims.d[i] != 1) {
      return Status(
          Status::Code::INTERNAL,
          "expects all dimensions (conditionally first) to be 1 but the model "
          "specified shape " +
              DimsDebugString(dims));
    }
  }
  return Status::Success;
}

Status
ValidateShapeValues(
    const std::vector<int32_t>& request_shape_values,
    const int32_t* min_shape_values, const int32_t* max_shape_values,
    size_t nb_shape_values, const bool support_batching)
{
  if (request_shape_values.size() != nb_shape_values) {
    return Status(
        Status::Code::INVALID_ARG,
        "mismatch between the number of shape values. Expecting " +
            std::to_string(nb_shape_values) + ". Got " +
            std::to_string(request_shape_values.size()));
  }

  for (size_t i = 0; i < nb_shape_values; i++) {
    if (request_shape_values[i] < *(min_shape_values + i) ||
        request_shape_values[i] > *(max_shape_values + i)) {
      return Status(
          Status::Code::INVALID_ARG,
          "The shape value at index " + std::to_string(i) +
              " is expected to be in range from " +
              std::to_string(*(min_shape_values + i)) + " to " +
              std::to_string(*(max_shape_values + i)) +
              ", Got: " + std::to_string(request_shape_values[i]));
    }
  }
  return Status::Success;
}

void
DimsToDimVec(const nvinfer1::Dims& model_dims, std::vector<int64_t>* dims)
{
  dims->clear();
  for (int i = 0; i < model_dims.nbDims; ++i) {
    dims->emplace_back(model_dims.d[i]);
  }
}

bool
DimVecToDims(const std::vector<int64_t>& dim_vec, nvinfer1::Dims* dims)
{
  if (dim_vec.size() > dims->MAX_DIMS) {
    return false;
  } else {
    dims->nbDims = dim_vec.size();
    for (int i = 0; i < dims->nbDims; ++i) {
      dims->d[i] = (int)dim_vec[i];
    }
  }
  return true;
}

bool
ContainsWildcard(const nvinfer1::Dims& dims)
{
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == WILDCARD_DIM) {
      return true;
    }
  }
  return false;
}

bool
ContainsWildcardAtExplicitBatchDim(const nvinfer1::Dims& dims)
{
  if (dims.d[0] == WILDCARD_DIM) {
    return true;
  }
  return false;
}


const std::string
DimsDebugString(const nvinfer1::Dims& dims)
{
  std::vector<int64_t> dims_vec;
  DimsToDimVec(dims, &dims_vec);
  return DimsListToString(dims_vec);
}

}}  // namespace nvidia::inferenceserver
