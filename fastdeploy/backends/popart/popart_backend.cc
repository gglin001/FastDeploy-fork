// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/backends/popart/popart_backend.h"

#include <memory>
#include <popart/datatype.hpp>

#include "fastdeploy/backends/ort/utils.h"
#include "fastdeploy/core/float16.h"
#include "fastdeploy/utils/utils.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

#include "popart/dataflow.hpp"
#include "popart/devicemanager.hpp"
#include "popart/names.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

using IArray = popart::IArray;
using StepIO = popart::StepIO;
using TensorId = popart::TensorId;

namespace {

class FDIArray final : public popart::IArray {
public:
  explicit FDIArray(fastdeploy::FDTensor *tensor) { tensor_ = tensor; }
  explicit FDIArray(fastdeploy::FDTensor &tensor) { tensor_ = &tensor; }

public:
  void *data() { return tensor_->MutableData(); }
  popart::DataType dataType() const {
    // return PhiDType2PopartDType(tensor_->dtype());
    // TODO
    return popart::DataType::FLOAT;
  }
  std::size_t rank() const { return tensor_->shape.size(); }
  int64_t dim(size_t index) const { return tensor_->shape.at(index); }
  std::size_t nelms() const { return tensor_->Numel(); }
  const popart::Shape shape() const { return tensor_->shape; }

private:
  fastdeploy::FDTensor *tensor_;
};

} // namespace

namespace fastdeploy {

void PopartBackend::BuildOption(const PopartBackendOption &option) {
  option_ = option;
}

bool PopartBackend::InitFromPaddle(const std::string &model_file,
                                   const std::string &params_file,
                                   const PopartBackendOption &option,
                                   bool verbose) {
  if (initialized_) {
    FDERROR << "PopartBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
#ifdef ENABLE_PADDLE_FRONTEND
  char *model_content_ptr;
  int model_content_size = 0;

  if (!paddle2onnx::Export(model_file.c_str(), params_file.c_str(),
                           &model_content_ptr, &model_content_size, 11, true,
                           verbose, true, true, true, nullptr)) {
    FDERROR << "Error occured while export PaddlePaddle to ONNX format."
            << std::endl;
    return false;
  }

  std::string onnx_model_proto(model_content_ptr,
                               model_content_ptr + model_content_size);
  delete[] model_content_ptr;
  model_content_ptr = nullptr;
  return InitFromOnnx(onnx_model_proto, option, true);
#else
  FDERROR << "Didn't compile with PaddlePaddle frontend, you can try to "
             "call `InitFromOnnx` instead."
          << std::endl;
#endif
  return false;
}

bool PopartBackend::InitFromOnnx(const std::string &model_file,
                                 const PopartBackendOption &option,
                                 bool from_memory_buffer) {
  auto out = GetOutputInfo(0);
  auto o = out.name;
  auto dataFlow = popart::DataFlow(1, {{o, popart::AnchorReturnType("ALL")}});
  auto ipuModelDevice =
      popart::DeviceManager::createDeviceManager().acquireAvailableDevice(1);
  session_ = popart::InferenceSession::createFromOnnxModel(model_file, dataFlow,
                                                           ipuModelDevice);
  session_->prepareDevice();
  return true;
}

bool PopartBackend::Infer(std::vector<FDTensor> &inputs,
                          std::vector<FDTensor> *outputs) {
  // inputs
  std::map<popart::TensorId, popart::IArray &> popart_inputs;
  std::map<popart::TensorId, FDIArray> input_wrappers;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto tensor_id = inputs.at(i).name;
    input_wrappers.emplace(tensor_id, PdIArray(inputs[i]));
    popart_inputs.emplace(tensor_id, input_wrappers.at(tensor_id));
  }

  // anchors
  std::map<popart::TensorId, popart::IArray &> popart_anchors;
  std::map<popart::TensorId, FDIArray> anchor_wrappers;
  for (size_t i = 0; i < outputs->size(); i++) {
    auto tensor = outputs->at(i);
    auto tensor_id = tensor.name;
    anchor_wrappers.emplace(tensor_id, PdIArray(tensor));
    popart_anchors.emplace(tensor_id, anchor_wrappers.at(tensor_id));
  }

  // run
  popart::StepIO stepio(popart_inputs, popart_anchors);
  session_->run(stepio);

  return true;
}

TensorInfo PopartBackend::GetInputInfo(int index) {
  TensorInfo info;
  return info;
}

std::vector<TensorInfo> PopartBackend::GetInputInfos() { return {}; }

TensorInfo PopartBackend::GetOutputInfo(int index) {
  TensorInfo info;
  return info;
}

std::vector<TensorInfo> PopartBackend::GetOutputInfos() {
  std::vector<TensorInfo> infos;
  return infos;
}

} // namespace fastdeploy
