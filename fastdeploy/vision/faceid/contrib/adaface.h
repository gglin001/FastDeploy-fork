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

#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"
#include "fastdeploy/vision/faceid/contrib/insightface_rec.h"

namespace fastdeploy {

namespace vision {

namespace faceid {

class FASTDEPLOY_DECL AdaFace : public InsightFaceRecognitionModel {
 public:
  AdaFace(const std::string& model_file, const std::string& params_file = "",
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const override {
    return "Zheng-Bicheng/AdaFacePaddleCLas";
  }

  bool Predict(cv::Mat* im, FaceRecognitionResult* result) override;

 private:
  bool Initialize() override;

  bool Preprocess(Mat* mat, FDTensor* output) override;

  bool Postprocess(std::vector<FDTensor>& infer_result,
                   FaceRecognitionResult* result) override;
};

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
