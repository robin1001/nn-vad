// Copyright (c) 2017 Personal (Binbin Zhang)
// Created on 2017-08-02
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <stdio.h>
#include <string>
#include <vector>

#include "fbank.h"
#include "net.h"

#ifndef FEATURE_PIPELINE_H_
#define FEATURE_PIPELINE_H_

struct FeaturePipelineConfig {
  int num_bins;
  int sample_rate;
  int frame_length;
  int frame_shift;
  int left_context, right_context;
  std::string cmvn_file;
  FeaturePipelineConfig():
      num_bins(40),  // 40 dim fbank
      sample_rate(16000),  // 16k sample rate
      frame_length(400),  // frame length 25ms,
      frame_shift(160),  // frame shift 16ms
      left_context(5),
      right_context(5) {
  }

  void Info() const {
    LOG("feature pipeline config\n");
    LOG("num_bins %d", num_bins);
    LOG("frame_length %d", frame_length);
    LOG("frame_shift %d", frame_shift);
  }
};

class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);

  void AcceptRawWav(const std::vector<float>& wav);
  int NumFramesReady() const;
  void SetDone();
  bool Done() const { return done_; }
  int FeatureDim() const {
    return (left_context_ + 1 + right_context_) * raw_feat_dim_;
  }
  int ReadFeature(int t, std::vector<float>* feat);
  int ReadAllFeature(std::vector<float>* feat);
  int ReadOneFrame(int t, float *data);
  void Reset() {
    done_ = false;
    num_frames_ = 0;
    feature_buf_.clear();
    ctx_wav_.clear();
  }
  int NumFrames(int size) const;
  bool IsLastFrame(int frame) const {
    if (done_ && (frame == num_frames_ - 1)) {
      return true;
    } else {
      return false;
    }
  }

 private:
  void ReadCmvn(const std::string& cmvn_file);
  const FeaturePipelineConfig &config_;
  // mean: first row, inv_var: second row
  int left_context_, right_context_;
  int raw_feat_dim_;
  Matrix<float> cmvn_;
  Fbank fbank_;
  std::vector<float> feature_buf_;
  int num_frames_;
  bool done_;
  std::vector<float> ctx_wav_;
  // TODO(Binbin Zhang): Add delta support
};

#endif  // FEATURE_PIPELINE_H_


