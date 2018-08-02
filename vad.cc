// Copyright (c) 2017 Personal (Binbin Zhang)
// Created on 2017-08-17
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


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#include "vad.h"

Vad::Vad(const VadConfig &config):
    config_(config),
    feature_pipeline_(config.feature_config),
    silence_frame_count_(0),
    speech_frame_count_(0),
    frame_count_(0),
    state_(kSilence),
    net_(config.net_file),
    endpoint_detected_(false), t_(0) {
}

void Vad::Reset() {
  silence_frame_count_ = 0;
  speech_frame_count_ = 0;
  frame_count_ = 0;
  state_ = kSilence;
  endpoint_detected_ = false;
  results_.clear();
  feature_pipeline_.Reset();
  t_ = 0;
}

bool Vad::DoVad(const std::vector<float> &wave, bool end_of_stream) {
  feature_pipeline_.AcceptRawWav(wave);
  if (end_of_stream) feature_pipeline_.SetDone();
  std::vector<float> feat;
  int num_frames_ready = feature_pipeline_.ReadFeature(t_, &feat);
  int feat_dim = feature_pipeline_.FeatureDim();
  Matrix<float> in(feat.data(), num_frames_ready, feat_dim), out;
  net_.Forward(in, &out);
  assert(out.NumCols() == 2);
  endpoint_detected_ = false;
  bool contains_speech = false;
  for (int i = 0; i < num_frames_ready; i++) {
    bool is_speech = true;
    if (out(i, 0) > config_.silence_thresh) is_speech = false;
    // printf("%d %f\n", i, out(i, 0));
    bool smooth_result = Smooth(is_speech);
    results_.push_back(smooth_result);
    if (smooth_result) contains_speech = true;
  }
  t_ += num_frames_ready;
  return contains_speech;
}

// return 1 if current frame is speech
bool Vad::Smooth(bool is_voice) {
  switch (state_) {
    case kSilence:
      if (is_voice) {
        speech_frame_count_++;
        if (speech_frame_count_ >= config_.silence_to_speech_thresh) {
          state_ = kSpeech;
          silence_frame_count_ = 0;
        }
      } else {
        speech_frame_count_ = 0;
        silence_frame_count_++;
        if (silence_frame_count_ >= config_.endpoint_trigger_thresh) {
          endpoint_detected_ = true;
        }
      }
      break;
    case kSpeech:
      if (!is_voice) {
        silence_frame_count_++;
        if (silence_frame_count_ >= config_.speech_to_sil_thresh) {
          state_ = kSilence;
          speech_frame_count_ = 0;
        }
      } else {
        silence_frame_count_ = 0;
        speech_frame_count_++;
      }
      break;
    default:
      assert(0);
  }
  if (state_ == kSpeech) {
    return true;
  } else {
    return false;
  }
}

void Vad::Lookback() {
  if (config_.num_frames_lookback > 0) {
    int cur = 0;
    while (cur < results_.size()) {
      // sil, go ahead
      while (cur < results_.size() && !results_[cur]) cur++;
      // end with silence, no more voice
      if (cur == results_.size()) break;
      // voice start, lookback, assign it to voice(true)
      int start = std::max(0, cur - config_.num_frames_lookback);
      for (int i = start; i < cur; i++) results_[i] = true;
      // voice, go ahead
      while (cur < results_.size() && results_[cur]) cur++;
    }
  }
}

