// Copyright (c) 2016 Personal (Binbin Zhang)
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


#include "wav.h"
#include "parse-option.h"
#include "vad.h"

int main(int argc, char *argv[]) {
  const char *usage = "Apply nn vad to wav files\n"
                      "Usage : apply-vad wav_net_file cmvn_file wav_file\n";

  ParseOptions option(usage);
  int left_context = 0;
  option.Register("left-context", &left_context, "left context of features");
  int right_context = 0;
  option.Register("right-context", &right_context, "right context of features");
  float silence_thresh = 0.5;
  option.Register("silence-thresh", &silence_thresh,
                  "threshold of silence(0 ~ 1)");
  int silence_to_speech_thresh = 3;
  option.Register("silence-to-speech-thresh", &silence_to_speech_thresh,
                  "continuous frames from silence to speech");
  int speech_to_sil_thresh = 15;
  option.Register("speech-to-sil-thresh", &speech_to_sil_thresh,
                  "continuous frames from speech to silence");
  int num_frames_lookback = 0;
  option.Register("num-frames-lookback", &num_frames_lookback,
                  "number of lookback frames");

  int min_length = 50;
  option.Register("min-length", &min_length,
                  "Minimum length of the voice segment");
  int max_length = 1000;
  option.Register("max-length", &max_length,
                  "Maximum length of the voice segment");

  option.Read(argc, argv);

  if (option.NumArgs() != 3) {
    option.PrintUsage();
    exit(1);
  }

  std::string net_file = option.GetArg(1),
              cmvn_file = option.GetArg(2),
              wav_file = option.GetArg(3);

  FeaturePipelineConfig config;
  config.num_bins = 40;
  config.frame_shift = 160;
  config.frame_length = 400;
  config.sample_rate = 16000;
  config.left_context = left_context;
  config.right_context = right_context;
  config.cmvn_file = cmvn_file;

  VadConfig vad_config;
  vad_config.feature_config = config;
  vad_config.net_file = net_file;
  vad_config.silence_thresh = silence_thresh;
  vad_config.silence_to_speech_thresh = silence_to_speech_thresh;
  vad_config.speech_to_sil_thresh = speech_to_sil_thresh;
  vad_config.num_frames_lookback = num_frames_lookback;

  Vad vad(vad_config);

  WavReader reader(wav_file.c_str());
  int sample_rate = reader.SampleRate();
  int num_channels = reader.NumChannel();
  int num_samples = reader.NumSample();
  if (num_channels != 1) {
    printf("only one channel wav file is supported");
    exit(-1);
  }
  std::vector<float> wave(reader.Data(), reader.Data() + num_samples);

  vad.DoVad(wave, true);
  vad.Lookback();
  const std::vector<bool> &results = vad.Results();
  float time = static_cast<float>(num_samples) / sample_rate;
  int cur = 0;
  printf("%s %f", wav_file.c_str(), time);
  while (cur < results.size()) {
    // silence go ahead
    while (cur < results.size() && !results[cur]) cur++;
    int start = cur;
    while (cur < results.size() && cur - start < max_length &&
           results[cur]) {
      cur++;
    }
    int end = cur;
    if (end - start < min_length) continue;
    // end of sentence, no more speech
    if (start == end) continue;
    printf(" [ %d %d ] ", start, end);
  }
  printf("\n");
  return 0;
}

