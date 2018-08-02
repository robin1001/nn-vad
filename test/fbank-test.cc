// Copyright (c) 2017 Personal (Binbin Zhang)
// Created on 2017-07-31
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


#include "fbank.h"
#include "wav.h"

int main(int argc, char *argv[]) {
  const char *usage = "Test fbank feature exactor\n"
                      "Usage: fbank-test wav_in_file\n";
  if (argc != 2) {
    printf("%s", usage);
    exit(-1);
  }

  WavReader reader(argv[1]);
  int sample_rate = reader.SampleRate();
  int num_channels = reader.NumChannel();
  int num_samples = reader.NumSample();
  if (num_channels != 1) {
    printf("only one channel wav file is supported");
    exit(-1);
  }

  std::vector<float> wave(reader.Data(), reader.Data() + num_samples);
  std::vector<float> feat;
  // frame_length 0.025 * 16000, frame_shift 0.010 * 16000
  Fbank fbank(40, sample_rate, 400, 160);
  fbank.Compute(wave, &feat);
  return 0;
}

