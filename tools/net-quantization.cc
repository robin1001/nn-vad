// Copyright (c) 2017 Personal (Binbin Zhang)
// Created on 2017-07-03
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

#include <iostream>

#include "net.h"
#include "../test/parse-option.h"

int main(int argc, char *argv[]) {
  const char *usage = "Convert float net to quantize net\n";
  ParseOptions option(usage);
  option.Read(argc, argv);
  if (option.NumArgs() != 2) {
    option.PrintUsage();
    exit(1);
  }
  std::string float_net_file = option.GetArg(1),
              quantize_net_file = option.GetArg(2);

  Net net(float_net_file), quantize_net;
  net.Quantize(&quantize_net);
  quantize_net.Write(quantize_net_file);
  quantize_net.Info();

  return 0;
}

