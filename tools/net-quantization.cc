// Created on 2017-07-03
// Author: Binbin Zhang
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
}

