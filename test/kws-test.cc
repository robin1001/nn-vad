/* Created on 2017-08-02
 * Author: Binbin Zhang
 */

#include "kws.h"
#include "wav.h"

int main(int argc, char *argv[]) {

    const char *usage = "Test kws\n";


    FeaturePipelineConfig config;
    config.num_bins = 40;
    config.frame_shift = 160;
    config.frame_length = 400;
    config.sample_rate = 16000;
    config.left_context = 10;
    config.right_context = 5;
    config.cmvn_file = "../model/kws.cmvn";

    DtwKwsConfig dtw_config;
    dtw_config.feature_config = config;
    dtw_config.net_file = "../model/kws.quantize.net"; //"../model/kws.net"
    dtw_config.window_size = 150;
    dtw_config.thresh = 0.80;

    DtwKws kws(dtw_config);

    for (int i = 0; i < 3; i++) {
        std::string filename = "data/train/" + std::to_string(i) + ".wav";
        WavReader reader(filename.c_str());
        int sample_rate = reader.SampleRate();
        int num_channels = reader.NumChannel();
        int num_samples = reader.NumSample();
        if (num_channels != 1) {
            printf("only one channel wav file is supported");
            exit(-1);
        }       
        std::vector<float> wave(reader.Data(), reader.Data() + num_samples);

        kws.RegisterOnce(wave);
    }
    kws.RegisterDone();

    for (int i = 4; i < 6; i++) {
        std::string filename = "data/test/150_f_0" + std::to_string(i) + ".wav";
        printf("%s\n", filename.c_str());
        WavReader reader(filename.c_str());
        int sample_rate = reader.SampleRate();
        int num_channels = reader.NumChannel();
        int num_samples = reader.NumSample();
        if (num_channels != 1) {
            printf("only one channel wav file is supported");
            exit(-1);
        }       
        std::vector<float> wave(reader.Data(), reader.Data() + num_samples);
       
        kws.ResetDetector();
        kws.Detect(wave, true);
    }

    return 0;
}


