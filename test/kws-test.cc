/* Created on 2017-08-02
 * Author: Binbin Zhang
 */

#include "kws.h"
#include "wav.h"

int main(int argc, char *argv[]) {

    const char *usage = "Test kws\n";

    WavReader reader("2.wav");
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
    FeaturePipeline feature_pipeline(fbank, "../model/kws.cmvn", 10, 5);
    feature_pipeline.AcceptRawWav(wave);
    feature_pipeline.SetDone();
    int num_frames = feature_pipeline.ReadAllFeature(&feat);
    int feat_dim = feature_pipeline.FeatureDim();

    Matrix<float> in(num_frames, feat_dim), out;
    memcpy(in.Data(), feat.data(), num_frames * feat_dim * sizeof(float));

    //Net net("../model/kws.net");
    Net net("../model/kws.quantize.net");
    net.Forward(in, &out);
    
    for (int i = 0; i < out.NumRows(); i++) {
        for (int j = 0; j < out.NumCols(); j++) {
            printf("%f ", out(i, j));
        }
        printf("\n");
    }
    return 0;
}


