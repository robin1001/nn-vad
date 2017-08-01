/* Created on 2017-07-31
 * Author: Binbin Zhang
 */

#include "fbank.h"
#include "wav.h"

int main(int argc, char *argv[]) {

    const char *usage = "Test fbank feature exactor\n"
                        "Usage: fbank-test wav_in_file\n";
    if (argc != 2) {
        printf(usage);
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
