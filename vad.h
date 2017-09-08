/* Created on 2017-08-17
 * Author: Binbin Zhang
 */

#ifndef VAD_H_
#define VAD_H_

#include "feature-pipeline.h"
#include "net.h"

struct VadConfig {
    FeaturePipelineConfig feature_config;
    float silence_thresh; // (0, 1)
    std::string net_file;
    int silence_to_speech_thresh;
    int speech_to_sil_thresh;
    int endpoint_trigger_thresh; // 1.0s, 100 frames
    int num_frames_lookback; 
    VadConfig(): silence_thresh(0.5),
                 silence_to_speech_thresh(3), 
                 speech_to_sil_thresh(15), 
                 endpoint_trigger_thresh(100),
                 num_frames_lookback(0) {}
};

typedef enum {
    kSpeech,
    kSilence
} VadState;

class Vad {
public:
    Vad(VadConfig &config);
    // return true is wave contains speech frame
    bool DoVad(const std::vector<float> &wave, bool end_of_stream);
    // internal state machine smooth
    bool Smooth(bool is_voice); 
    void Reset(); 
    bool EndpointDetected() const { return endpoint_detected_; }
    const std::vector<bool>& Results() const {
        return results_;
    }
    void Lookback();
private:
    int silence_frame_count_, speech_frame_count_, frame_count_;
    VadState state_;
    bool endpoint_detected_;
    const VadConfig &config_;
    FeaturePipeline feature_pipeline_;
    Net net_;
    std::vector<bool> results_;
    int t_;
};


#endif
