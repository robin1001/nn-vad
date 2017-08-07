/* Created on 2017-08-02
 * Author: Binbin Zhang
 */

#include <stdio.h>

#include <string>

#include "fbank.h"
#include "net.h"

#ifndef KWS_H_
#define KWS_H_

struct FeaturePipelineConfig {
    int num_bins;
    int sample_rate;
    int frame_length;
    int frame_shift;
    int left_context, right_context;
    std::string cmvn_file;
    FeaturePipelineConfig(): 
        num_bins(40), // 40 dim fbank
        sample_rate(16000), // 16k sample rate
        frame_length(400), // frame length 25ms,
        frame_shift(160), // frame shift 16ms
        left_context(10),
        right_context(5) {
    }
};

class FeaturePipeline {
public:
    FeaturePipeline(const FeaturePipelineConfig &config);

    void AcceptRawWav(const std::vector<float> &wav);
    int NumFramesReady() const;
    void SetDone(); 
    bool Done() const { return done_; }
    int FeatureDim () const {
        return (left_context_ + 1 + right_context_) * raw_feat_dim_;
    }
    int ReadFeature(int t, std::vector<float> *feat);
    int ReadAllFeature(std::vector<float> *feat); 
    void Reset() {
        done_ = false;
        num_frames_ = 0;
        feature_buf_.clear();
    }
private:
    void ReadCmvn(const std::string cmvn_file);
    const FeaturePipelineConfig &config_;
    // mean: first row, inv_var: second row
    int left_context_, right_context_;
    int raw_feat_dim_;
    Matrix<float> cmvn_;
    Fbank fbank_;
    std::vector<float> feature_buf_;
    int num_frames_;
    bool done_;
    // TODO
    // add delta support
};


class Kws {
};

struct DtwKwsConfig {
    FeaturePipelineConfig feature_config;
    std::string net_file;
    int window_size; // frames
    float thresh; // dtw threshold, 
    DtwKwsConfig(): window_size(150), thresh(0.8) {}
};

class DtwKws : public Kws {
public:
    DtwKws(const DtwKwsConfig &config): 
        config_(config), feature_pipeline_(config.feature_config), 
        net_(config.net_file), registered_(false) {}
    void RegisterOnce(const std::vector<float> &wav);
    int RegisterDone();

    bool Detect(const std::vector<float> &wav, bool end_of_stream = true);
    void ResetDetector();
private:
    float Cos(const Vector<float> &vec1, const Vector<float> &vec2) const;
    void AllRowDistance(const Matrix<float> &mat1, const Matrix<float> &mat2,
            Matrix<float> *distance) const; 
    void MinMaxNormalization(Matrix<float> *distance) const;
    float Dtw(const Matrix<float> &mat1, const Matrix<float> &mat2) const; 
    float DtwWithAlign(const Matrix<float> &mat1, const Matrix<float> &mat2,
        std::vector<std::pair<int, int> > *align) const; 
private:
    const DtwKwsConfig &config_;
    Net net_;
    FeaturePipeline feature_pipeline_;
    std::vector<Matrix<float> *> register_samples_;
    Matrix<float> template_;
    bool registered_;
    int t_; // time index for t_;
    std::vector<float> confidence_;
};

#endif
