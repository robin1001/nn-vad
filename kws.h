/* Created on 2017-08-02
 * Author: Binbin Zhang
 */

#include <stdio.h>

#include <string>

#include "fbank.h"
#include "net.h"

#ifndef KWS_H_
#define KWS_H_

class FeaturePipeline {
public:
    FeaturePipeline(Fbank &fbank, std::string cmvn_file, 
        int left_context = 0, int right_context = 0);

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
    Fbank &fbank_;
    // mean: first row, inv_var: second row
    Matrix<float> cmvn_;
    int left_context_, right_context_;
    std::vector<float> feature_buf_;
    int raw_feat_dim_;
    int num_frames_;
    bool done_;
    // TODO
    // add delta support
};


class Kws {
};

class DtwKws : public Kws {
public:
   DtwKws(Net &net): net_(net) {}

private:
    Net &net_;
};

#endif

