/* Created on 2017-08-02
 * Author: Binbin Zhang
 */
#include "kws.h"

#include <algorithm>

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig &config):
        config_(config),
        left_context_(config.left_context), 
        right_context_(config.right_context),
        raw_feat_dim_(config.num_bins), 
        fbank_(config.num_bins, config.sample_rate, config.frame_length, config.frame_shift),
        num_frames_(0),
        done_(false) {
    ReadCmvn(config.cmvn_file);
}

void FeaturePipeline::ReadCmvn(const std::string cmvn_file) {
    std::ifstream is(cmvn_file, std::ifstream::binary);   
    if (is.fail()) {
       ERROR("read file %s error, check!!!", cmvn_file.c_str()); 
    }
    cmvn_.Read(is);
}

void FeaturePipeline::AcceptRawWav(const std::vector<float> &wav) {
    std::vector<float> feat;
    int num_frames = fbank_.Compute(wav, &feat);
    // do cmvn
    assert(raw_feat_dim_ == cmvn_.NumCols());
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < raw_feat_dim_; j++) {
            feat[i*raw_feat_dim_+j] = 
                (feat[i*raw_feat_dim_+j] - cmvn_(0, j)) * cmvn_(1, j);
            //printf("%f ", feat[i*raw_feat_dim+j]);
        }
        //printf("\n");
    }
    if (feature_buf_.size() == 0 && left_context_ > 0) { 
        for (int i = 0; i < left_context_; i++) {
            feature_buf_.insert(feature_buf_.end(), 
                                feat.begin(), feat.begin() + raw_feat_dim_);
        }
    }
    feature_buf_.insert(feature_buf_.end(), feat.begin(), feat.end());
    num_frames_ += num_frames;
}

int FeaturePipeline::NumFramesReady() const {
    if (num_frames_ < right_context_) return 0;
    if (done_) return num_frames_;
    else return num_frames_ - right_context_;
}

void FeaturePipeline::SetDone() { 
    assert(!done_);
    done_ = true; 
    if (num_frames_ == 0) return;
    // copy last frames to buffer
    std::vector<float> last_feat(feature_buf_.end() - raw_feat_dim_, feature_buf_.end());
    for (int i = 0; i < right_context_; i++) {
        feature_buf_.insert(feature_buf_.end(), last_feat.begin(), last_feat.end());
    }
}

int FeaturePipeline::ReadFeature(int t, std::vector<float> *feat) {
    assert(t < num_frames_);
    int num_frames_ready = NumFramesReady();
    if (num_frames_ready <= 0) return 0;
    int total_frame = num_frames_ready - t;
    int feat_dim = (left_context_ + 1 + right_context_) * raw_feat_dim_;
    feat->resize(total_frame * feat_dim);
    for (int i = t; i < num_frames_ready; i++) {
        memcpy(feat->data() + (i - t) * feat_dim,
               feature_buf_.data() + i * raw_feat_dim_,
               sizeof(float) * feat_dim);
    }
    return total_frame;
}

int FeaturePipeline::ReadAllFeature(std::vector<float> *feat) {
    return ReadFeature(0, feat);
}

void DtwKws::RegisterOnce(const std::vector<float> &wave) {
    feature_pipeline_.Reset();
    feature_pipeline_.AcceptRawWav(wave);
    feature_pipeline_.SetDone();
    std::vector<float> feat;
    int num_frames = feature_pipeline_.ReadAllFeature(&feat);
    int feat_dim = feature_pipeline_.FeatureDim();
    Matrix<float> in(feat.data(), num_frames, feat_dim), 
                  *out = new Matrix<float>;
    net_.Forward(in, out);
    register_samples_.push_back(out);    
}

int DtwKws::RegisterDone() {
    assert(register_samples_.size() > 0);
    // select reference sample
    std::vector<int> length;
    for (int i = 0; i < register_samples_.size(); i++) {
        length.push_back(register_samples_[i]->NumRows());
    }
    std::sort(length.begin(), length.end());
    int ref_length = length[length.size() / 2];
    int ref = 0;
    for (int i = 0; i < register_samples_.size(); i++) {
        if (register_samples_[i]->NumRows() == ref_length) {
            ref = i;
            break;
        }
    }

    // dtw
    std::vector<std::vector<std::pair<int, int> > > aligns(register_samples_.size());
    for (int i = 0; i < register_samples_.size(); i++) {
        if (i != ref) {
            float score = DtwWithAlign(*register_samples_[ref], 
                         *register_samples_[i], &aligns[i]);
            //float score = Dtw(*register_samples_[ref], *register_samples_[i]);
            printf("ref %d %d %f\n", ref, i, score);
        }
    }

    // sum and average
    template_.CopyFrom(*(register_samples_[ref]));
    std::vector<int> count(template_.NumRows(), 1);
    for (int i = 0; i < register_samples_.size(); i++) {
        if (i != ref) {
            for (int j = 0; j < aligns[i].size(); j++) {
                int src = aligns[i][j].first, dst = aligns[i][j].second;
                assert(src < count.size());
                assert(dst < register_samples_[i]->NumRows());
                count[src]++;
                template_.Row(src).Add(register_samples_[i]->Row(dst));
            }
        }
    }
    for (int i = 0; i < template_.NumRows(); i++) {
        template_.Row(i).Scale(1.0 / count[i]);
    }
    registered_ = true;
}

float DtwKws::Cos(const Vector<float> &vec1, const Vector<float> &vec2) const {
    assert(vec1.Dim() == vec2.Dim());
    float inner = 0.0, sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < vec1.Dim(); i++) {
        inner += vec1(i) * vec2(i);
        sum1 += vec1(i) * vec1(i);
        sum2 += vec2(i) * vec2(i);
    }
    return inner / sqrt(sum1 * sum2);
}

void DtwKws::MinMaxNormalization(Matrix<float> *distance) const {
    for (int i = 0; i < distance->NumRows(); i++) {
        float min = (*distance)(i, 0), max = (*distance)(i, 0);
        for (int j = 1; j < distance->NumCols(); j++) {
            if ((*distance)(i, j) < min) min = (*distance)(i, j);
            if ((*distance)(i, j) > max) max = (*distance)(i, j);
        }
        for (int j = 0; j < distance->NumCols(); j++) {
            (*distance)(i, j) = ((*distance)(i, j) - min) / (max - min);
        }
    }
}

void DtwKws::AllRowDistance(const Matrix<float> &mat1, const Matrix<float> &mat2,
        Matrix<float> *distance) const {
    int row1 = mat1.NumRows(), row2 = mat2.NumRows();
    distance->Resize(row1, row2);

    //for (int i = 0; i < row1; i++) {
    //    for (int j = 0; j < row2; j++) {
    //        (*distance)(i, j) = -Cos(mat1.Row(i), mat2.Row(j));
    //    }
    //}
    
    // using cos distance here, using gemm for speedup
    distance->Mul(mat1, mat2, true);
    std::vector<float> sum1(row1, 0.0), sum2(row2, 0.0);
    for (int i = 0; i < mat1.NumRows(); i++) {
        for (int j = 0; j < mat1.NumCols(); j++) {
            sum1[i] += mat1(i, j) * mat1(i, j);
        }
    }
    for (int i = 0; i < mat2.NumRows(); i++) {
        for (int j = 0; j < mat2.NumCols(); j++) {
            sum2[i] += mat2(i, j) * mat2(i, j);
        }
    }
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < row2; j++) {
            (*distance)(i, j) = -(*distance)(i, j) / sqrt(sum1[i] * sum2[j]);
        }
    }

    MinMaxNormalization(distance);
}

float DtwKws::Dtw(const Matrix<float> &mat1, const Matrix<float> &mat2) const {
    int row1 = mat1.NumRows(), row2 = mat2.NumRows();
    Matrix<float> distance;
    AllRowDistance(mat1, mat2, &distance);
    Matrix<float> dtw_distance(row1, row2);
    dtw_distance(0, 0) = distance(0, 0);
    for (int i = 1; i < row1; i++) {
        dtw_distance(i, 0) = dtw_distance(i-1, 0) + distance(i, 0);
    }
    for (int j = 1; j < row2; j++) {
        dtw_distance(0, j) = dtw_distance(0, j-1) + distance(0, j); 
    }
    for (int i = 1; i < row1; i++) {
        for (int j = 1; j < row2; j++) {
            dtw_distance(i, j) = distance(i, j) + 
                std::min(std::min(dtw_distance(i-1, j), dtw_distance(i, j-1)),
                        dtw_distance(i-1, j-1));
        }
    }
    return 1 - dtw_distance(row1-1, row2-1) / (row1 + row2);
}

float DtwKws::DtwWithAlign(const Matrix<float> &mat1, const Matrix<float> &mat2,
        std::vector<std::pair<int, int> > *align) const {
    int row1 = mat1.NumRows(), row2 = mat2.NumRows();
    Matrix<float> distance;
    AllRowDistance(mat1, mat2, &distance);
    Matrix<float> dtw_distance(row1, row2);
    // trace path -1 : start, 0 : left, 1 : up, 2 : left_up
    Matrix<int> path(row1, row2);
    dtw_distance(0, 0) = distance(0, 0);
    path(0, 0) = -1; 
    for (int i = 1; i < row1; i++) {
        dtw_distance(i, 0) = dtw_distance(i-1, 0) + distance(i, 0);
        path(i, 0) = 1;
    }
    for (int j = 1; j < row2; j++) {
        dtw_distance(0, j) = dtw_distance(0, j-1) + distance(0, j); 
        path(0, j) = 0;
    }
    for (int i = 1; i < row1; i++) {
        for (int j = 1; j < row2; j++) {
            float min = std::min(std::min(dtw_distance(i-1, j), dtw_distance(i, j-1)),
                         dtw_distance(i-1, j-1));
            dtw_distance(i, j) = distance(i, j) + min;
            if (min < dtw_distance(i, j-1) && min < dtw_distance(i-1, j-1)) path(i, j) = 1;
            else if (min < dtw_distance(i-1, j) && min < dtw_distance(i-1, j-1)) path(i, j) = 0;
            else path(i, j) = 2;
        }
    }

    align->clear();
    int v = row1 - 1, h = row2 - 1;
    while (path(v, h) >= 0) {
        assert(v >= 0);
        assert(h >= 0);
        align->push_back(std::pair<int, int>(v, h));
        switch (path(v, h)) {
            case 0:
                h--;
                break;
            case 1:
                v--;
                break;
            case 2:
                h--;
                v--;
                break;
            defaut:
                assert(0);
        }
    }
    assert(h == 0 && v == 0);
    align->push_back(std::pair<int, int>(0, 0)); // 0 align with 0
    std::reverse(align->begin(), align->end());
    //for (int i = 0; i < align->size(); i++) {
    //    printf("%d %d\n", (*align)[i].first, (*align)[i].second);
    //}
    return 1 - dtw_distance(row1-1, row2-1) / (row1 + row2);
}


bool DtwKws::Detect(const std::vector<float> &wave, bool end_of_stream) {
    assert(registered_);
    feature_pipeline_.AcceptRawWav(wave);
    if (end_of_stream) feature_pipeline_.SetDone();
    std::vector<float> feat;
    int num_frames = feature_pipeline_.ReadFeature(t_, &feat);
    int feat_dim = feature_pipeline_.FeatureDim();
    Matrix<float> in(feat.data(), num_frames, feat_dim), out;
    net_.Forward(in, &out);
    
    for (int i = 0; i < num_frames - config_.window_size; i++) {
        Matrix<float> mat = out.RowRange(i, config_.window_size);
        float score = Dtw(template_, mat);    
        printf("%d %f\n", i, score);
    }

    t_ += num_frames;
}

void DtwKws::ResetDetector() {
    t_ = 0; 
    confidence_.clear();
    feature_pipeline_.Reset();
}

