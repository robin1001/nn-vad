// Created on 2017-06-07
// Author: Binbin Zhang

#ifdef USE_BLAS
#include <cblas.h>
#endif

#include <math.h>

#include <algorithm>

#include "net.h"
#include "third_party/gemmlowp/public/gemmlowp.h"

/* Matrix & Vector Defination */

template <typename DType>
Matrix<DType>::Matrix(int32_t row, int32_t col): 
        rows_(row), cols_(col), data_(NULL), holder_(true) {
    Resize(rows_, cols_);
}

template <typename DType>
Matrix<DType>::Matrix(DType *data, int32_t row, int32_t col):
        rows_(row), cols_(col), data_(data), holder_(false) {
}

template <typename DType>
void Matrix<DType>::Resize(int32_t row, int32_t col) {
    if (row * col == 0) return;
    if (data_ == NULL || row * col != rows_ * cols_) {
        if (holder_ && data_ != NULL) delete [] data_;
        rows_ = row;
        cols_ = col;
        data_ = new DType[row * col]();
        holder_ = true;
    }
}

template <typename DType>
void Matrix<DType>::Read(std::istream &is) {
    is.read((char *)&rows_, sizeof(int32_t)); 
    is.read((char *)&cols_, sizeof(int32_t)); 
    Resize(rows_, cols_);
    is.read((char *)data_, sizeof(DType) * rows_ * cols_);
}

template <typename DType>
void Matrix<DType>::Write(std::ostream &os) {
    os.write((char *)&rows_, sizeof(int32_t)); 
    os.write((char *)&cols_, sizeof(int32_t)); 
    os.write((char *)data_, sizeof(DType) * rows_ * cols_);
}

template <typename DType>
void Matrix<DType>::Mul(const Matrix<DType> &mat1, const Matrix<DType> &mat2, 
        bool transpose, float alpha) {
    if (!transpose) {
        assert(mat1.NumCols() == mat2.NumRows());
        assert(rows_ == mat1.NumRows());
        assert(cols_ == mat2.NumCols());
        //this->Resize(mat1.NumRows(), mat2.NumCols());
        for (int i = 0; i < mat1.NumRows(); i++) {
            for (int j = 0; j < mat2.NumCols(); j++) {
                (*this)(i, j) *= alpha; 
                for (int k = 0; k < mat1.NumCols(); k++) {
                    (*this)(i, j) += mat1(i, k) * mat2(k, j); 
                }
            }
        }
    }
    else {
        assert(mat1.NumCols() == mat2.NumCols());
        assert(rows_ == mat1.NumRows());
        assert(cols_ == mat2.NumRows());
        this->Resize(mat1.NumRows(), mat2.NumRows());
        for (int i = 0; i < mat1.NumRows(); i++) {
            for (int j = 0; j < mat2.NumRows(); j++) {
                (*this)(i, j) *= alpha; 
                for (int k = 0; k < mat1.NumCols(); k++) {
                    (*this)(i, j) += mat1(i, k) * mat2(j, k); 
                }
            }
        }
    }
}

// cblas_sger
template<typename DType>
void Matrix<DType>::AddVec(const Vector<DType> &vec) {
    assert(cols_ == vec.Dim());
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            (*this)(i, j) += vec(j); 
        }
    }
}

#ifdef USE_BLAS
template <>
void Matrix<float>::Mul(const Matrix<float> &mat1, const Matrix<float> &mat2, 
        bool transpose, float alpha) {
    assert((!transpose && mat1.NumCols() == mat2.NumRows() && 
            rows_ == mat1.NumRows() && cols_ == mat2.NumCols()) ||
            (transpose && mat1.NumCols() == mat2.NumCols() && 
            rows_ == mat1.NumRows() && cols_ == mat2.NumRows()));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, !transpose ? CblasNoTrans : CblasTrans,
                rows_, cols_, mat1.NumCols(), 1.0, 
                mat1.Data(), mat1.NumCols(), mat2.Data(), mat2.NumCols(),
                alpha, data_, cols_);
}
#endif

template <typename DType>
void Matrix<DType>::Transpose(const Matrix<DType> &mat) {
    this->Resize(mat.NumCols(), mat.NumRows());
    for (int i = 0; i < mat.NumRows(); i++) {
        for (int j = 0; j < mat.NumCols(); j++) {
            (*this)(j, i) = mat(i, j);
        }
    }
}

template <typename DType>
void Matrix<DType>::CopyFrom(const Matrix<DType> &mat) {
    Resize(mat.NumRows(), mat.NumCols());
    memcpy(data_, mat.Data(), rows_ * cols_ * sizeof(DType));
}

template <typename DType>
Matrix<DType> Matrix<DType>::RowRange(int start, int length) const {
    return Matrix<DType>(data_ + start * cols_, length, cols_);
}

template <typename DType>
Vector<DType> Matrix<DType>::Row(int row) const {
    return Vector<DType>(data_ + row * cols_, cols_);
}
template <typename DType>
Vector<DType>::Vector(int32_t dim): 
        dim_(dim), data_(NULL), holder_(true) {
    Resize(dim);
}

template <typename DType>
Vector<DType>::Vector(DType *data, int32_t dim): 
        dim_(dim), data_(data), holder_(false) {
}

template <typename DType>
void Vector<DType>::Resize(int32_t dim) {
    if (dim == 0) return;
    if (data_ == NULL || dim != dim_) {
        if (holder_ && data_ != NULL) delete [] data_;
        dim_ = dim;
        data_ = new DType[dim]();
        holder_ = true;
    }
}

template <typename DType>
void Vector<DType>::Read(std::istream &is) {
    is.read((char *)&dim_, sizeof(int32_t)); 
    Resize(dim_);
    is.read((char *)data_, sizeof(DType) * dim_);
}

template <typename DType>
void Vector<DType>::Write(std::ostream &os) {
    os.write((char *)&dim_, sizeof(int32_t)); 
    os.write((char *)data_, sizeof(DType) * dim_);
}

template <typename DType>
void Vector<DType>::CopyFrom(const Vector<DType> &vec) {
    Resize(vec.Dim());
    memcpy(data_, vec.Data(), dim_ * sizeof(DType));
}

template <typename DType>
void Vector<DType>::Add(const Vector<DType> &vec, float alpha) {
    for (int i = 0; i < dim_; i++) {
        (*this)(i) += alpha * vec(i);
    }
}

template <typename DType>
void Vector<DType>::Scale(float alpha) {
    for (int i = 0; i < dim_; i++) {
        (*this)(i) *= alpha;
    }
}

/* Quantization Functions */

void FindMinMax(float *data, int n, float *min, float *max) {
    *min = *max = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > *max) *max = data[i];
        if (data[i] < *min) *min = data[i];
    }
}

void ChooseQuantizationParams(float min, float max, 
        float *scale, uint8_t *zero_point) {
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);
    // the min and max quantized values, as floating-point values
    const float qmin = 0;
    const float qmax = 255;
    // First determine the scale.
    const double scale_double = (max - min) / (qmax - qmin);
    const double initial_zero_point = qmin - min / scale_double;
    std::uint8_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
            static_cast<std::uint8_t>(round(initial_zero_point));
    }
    *zero_point = nudged_zero_point;
    *scale = scale_double;
}

void QuantizeData(float *src, int n, float *scale, 
        uint8_t *zero_point, uint8_t *dest) {
    float min, max;
    FindMinMax(src, n, &min, &max);
    ChooseQuantizationParams(min, max, scale, zero_point);
    for (int i = 0; i < n; i++) {
        float point = (*zero_point) + src[i] / (*scale);  
        float round_point = std::max(0.f, std::min(255.f, point));
        dest[i] = static_cast<uint8_t>(round(round_point));
    }
}

template <typename DType>
void DequantizeData(DType *src, int n, float scale,
        uint8_t zero_point, float *dest) {
    for (int i = 0; i < n; i++) {
        dest[i] = scale * (src[i] - zero_point);
    }
}

// @params transpose: if mat2 need transpose
template <bool transpose>
void IntegerGemm(const Matrix<uint8_t> &mat1, const Matrix<uint8_t> &mat2, 
        int offset1, int offset2, Matrix<int32_t> *out) {
    assert((!transpose && mat1.NumCols() == mat2.NumRows() && 
            out->NumRows() == mat1.NumRows() && out->NumCols() == mat2.NumCols()) ||
            (transpose && mat1.NumCols() == mat2.NumCols() && 
            out->NumRows() == mat1.NumRows() && out->NumCols() == mat2.NumRows()));
    using namespace gemmlowp;
    //left(right)-hand side
    MatrixMap<const uint8_t, MapOrder::RowMajor> 
        lhs(mat1.Data(), mat1.NumRows(), mat1.NumCols(), mat2.NumCols());
    MatrixMap<const uint8_t, !transpose ? MapOrder::RowMajor : MapOrder::ColMajor> 
        rhs(mat2.Data(), !transpose ? mat2.NumRows() : mat2.NumCols(), 
        !transpose ? mat2.NumCols() : mat2.NumRows(), 
        !transpose ? mat2.NumCols() : mat2.NumCols());
    MatrixMap<int32_t, MapOrder::RowMajor>
        result(out->Data(), out->NumRows(), out->NumCols(), out->NumCols());
    const std::tuple<> empty_pipeline = {};
    GemmContext context;
    GemmWithOutputPipeline<uint8_t, int32_t, DefaultL8R8BitDepthParams>(
        &context, lhs, rhs, &result, -offset1, -offset2, empty_pipeline);
}

std::string LayerTypeToString(LayerType type) {
    switch (type) {
        case kFullyConnect: return "<FullyConnect>";
        case kReLU: return "<ReLU>";
        case kSigmoid: return "<Sigmoid>";
        case kTanh: return "<Tanh>";
        case kSoftmax: return "<Softmax>";
        case kQuantizeFullyConnect: return "<QuantizeFullyConnect>";
        defaut: return "<Unknown>";
    }
}

void Layer::Read(std::istream &is) {
    char t = static_cast<char>(type_);
    is.read(&t, 1); 
    is.read((char *)&in_dim_, sizeof(int32_t)); 
    is.read((char *)&out_dim_, sizeof(int32_t)); 
    ReadData(is);
}

void Layer::Write(std::ostream &os) {
    char t = static_cast<char>(type_);
    os.write(&t, 1); 
    os.write((char *)&in_dim_, sizeof(int32_t)); 
    os.write((char *)&out_dim_, sizeof(int32_t)); 
    WriteData(os);
}

void Layer::Forward(const Matrix<float> &in, Matrix<float> *out) {
    assert(in.NumRows() != 0);
    assert(in.NumCols() != 0);
    assert(out != NULL);
    out->Resize(in.NumRows(), out_dim_);
    ForwardFunc(in, out);
}

void Softmax::ForwardFunc(const Matrix<float> &in, Matrix<float> *out) {
    for (int i = 0; i < in.NumRows(); i++) {
        float max = in(i, 0), sum = 0.0; 
        for (int j = 1; j < in.NumCols(); j++) {
            max = std::max(in(i, j), max);
        }
        for (int j = 0; j < in.NumCols(); j++) {
            sum += (*out)(i, j) = exp(in(i, j) - max);
        }
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) /= sum;
        }
    }
}

void ReLU::ForwardFunc(const Matrix<float> &in, Matrix<float> *out) {
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = std::max(in(i, j), 0.0f);
        }
    }
}

void Sigmoid::ForwardFunc(const Matrix<float> &in, Matrix<float> *out) {
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = 1.0 / (1 + exp(-in(i, j)));
        }
    }
}

void Tanh::ForwardFunc(const Matrix<float> &in, Matrix<float> *out) {
    for (int i = 0; i < in.NumRows(); i++) {
        for (int j = 0; j < in.NumCols(); j++) {
            (*out)(i, j) = tanh(in(i, j));
        }
    }
}

void FullyConnect::ReadData(std::istream &is) {
    w_.Read(is);
    b_.Read(is);
    assert(w_.NumRows() == b_.Dim());
}

void FullyConnect::WriteData(std::ostream &os) {
    w_.Write(os);
    b_.Write(os);
}

void FullyConnect::ForwardFunc(const Matrix<float> &in, Matrix<float> *out) {
    out->Mul(in, w_, true);
    out->AddVec(b_);
}

void QuantizeFullyConnect::QuantizeFrom(const Matrix<float> &w, 
    const Vector<float> &b) {
    w_.Resize(w.NumRows(), w.NumCols());
    int w_size = w.NumRows() * w.NumCols();
    QuantizeData(w.Data(), w_size, &w_scale_, &w_zero_point_, w_.Data());
#ifdef QUANTIZE_BIAS
    b_.Resize(b.Dim()); 
    dequantize_b_.Resize(b.Dim());
    QuantizeData(b.Data(), b.Dim(), &b_scale_, &b_zero_point_, b_.Data());
    DequantizeData(b_.Data(), b_.Dim(), b_scale_, b_zero_point_, dequantize_b_.Data());
#else
    b_.CopyFrom(b);
#endif
}

void QuantizeFullyConnect::ReadData(std::istream &is) {
    is.read((char *)&w_scale_, sizeof(float));
    is.read((char *)&w_zero_point_, sizeof(uint8_t));
    w_.Read(is);
#ifdef QUANTIZE_BIAS
    is.read((char *)&b_scale_, sizeof(float));
    is.read((char *)&b_zero_point_, sizeof(uint8_t));
    b_.Read(is);
    dequantize_b_.Resize(b_.Dim());
    DequantizeData(b_.Data(), b_.Dim(), b_scale_, b_zero_point_, dequantize_b_.Data());
#else
    b_.Read(is);
#endif
    assert(w_.NumRows() == b_.Dim());
}

void QuantizeFullyConnect::WriteData(std::ostream &os) {
    os.write((char *)&w_scale_, sizeof(float));
    os.write((char *)&w_zero_point_, sizeof(uint8_t));
    w_.Write(os);
#ifdef QUANTIZE_BIAS
    os.write((char *)&b_scale_, sizeof(float));
    os.write((char *)&b_zero_point_, sizeof(uint8_t));
#endif
    b_.Write(os);
}

void QuantizeFullyConnect::ForwardFunc(const Matrix<float> &in, 
        Matrix<float> *out) {
    // quantize in
    float in_scale;
    uint8_t in_zero_point;
    quantize_in_.Resize(in.NumRows(), in.NumCols());
    QuantizeData(in.Data(), in.NumRows() * in.NumCols(), &in_scale, 
        &in_zero_point, quantize_in_.Data());
    //// uint8 gemm
    quantize_out_.Resize(out->NumRows(), out->NumCols());
    IntegerGemm<true>(quantize_in_, w_, static_cast<int>(in_zero_point), 
        static_cast<int>(w_zero_point_), &quantize_out_);
    //// dequantize
    float out_scale = in_scale * w_scale_;
    DequantizeData(quantize_out_.Data(), out->NumRows() * out->NumCols(), 
        out_scale, 0, out->Data());
    //// add bias
#ifdef QUANTIZE_BIAS
    out->AddVec(dequantize_b_); 
#else
    out->AddVec(b_);
#endif
}

Net::~Net() {
    Clear();
}

void Net::Clear() {
    for (int i = 0; i < layers_.size(); i++) {
        delete layers_[i];
    }
    for (int i = 0; i < forward_buf_.size(); i++) {
        delete forward_buf_[i];
    }
}

void Net::Read(const std::string &filename) {
    std::ifstream is(filename, std::ifstream::binary);   
    if (is.fail()) {
       ERROR("read file %s error, check!!!", filename.c_str()); 
    } 
    while (!is.eof()) {
        int t = is.peek(); 
        if (t == EOF) break;
        LayerType type = static_cast<LayerType>(t);
        Layer *layer = NULL;
        switch (type) {
            case kFullyConnect:
                layer = new FullyConnect();
                break;
            case kReLU:
                layer = new ReLU();
                break;
            case kSigmoid:
                layer = new Sigmoid();
                break;
            case kTanh:
                layer = new Tanh();
                break;
            case kSoftmax:
                layer = new Softmax();
                break;
            case kQuantizeFullyConnect:
                layer = new QuantizeFullyConnect();
                break;
            default:
                ERROR("Unknown layer type %d", t);
        }
        assert(layer != NULL);
        layer->Read(is);
        layers_.push_back(layer);
    }
}

void Net::Write(const std::string &filename) {
    std::ofstream os(filename, std::ofstream::binary);   
    if (os.fail()) {
       ERROR("write file %s error, check!!!", filename.c_str()); 
    } 
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->Write(os);
    }
}

void Net::Forward(const Matrix<float> &in, Matrix<float> *out) {
    assert(out != NULL);
    assert(layers_.size() > 0);
    int num_layers = layers_.size();
    if (forward_buf_.size() != num_layers) {
        for (int i = 0; i < num_layers - 1; i++) {
            forward_buf_.push_back(new Matrix<float>()); 
        }
    }
    if (layers_.size() == 1) {
        layers_[0]->Forward(in, out);
    }
    else {
        layers_[0]->Forward(in, forward_buf_[0]);
        for (int i = 1; i < layers_.size() - 1; i++) {
            layers_[i]->Forward(*(forward_buf_[i-1]), forward_buf_[i]);
        }
        layers_[num_layers-1]->Forward(*(forward_buf_[num_layers-2]), out);
    }
}

void Net::Info() const {
    for (int i = 0; i < layers_.size(); i++) {
        layers_[i]->Info();
    }
}

void Net::Quantize(Net *quantize_net) const {
    quantize_net->Clear();
    for (int i = 0; i < layers_.size(); i++) {
        int32_t in_dim = layers_[i]->InDim(), out_dim = layers_[i]->OutDim();
        LayerType type = layers_[i]->Type();
        switch (type) {
            case kFullyConnect: {
                    QuantizeFullyConnect *quantize_fully_layer = 
                        new QuantizeFullyConnect(in_dim, out_dim);
                    FullyConnect *fully_layer = dynamic_cast<FullyConnect*>(layers_[i]); 
                    const Matrix<float> &w = fully_layer->W();
                    const Vector<float> &b = fully_layer->B();
                    quantize_fully_layer->QuantizeFrom(w, b);
                    quantize_net->AddLayer(quantize_fully_layer);
                }
                break;
            case kSoftmax: {
                    Softmax *softmax = new Softmax(in_dim, out_dim);
                    quantize_net->AddLayer(softmax);
                }
                break;
            case kReLU: {
                    ReLU *relu = new ReLU(in_dim, out_dim);
                    quantize_net->AddLayer(relu);
                }
                break;
            case kSigmoid: {
                    Sigmoid *sigmoid = new Sigmoid(in_dim, out_dim);
                    quantize_net->AddLayer(sigmoid);
                }
                break;
            case kTanh: {
                    Tanh *tanh = new Tanh(in_dim, out_dim);
                    quantize_net->AddLayer(tanh);
                }
                break;
            default:
                ERROR("Unable to quantize layer, type %s", 
                        LayerTypeToString(type).c_str());
        }
    }
}

template class Matrix<uint8_t>;
template class Matrix<int>;
template class Matrix<float>;
template class Vector<uint8_t>;
template class Vector<int>;
template class Vector<float>;

