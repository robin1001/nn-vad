// Created on 2017-06-07
// Author: Binbin Zhang
#ifndef NET_H_
#define NET_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <vector>
#include <iostream>
#include <fstream>

#include "utils.h"

/* Matrix & Vector Defination */

template <typename DType>
class Vector;

template <typename DType>
class Matrix {
public: 
    Matrix(int32_t row = 0, int32_t col = 0); 
    Matrix(DType *data, int32_t row, int32_t col); 
    virtual ~Matrix() { 
        if (holder_ && data_ != NULL) delete [] data_; 
    }
    void Resize(int32_t row, int32_t col);
    DType *Data() const { return data_; }
    int32_t NumRows() const { return rows_; }
    int32_t NumCols() const { return cols_; }
    void Read(std::istream &is);
    void Write(std::ostream &os);
    const DType operator () (int r, int c) const {
        assert(r < rows_);
        assert(c < cols_);
        return *(data_ + r * cols_ + c);
    }
    DType& operator () (int r, int c) {
        assert(r < rows_);
        assert(c < cols_);
        return *(data_ + r * cols_ + c);
    }
    // *this = alpha*this + mat1*mat2
    void Mul(const Matrix<DType> &mat1, const Matrix<DType> &mat2, 
             bool transpose = false, float alpha = 0.0);
    void Transpose(const Matrix<DType> &mat);
    void AddVec(const Vector<DType> &vec);
    void CopyFrom(const Matrix<DType> &mat); 
    Vector<DType> Row(int row) const;
    Matrix<DType> RowRange(int start, int length) const;
private:
    int32_t rows_, cols_;
    DType *data_;
    bool holder_; // if hold the memory
    //DISALLOW_COPY_AND_ASSIGN(Matrix);
};


template <typename DType>
class Vector {
public: 
    Vector(int32_t dim = 0); 
    Vector(DType *data, int32_t dim); 
    virtual ~Vector() { 
        if (holder_ && data_ != NULL) delete [] data_; 
    }
    void Resize(int32_t dim);
    DType *Data() const { return data_; }
    int32_t Dim() const { return dim_; }
    void Read(std::istream &is);
    void Write(std::ostream &os);
    const DType operator () (int n) const {
        assert(n < dim_);
        return *(data_ + n);
    }
    DType& operator () (int n) {
        assert(n < dim_);
        return *(data_ + n);
    }
    void CopyFrom(const Vector<DType> &vec);
    void Add(const Vector<DType> &vec, float alpha = 1.0);
    void Scale(float alpha);
private:
    int32_t dim_;
    DType *data_;
    bool holder_; // if hold the memory
    //DISALLOW_COPY_AND_ASSIGN(Vector);
};

/* Quantization Functions */

void FindMinMax(float *data, int n, float *min, float *max);
void ChooseQuantizationParams(float min, float max, 
        float *scale, uint8_t *zero_point);
void QuantizeData(float *src, int n, float scale, 
    uint8_t zero_point, uint8_t *dest);


/* Layer Defination */
typedef enum {
    kFullyConnect = 0x00,
    kReLU,
    kSigmoid,
    kTanh,
    kSoftmax, 
    kQuantizeFullyConnect,
    kUnknown
} LayerType;

std::string LayerTypeToString(LayerType type); 

class Layer {
public:
    Layer(int32_t in_dim = 0, int32_t out_dim = 0, LayerType type = kUnknown): 
        in_dim_(in_dim), out_dim_(out_dim), type_(type) {}
    void Read(std::istream &is);
    void Write(std::ostream &os);
    void Forward(const Matrix<float> &in, Matrix<float> *out);
    int32_t InDim() const { return in_dim_; }
    int32_t OutDim() const { return out_dim_; }
    virtual LayerType Type() const { return type_; };
    void Info() const {
        std::cout << LayerTypeToString(type_) << " in_dim " << in_dim_ 
                  << " out_dim " << out_dim_ << "\n";
    }
protected:
    virtual void ForwardFunc(const Matrix<float> &in, Matrix<float> *out) = 0;
    virtual void ReadData(std::istream &is) {};
    virtual void WriteData(std::ostream &os) {};
    int32_t in_dim_,out_dim_;
    LayerType type_;
};

class ReLU: public Layer {
public:
    ReLU(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kReLU) {}
private:
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
};

class Sigmoid: public Layer {
public:
    Sigmoid(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kSigmoid) {}
private:
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
};

class Tanh: public Layer {
public:
    Tanh(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kTanh) {}
private:
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
};

class Softmax: public Layer {
public:
    Softmax(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kSoftmax) {}
private:
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
};

class FullyConnect : public Layer {
public:
    FullyConnect(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kFullyConnect) {}
    const Matrix<float> & W() { return w_; }
    const Vector<float> & B() { return b_; }
private:
    void ReadData(std::istream &is);
    void WriteData(std::ostream &os);
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
    Matrix<float> w_; // w_ is cols major, so it's size (out_dim, in_dim)
    Vector<float> b_; // size(out_dim)
};

class QuantizeFullyConnect : public Layer {
public:
    QuantizeFullyConnect(int32_t in_dim = 0, int32_t out_dim = 0): 
        Layer(in_dim, out_dim, kQuantizeFullyConnect) {}
    void QuantizeFrom(const Matrix<float> &w, const Vector<float> &b);
private:
    void ReadData(std::istream &is);
    void WriteData(std::ostream &os);
    void ForwardFunc(const Matrix<float> &in, Matrix<float> *out);
    Matrix<uint8_t> w_; // w_ is cols major, so it's size (out_dim, in_dim)
    float w_scale_;
    uint8_t w_zero_point_;
#ifdef QUANTIZE_BIAS
    Vector<uint8_t> b_;
    Vector<float> dequantize_b_;
    float b_scale_;
    uint8_t b_zero_point_;
#else
    Vector<float> b_; // use float bias
#endif
    Matrix<int32_t> quantize_out_;
    Matrix<uint8_t> quantize_in_;
};


/* Net Defination */
class Net {
public:
    Net(std::string filename) {
        Read(filename);
    }
    Net() {}
    ~Net();
    void Clear(); 
    void Read(const std::string &filename);
    void Write(const std::string &filename);
    int32_t InDim() const { 
        assert(layers_.size() > 0);
        return layers_[0]->InDim();
    }
    int32_t OutDim() const { 
        assert(layers_.size() > 0);
        return layers_[layers_.size() - 1]->OutDim();
    }

    void Forward(const Matrix<float> &in, Matrix<float> *out); 
    void Info() const;
    void AddLayer(Layer *layer) {
        layers_.push_back(layer);
    }

    // For Quantization
    void Quantize(Net *quantize_net) const;
protected:
    std::vector<Layer *> layers_;
    std::vector<Matrix<float> *> forward_buf_;
};



#endif
