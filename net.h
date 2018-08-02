// Copyright (c) 2017 Personal (Binbin Zhang)
// Created on 2017-06-07
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NET_H_
#define NET_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "utils.h"

/* Matrix & Vector Defination */
template <class DType, int32_t Dim>
class Tensor {
 public:
  explicit Tensor(DType* data = nullptr): data_(data), shape_(Dim, 0),
                                          holder_(false) {}
  explicit Tensor(const Tensor<DType, Dim>& tensor) {
    CopyFrom(tensor);
  }
  virtual ~Tensor() {
    if (holder_ && data_ != nullptr) delete [] data_;
  }
  virtual void Read(const std::string& filename);
  virtual void Read(std::istream& is);
  virtual void Write(const std::string& filename) const;
  virtual void Write(std::ostream& os) const;
  void Resize(const std::vector<int32_t>& shape);
  int32_t Size() const {
    return GetShapeSize(shape_);
  }
  DType* Data() const { return data_; }
  std::vector<int32_t> Shape() const { return shape_; }
  virtual void CopyFrom(const Tensor<DType, Dim>& tensor);
  virtual void Scale(float alpha);

 protected:
  int32_t GetShapeSize(const std::vector<int32_t>& shape) const;

 protected:
  DType* data_;
  std::vector<int32_t> shape_;
  bool holder_;
};

template <typename DType>
class Vector;

template <typename DType>
class Matrix : public Tensor<DType, 2> {
 public:
  explicit Matrix(int32_t row = 0, int32_t col = 0) {
    Resize(row, col);
  }
  Matrix(DType* data, int32_t row, int32_t col): Tensor<DType, 2>(data) {
    this->shape_[0] = row;
    this->shape_[1] = col;
  }
  void Resize(int32_t row, int32_t col) {
    std::vector<int32_t> shape = { row, col };
    Tensor<DType, 2>::Resize(shape);
  }
  int32_t NumRows() const { return this->shape_[0]; }
  int32_t NumCols() const { return this->shape_[1]; }
  const DType operator () (int r, int c) const {
    CHECK(r < NumRows());
    CHECK(c < NumCols());
    return *(this->data_ + r * NumCols() + c);
  }
  DType& operator () (int r, int c) {
    CHECK(r < NumRows());
    CHECK(c < NumCols());
    return *(this->data_ + r * NumCols() + c);
  }
  // *this = alpha*this + mat1*mat2
  void Mul(const Matrix<DType>& mat1, const Matrix<DType>& mat2,
           bool transpose = false, float alpha = 0.0);
  void Transpose(const Matrix<DType> &mat);
  void AddVec(const Vector<DType> &vec, float alpha = 1.0f);
  Vector<DType> Row(int row) const;
  Matrix<DType> RowRange(int start, int length) const;
};

template <class DType>
class Vector: public Tensor<DType, 1> {
 public:
  explicit Vector(int32_t dim = 0) {
    Resize(dim);
  }
  Vector(DType* data, int dim): Tensor<DType, 1>(data) {
    CHECK(this->shape_.size() == 1);
    this->shape_[0] = dim;
  }
  void Resize(int32_t dim) {
    std::vector<int32_t> shape = { dim };
    Tensor<DType, 1>::Resize(shape);
  }
  const DType operator () (int n) const {
    CHECK(n < this->shape_[0]);
    return *(this->data_ + n);
  }
  DType& operator () (int n) {
    CHECK(n < this->shape_[0]);
    return *(this->data_ + n);
  }
  void Add(const Vector<DType>& vec, float alpha = 1.0);
};


/* Quantization Functions */

void FindMinMax(const float* data, int n, float* min, float* max);
void ChooseQuantizationParams(float min, float max, float* scale,
                              uint8_t* zero_point);
void QuantizeData(const float* src, int n, float scale, uint8_t zero_point,
                  uint8_t* dest);

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
  explicit Layer(int32_t in_dim = 0, int32_t out_dim = 0,
                 LayerType type = kUnknown):
        in_dim_(in_dim), out_dim_(out_dim), type_(type) {}
  virtual ~Layer() {}
  void Read(std::istream& is);
  void Write(std::ostream& os);
  void Forward(const Matrix<float>& in, Matrix<float>* out);
  int32_t InDim() const { return in_dim_; }
  int32_t OutDim() const { return out_dim_; }
  void SetInputDim(int32_t in_dim) { in_dim_ = in_dim; }
  void SetOutputDim(int32_t out_dim) { out_dim_ = out_dim; }
  virtual LayerType Type() const { return type_; }
  void Info() const {
    std::cout << LayerTypeToString(type_) << " in_dim " << in_dim_
              << " out_dim " << out_dim_ << "\n";
  }
  virtual Layer* Copy() const = 0;
  virtual Layer* Quantize() const {
    return this->Copy();
  }

 protected:
  virtual void ForwardFunc(const Matrix<float>& in, Matrix<float>* out) = 0;
  virtual void ReadData(std::istream& is) {}
  virtual void WriteData(std::ostream& os) {}
  int32_t in_dim_, out_dim_;
  LayerType type_;
};

class ReLU: public Layer {
 public:
  explicit ReLU(int32_t in_dim = 0, int32_t out_dim = 0):
      Layer(in_dim, out_dim, kReLU) {}
  Layer* Copy() const { return new ReLU(*this); }

 private:
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
};

class Sigmoid: public Layer {
 public:
  explicit Sigmoid(int32_t in_dim = 0, int32_t out_dim = 0):
      Layer(in_dim, out_dim, kSigmoid) {}
  Layer* Copy() const { return new Sigmoid(*this); }

 private:
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
};

class Tanh: public Layer {
 public:
  explicit Tanh(int32_t in_dim = 0, int32_t out_dim = 0):
      Layer(in_dim, out_dim, kTanh) {}
  Layer* Copy() const { return new Tanh(*this); }

 private:
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
};

class Softmax: public Layer {
 public:
  explicit Softmax(int32_t in_dim = 0, int32_t out_dim = 0):
          Layer(in_dim, out_dim, kSoftmax) {}
  Layer* Copy() const { return new Softmax(*this); }

 private:
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
};

class FullyConnect : public Layer {
 public:
  explicit FullyConnect(int32_t in_dim = 0, int32_t out_dim = 0):
      Layer(in_dim, out_dim, kFullyConnect) {}
  const Matrix<float>& W() { return w_; }
  const Vector<float>& B() { return b_; }
  Layer* Copy() const { return new FullyConnect(*this); }
  virtual Layer* Quantize() const;

 private:
  void ReadData(std::istream& is);
  void WriteData(std::ostream& os);
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
  Matrix<float> w_;  // w_ is cols major, so it's size (out_dim, in_dim)
  Vector<float> b_;  // size(out_dim)
};

class QuantizeFullyConnect : public Layer {
 public:
  explicit QuantizeFullyConnect(int32_t in_dim = 0, int32_t out_dim = 0):
      Layer(in_dim, out_dim, kQuantizeFullyConnect) {}
  void QuantizeFrom(const Matrix<float>& w, const Vector<float>& b);
  Layer* Copy() const { return new QuantizeFullyConnect(*this); }
  void SetWeight(const Matrix<uint8_t>& weight) { w_.CopyFrom(weight); }
  void SetBias(const Vector<float>& bias) { b_.CopyFrom(bias); }
  void SetWeightScale(float scale) { w_scale_ = scale; }
  void SetWeightZeroPoint(uint8_t zero_point) { w_zero_point_ = zero_point; }

 private:
  void ReadData(std::istream& is);
  void WriteData(std::ostream& os);
  void ForwardFunc(const Matrix<float>& in, Matrix<float>* out);
  Matrix<uint8_t> w_;  // w_ is cols major, so it's size (out_dim, in_dim)
  float w_scale_;
  uint8_t w_zero_point_;

  Vector<float> b_;  // use float bias
  Matrix<int32_t> quantize_out_;
  Matrix<uint8_t> quantize_in_;
};


/* Net Defination */
class Net {
 public:
  explicit Net(std::string filename) {
    Read(filename);
  }
  Net() {}
  ~Net();
  void Clear();
  void Read(const std::string& filename);
  void Write(const std::string& filename);
  int32_t InDim() const {
    CHECK(layers_.size() > 0);
    return layers_[0]->InDim();
  }
  int32_t OutDim() const {
    CHECK(layers_.size() > 0);
    return layers_[layers_.size() - 1]->OutDim();
  }

  void Forward(const Matrix<float>& in, Matrix<float>* out);
  void Info() const;
  void AddLayer(Layer* layer) {
    layers_.push_back(layer);
  }

  // For Quantization
  void Quantize(Net* quantize_net) const;
  // For xdecoder
  bool IsLastLayerSoftmax() const {
    CHECK(layers_.size() > 0);
    return layers_[layers_.size() - 1]->Type() == kSoftmax;
  }

 protected:
  std::vector<Layer*> layers_;
  std::vector<Matrix<float>*> forward_buf_;
};

#endif  // UTILS_H_
