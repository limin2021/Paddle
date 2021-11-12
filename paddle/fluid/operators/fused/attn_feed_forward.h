/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/fused/attn_bias_add.cu.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
class FeedForward {
 public:
  FeedForward(const platform::CUDADeviceContext& dev_ctx, int bsz_seq,
              int output_size, int input_size, bool compute_bias)
      : dev_ctx_(dev_ctx),
        bsz_seq_(bsz_seq),
        output_size_(output_size),
        input_size_(input_size),
        compute_bias_(compute_bias) {}

  ~FeedForward() {}

  // void ComputeForward(const T* weight_data, const T* input_data,
  //                     const T* bias_data, T* output_data, T* bias_out_data) {
  void ComputeForward(const framework::Tensor* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias, framework::Tensor* output,
                      framework::Tensor* bias_out) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // To convert to col-major expression, transa<->transb, A<->B，m<->n.

    // column-major: gemm-tn.
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // column-major: (m,n,k) = output_size,bsz_seq,input_size (weight*input=out)
    // here: (m,n,k) = bsz_seq,output_size,input_size (input*weight=out)
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    blas.GEMM(transA, transB, bsz_seq_, output_size_, input_size_, alpha,
              input->data<T>(), weight->data<T>(), beta, output->data<T>());
    if (compute_bias_) {
      // LaunchBiasAddFwKernel(dev_ctx_, bsz_seq_, output_size_, output_data,
      //                       bias_data, bias_out_data);
      std::vector<const Tensor*> ins;
      std::vector<Tensor*> outs;
      ins.emplace_back(output);
      ins.emplace_back(bias);
      outs.emplace_back(bias_out);
      int elewise_add_axis = -1;
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, elewise_add_axis, AddFunctor<T>());
    }
  }

  // void ComputeBackward(T* input, T* weight, T* d_output, T* d_input,
  //                      T* d_weight, T* d_bias) {
  void ComputeBackward(const framework::Tensor* input,
                       const framework::Tensor* weight,
                       const framework::Tensor* d_output,
                       framework::Tensor* d_input, framework::Tensor* d_weight,
                       framework::Tensor* d_bias) {
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);

    // column-major: gemm-nt, get d_weight.
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    // column-major: (m,n,k): input_size,output_size,bsz (input*dout=dweight)
    // here: (m,n,k): output_size,input_size,bsz (dout*input=dweight)
    blas.GEMM(transA, transB, output_size_, input_size_, bsz_seq_, alpha,
              d_output->data<T>(), input->data<T>(), beta, d_weight->data<T>());

    // column-major: gemm-nn: get d_input.
    transA = CblasNoTrans;
    // column-major: (m,n,k): input_size,bsz,output_size (weight*dout=dinput)
    // here: (m, n, k): bsz, input_size, output_size, (dout*weight=dinput)
    blas.GEMM(transA, transB, bsz_seq_, input_size_, output_size_, alpha,
              d_output->data<T>(), weight->data<T>(), beta, d_input->data<T>());
    if (compute_bias_) {
      LaunchBiasAddBwKernel(dev_ctx_, bsz_seq_, output_size_,
                            d_output->data<T>(), d_bias->data<T>());
      // const auto input_dims = d_output->dims();
      // const auto output_dims = d_bias->dims();
      // bool support_case_1 =
      //     (input_dims.size() == 5 && output_dims.size() == 3 &&
      //      (input_dims[2] == output_dims[0]) &&
      //      (input_dims[3] == output_dims[1]) &&
      //      (input_dims[4] == output_dims[2]));
      // bool support_case_2 =
      //     (input_dims.size() == 3 && output_dims.size() == 1 &&
      //      (input_dims[2] == output_dims[0]));
      // if (support_case_1 || support_case_2) {
      //   gpuStream_t stream = dev_ctx_.stream();
      //   TensorReduceFunctorImpl<T, T, CustomSum>(*d_output, d_bias, {0, 1},
      //                                            stream);
      // } else {
      //   PADDLE_THROW(platform::errors::InvalidArgument(
      //       "Only support reduce when the input dims are [0,1,2,3,4] and "
      //       "output is [2,3,4]"
      //       "or input is [0,1,2] and output is [2]."));
      // }
    }
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;
  int bsz_seq_, output_size_, input_size_;
  bool compute_bias_;
};

}  // namespace operators
}  // namespace paddle
