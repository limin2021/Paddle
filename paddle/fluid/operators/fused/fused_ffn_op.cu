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

#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/blas.h"

#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/fused/fused_ffn_op.h"
#include "paddle/fluid/operators/layer_norm_kernel.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class FusedFfnKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const platform::CUDADeviceContext& ctx,
              const framework::Tensor& a, const framework::Tensor& b,
              framework::Tensor* c) const {
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto a_2d = FoldInitDims(a);
    auto b_2d = FoldInitDims(b);
    auto mat_dim_a = math::CreateMatrixDescriptor(a_2d.dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(b_2d.dims(), 0, false);
    T alpha = static_cast<T>(1.0);
    blas.MatMul(a, mat_dim_a, b, mat_dim_b, alpha, c, T(0));
  }

  void FFN(const framework::Tensor& x, const framework::Tensor& linear1_weight,
           const framework::Tensor* linear1_bias,
           const framework::Tensor& linear2_weight,
           const framework::Tensor* linear2_bias,
           const framework::Tensor* ln1_scale,
           const framework::Tensor* ln1_bias,
           const framework::Tensor* ln2_scale,
           const framework::Tensor* ln2_bias, framework::Tensor* out,
           framework::Tensor* dropout1_mask, framework::Tensor* dropout2_mask,
           framework::Tensor* ln1_mean, framework::Tensor* ln1_variance,
           framework::Tensor* ln2_mean, framework::Tensor* ln2_variance,
           framework::Tensor* linear1_out, framework::Tensor* ln1_out,
           framework::Tensor* dropout1_out, framework::Tensor* dropout2_out,
           const int bsz_seq, const int d_model, const int dim_feedforward,
           const std::string& act_method, const bool normalize_pre_or_post,
           const float epsilon1, const float epsilon2,
           const DropoutParam& dropout_param1,
           const DropoutParam& dropout_param2,
           const platform::CUDADeviceContext& ctx) const {
    FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
        bsz_seq, d_model, epsilon1);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        ctx, bsz_seq, dim_feedforward, dropout_param1);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx, bsz_seq, d_model, dropout_param2, epsilon2);

    auto place = ctx.GetPlace();
    using U = LayerNormParamType<T>;
    const framework::Tensor* in = &x;

    const U* ln1_scale_ptr =
        ln1_scale == nullptr ? nullptr : ln1_scale->data<U>();
    const U* ln1_bias_ptr = ln1_bias == nullptr ? nullptr : ln1_bias->data<U>();
    const U* ln2_scale_ptr =
        ln2_scale == nullptr ? nullptr : ln2_scale->data<U>();
    const U* ln2_bias_ptr = ln2_bias == nullptr ? nullptr : ln2_bias->data<U>();
    const T* linear1_bias_ptr =
        linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
    const T* linear2_bias_ptr =
        linear2_bias == nullptr ? nullptr : linear2_bias->data<T>();

    if (normalize_pre_or_post) {
      pre_layernorm_helper.LayerNorm(
          ctx, x.data<T>(), ln1_scale_ptr, ln1_bias_ptr, ln1_out->data<T>(),
          ln1_mean->data<U>(), ln1_variance->data<U>());
      in = ln1_out;
    }
    { MatMul(ctx, *in, linear1_weight, linear1_out); }
    {
      fused_act_dropout_helper.DropoutActBias(
          ctx, linear1_out->data<T>(), linear1_bias_ptr, act_method,
          dropout1_out->data<T>(), dropout1_mask->data<uint8_t>());
    }
    framework::Tensor linear2_out;
    {
      linear2_out.Resize({bsz_seq, d_model});
      linear2_out.mutable_data<T>(place);
      MatMul(ctx, *dropout1_out, linear2_weight, &linear2_out);
    }
    {
      if (!normalize_pre_or_post) {
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            ctx, linear2_out.data<T>(), x.data<T>(), linear2_bias_ptr,
            ln2_scale_ptr, ln2_bias_ptr, dropout2_out->data<T>(),
            dropout2_mask->data<uint8_t>(), out->data<T>(), ln2_mean->data<U>(),
            ln2_variance->data<U>());
      } else {
        fused_dropout_layernorm_helper.ResidualDropoutBias(
            ctx, linear2_out.data<T>(), x.data<T>(), linear2_bias_ptr,
            out->data<T>(), dropout2_mask->data<uint8_t>());
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto x = context.Input<framework::Tensor>("X");
    auto linear1_weight = context.Input<framework::Tensor>("Linear1Weight");
    auto linear1_bias = context.Input<framework::Tensor>("Linear1Bias");
    auto linear2_weight = context.Input<framework::Tensor>("Linear2Weight");
    auto linear2_bias = context.Input<framework::Tensor>("Linear2Bias");
    auto ln1_scale = context.Input<framework::Tensor>("Ln1Scale");
    auto ln1_bias = context.Input<framework::Tensor>("Ln1Bias");
    auto ln2_scale = context.Input<framework::Tensor>("Ln2Scale");
    auto ln2_bias = context.Input<framework::Tensor>("Ln2Bias");

    auto ln1_mean = context.Output<framework::Tensor>("Ln1Mean");
    auto ln1_variance = context.Output<framework::Tensor>("Ln1Variance");
    auto ln2_mean = context.Output<framework::Tensor>("Ln2Mean");
    auto ln2_variance = context.Output<framework::Tensor>("Ln2Variance");
    auto out = context.Output<framework::Tensor>("Out");
    auto dropout1_mask = context.Output<framework::Tensor>("Dropout1Mask");
    auto dropout2_mask = context.Output<framework::Tensor>("Dropout2Mask");
    auto linear1_out = context.Output<framework::Tensor>("Linear1Out");
    auto ln1_out = context.Output<framework::Tensor>("Ln1Out");
    auto dropout1_out = context.Output<framework::Tensor>("Dropout1Out");
    auto dropout2_out = context.Output<framework::Tensor>("Dropout2Out");

    const std::string act_method = context.Attr<std::string>("act_method");
    const bool normalize_pre_or_post =
        context.Attr<bool>("normalize_pre_or_post");
    const float epsilon1 = context.Attr<float>("epsilon1");
    const float epsilon2 = context.Attr<float>("epsilon2");

    DropoutParam dropout_param1(context, 1);
    DropoutParam dropout_param2(context, 2);

    using U = LayerNormParamType<T>;
    auto place = context.GetPlace();
    out->mutable_data<T>(place);
    dropout1_mask->mutable_data<uint8_t>(place);
    dropout2_mask->mutable_data<uint8_t>(place);
    ln1_mean->mutable_data<U>(place);
    ln1_variance->mutable_data<U>(place);
    ln2_mean->mutable_data<U>(place);
    ln2_variance->mutable_data<U>(place);
    linear1_out->mutable_data<T>(place);
    ln1_out->mutable_data<T>(place);
    dropout1_out->mutable_data<T>(place);
    dropout2_out->mutable_data<T>(place);

    auto x_dim = x->dims();
    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(x_dim), 0, false);

    auto dim = linear1_weight->dims();
    int d_model = dim[0];
    int dim_feedforward = dim[dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

    FFN(*x, *linear1_weight, linear1_bias, *linear2_weight, linear2_bias,
        ln1_scale, ln1_bias, ln2_scale, ln2_bias, out, dropout1_mask,
        dropout2_mask, ln1_mean, ln1_variance, ln2_mean, ln2_variance,
        linear1_out, ln1_out, dropout1_out, dropout2_out, bsz_seq, d_model,
        dim_feedforward, act_method, normalize_pre_or_post, epsilon1, epsilon2,
        dropout_param1, dropout_param2, context.cuda_device_context());
  }
};

template <typename DeviceContext, typename T>
class FusedFfnGradKernel : public framework::OpKernel<T> {
 public:
  void MatMulGrad(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& d_out, const framework::Tensor& a,
                  const framework::Tensor& b, framework::Tensor* d_a,
                  framework::Tensor* d_b) const {
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto a_2d = FoldInitDims(a);
    auto b_2d = FoldInitDims(b);
    auto mat_dim_a = math::CreateMatrixDescriptor(a_2d.dims(), 0, true);
    auto mat_dim_b = math::CreateMatrixDescriptor(b_2d.dims(), 0, true);
    auto mat_dim_dout = math::CreateMatrixDescriptor(d_out.dims(), 0, false);
    T alpha = static_cast<T>(1.0);
    blas.MatMul(d_out, mat_dim_dout, b, mat_dim_b, alpha, d_a, T(0));
    blas.MatMul(a, mat_dim_a, d_out, mat_dim_dout, alpha, d_b, T(0));
  }

  void FFNGrad(
      const framework::Tensor& d_out, const framework::Tensor& x,
      const framework::Tensor& dropout1_mask,
      const framework::Tensor& dropout2_mask,
      const framework::Tensor& linear1_out, const framework::Tensor& ln1_out,
      const framework::Tensor& dropout1_out,
      const framework::Tensor& dropout2_out,
      const framework::Tensor& linear1_weight,
      const framework::Tensor* linear1_bias,
      const framework::Tensor& linear2_weight,
      const framework::Tensor* ln1_gamma, const framework::Tensor* ln1_beta,
      const framework::Tensor& ln1_mean, const framework::Tensor& ln1_variance,
      const framework::Tensor* ln2_gamma, const framework::Tensor* ln2_beta,
      const framework::Tensor& ln2_mean, const framework::Tensor& ln2_variance,
      framework::Tensor* d_x, framework::Tensor* d_linear1_weight,
      framework::Tensor* d_linear1_bias, framework::Tensor* d_linear2_weight,
      framework::Tensor* d_linear2_bias, framework::Tensor* d_ln1_gamma,
      framework::Tensor* d_ln1_beta, framework::Tensor* d_ln2_gamma,
      framework::Tensor* d_ln2_beta, const int bsz_seq, const int d_model,
      const int dim_feedforward, const DropoutParam& dropout_param1,
      const DropoutParam& dropout_param2, const std::string& act_method,
      const bool normalize_pre_or_post, const float epsilon1,
      const float epsilon2, const platform::CUDADeviceContext& ctx) const {
    FusedDropoutLayerNormHelper<T, uint8_t> pre_layernorm_helper(
        bsz_seq, d_model, epsilon1);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        ctx, bsz_seq, dim_feedforward, dropout_param1);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx, bsz_seq, d_model, dropout_param2, epsilon2);

    auto place = ctx.GetPlace();
    using U = LayerNormParamType<T>;
    const U* ln1_gamma_ptr =
        ln1_gamma == nullptr ? nullptr : ln1_gamma->data<U>();
    const U* ln1_beta_ptr = ln1_beta == nullptr ? nullptr : ln1_beta->data<U>();
    const U* ln2_gamma_ptr =
        ln2_gamma == nullptr ? nullptr : ln2_gamma->data<U>();
    const U* ln2_beta_ptr = ln2_beta == nullptr ? nullptr : ln2_beta->data<U>();
    const T* linear1_bias_ptr =
        linear1_bias == nullptr ? nullptr : linear1_bias->data<T>();
    T* d_linear1_bias_ptr =
        d_linear1_bias == nullptr ? nullptr : d_linear1_bias->data<T>();
    T* d_linear2_bias_ptr =
        d_linear2_bias == nullptr ? nullptr : d_linear2_bias->data<T>();
    U* d_ln1_gamma_ptr =
        d_ln1_gamma == nullptr ? nullptr : d_ln1_gamma->data<U>();
    U* d_ln1_beta_ptr = d_ln1_beta == nullptr ? nullptr : d_ln1_beta->data<U>();
    U* d_ln2_gamma_ptr =
        d_ln2_gamma == nullptr ? nullptr : d_ln2_gamma->data<U>();
    U* d_ln2_beta_ptr = d_ln2_beta == nullptr ? nullptr : d_ln2_beta->data<U>();

    framework::Tensor d_linear2_out, d_dropout2_out, d_residual;
    d_linear2_out.Resize({bsz_seq, d_model});
    d_linear2_out.mutable_data<T>(place);
    d_dropout2_out.Resize({bsz_seq, d_model});
    d_dropout2_out.mutable_data<T>(place);
    d_residual.Resize({bsz_seq, d_model});
    d_residual.mutable_data<T>(place);

    if (normalize_pre_or_post) {
      fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
          ctx, d_out.data<T>(), dropout2_mask.data<uint8_t>(),
          d_linear2_out.data<T>(), d_residual.data<T>(), d_linear2_bias_ptr);
    } else {
      fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
          ctx, d_out.data<T>(), dropout2_out.data<T>(),
          dropout2_mask.data<uint8_t>(), ln2_gamma_ptr, ln2_mean.data<U>(),
          ln2_variance.data<U>(), d_dropout2_out.data<T>(), d_ln2_gamma_ptr,
          d_ln2_beta_ptr, d_linear2_out.data<T>(), d_linear2_bias_ptr,
          d_residual.data<T>());
    }

    framework::Tensor d_dropout1_out;
    d_dropout1_out.Resize({bsz_seq, dim_feedforward});
    d_dropout1_out.mutable_data<T>(place);
    MatMulGrad(ctx, d_linear2_out, dropout1_out, linear2_weight,
               &d_dropout1_out, d_linear2_weight);

    framework::Tensor d_linear1_out;
    d_linear1_out.Resize({bsz_seq, dim_feedforward});
    d_linear1_out.mutable_data<T>(place);
    fused_act_dropout_helper.DropoutActBiasGrad(
        ctx, d_dropout1_out.data<T>(), linear1_out.data<T>(), linear1_bias_ptr,
        dropout1_mask.data<uint8_t>(), d_linear1_out.data<T>(),
        d_linear1_bias_ptr, act_method);

    if (normalize_pre_or_post) {
      framework::Tensor d_ln1_out;
      d_ln1_out.Resize({bsz_seq, d_model});
      d_ln1_out.mutable_data<T>(place);
      MatMulGrad(ctx, d_linear1_out, ln1_out, linear1_weight, &d_ln1_out,
                 d_linear1_weight);

      pre_layernorm_helper.LayerNormGrad(ctx, d_ln1_out.data<T>(), x.data<T>(),
                                         ln1_gamma_ptr, ln1_mean.data<U>(),
                                         ln1_variance.data<U>(), d_x->data<T>(),
                                         d_ln1_gamma_ptr, d_ln1_beta_ptr);
    } else {
      MatMulGrad(ctx, d_linear1_out, x, linear1_weight, d_x, d_linear1_weight);
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    using U = LayerNormParamType<T>;
    auto d_out =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto x = *context.Input<framework::Tensor>("X");
    auto dropout1_mask = *context.Input<framework::Tensor>("Dropout1Mask");
    auto dropout2_mask = *context.Input<framework::Tensor>("Dropout2Mask");
    auto linear1_out = *context.Input<framework::Tensor>("Linear1Out");
    auto ln1_out = *context.Input<framework::Tensor>("Ln1Out");
    auto dropout1_out = *context.Input<framework::Tensor>("Dropout1Out");
    auto dropout2_out = *context.Input<framework::Tensor>("Dropout2Out");
    auto linear1_weight = *context.Input<framework::Tensor>("Linear1Weight");
    auto linear1_bias = context.Input<framework::Tensor>("Linear1Bias");
    auto linear2_weight = *context.Input<framework::Tensor>("Linear2Weight");
    auto ln1_mean = *context.Input<framework::Tensor>("Ln1Mean");
    auto ln1_variance = *context.Input<framework::Tensor>("Ln1Variance");
    auto ln1_scale = context.Input<framework::Tensor>("Ln1Scale");
    auto ln1_bias = context.Input<framework::Tensor>("Ln1Bias");
    auto ln2_mean = *context.Input<framework::Tensor>("Ln2Mean");
    auto ln2_variance = *context.Input<framework::Tensor>("Ln2Variance");
    auto ln2_scale = context.Input<framework::Tensor>("Ln2Scale");
    auto ln2_bias = context.Input<framework::Tensor>("Ln2Bias");

    auto d_x = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto d_ln1_scale =
        context.Output<framework::Tensor>(framework::GradVarName("Ln1Scale"));
    auto d_ln1_bias =
        context.Output<framework::Tensor>(framework::GradVarName("Ln1Bias"));
    auto d_ln2_scale =
        context.Output<framework::Tensor>(framework::GradVarName("Ln2Scale"));
    auto d_ln2_bias =
        context.Output<framework::Tensor>(framework::GradVarName("Ln2Bias"));
    auto d_linear1_weight = context.Output<framework::Tensor>(
        framework::GradVarName("Linear1Weight"));
    auto d_linear1_bias = context.Output<framework::Tensor>(
        framework::GradVarName("Linear1Bias"));
    auto d_linear2_weight = context.Output<framework::Tensor>(
        framework::GradVarName("Linear2Weight"));
    auto d_linear2_bias = context.Output<framework::Tensor>(
        framework::GradVarName("Linear2Bias"));

    const float epsilon1 = context.Attr<float>("epsilon1");
    const float epsilon2 = context.Attr<float>("epsilon2");
    const bool normalize_pre_or_post =
        context.Attr<bool>("normalize_pre_or_post");
    const std::string act_method = context.Attr<std::string>("act_method");
    DropoutParam dropout_param1(context, 1);
    DropoutParam dropout_param2(context, 2);

    auto place = context.GetPlace();
    d_x->mutable_data<T>(place);
    if (d_ln1_scale) {
      d_ln1_scale->mutable_data<U>(place);
    }
    if (d_ln1_bias) {
      d_ln1_bias->mutable_data<U>(place);
    }
    if (d_ln2_scale) {
      d_ln2_scale->mutable_data<U>(place);
    }
    if (d_ln2_bias) {
      d_ln2_bias->mutable_data<U>(place);
    }
    if (d_linear1_bias) {
      d_linear1_bias->mutable_data<T>(place);
    }
    if (d_linear2_bias) {
      d_linear2_bias->mutable_data<T>(place);
    }
    d_linear1_weight->mutable_data<T>(place);
    d_linear2_weight->mutable_data<T>(place);

    auto x_dim = x.dims();
    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(x_dim), 0, false);

    auto linear1_weight_dim = linear1_weight.dims();
    int d_model = linear1_weight_dim[0];
    int dim_feedforward = linear1_weight_dim[linear1_weight_dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

    FFNGrad(d_out, x, dropout1_mask, dropout2_mask, linear1_out, ln1_out,
            dropout1_out, dropout2_out, linear1_weight, linear1_bias,
            linear2_weight, ln1_scale, ln1_bias, ln1_mean, ln1_variance,
            ln2_scale, ln2_bias, ln2_mean, ln2_variance, d_x, d_linear1_weight,
            d_linear1_bias, d_linear2_weight, d_linear2_bias, d_ln1_scale,
            d_ln1_bias, d_ln2_scale, d_ln2_bias, bsz_seq, d_model,
            dim_feedforward, dropout_param1, dropout_param2, act_method,
            normalize_pre_or_post, epsilon1, epsilon2,
            context.cuda_device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_ffn, ops::FusedFfnKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FusedFfnKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FusedFfnKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    fused_ffn_grad,
    ops::FusedFfnGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FusedFfnGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::FusedFfnGradKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::float16>);
