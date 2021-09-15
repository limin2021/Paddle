#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
import paddle
from paddle import _C_ops

__all__ = []


def _verify_dropout_param(p, mode):
    if not isinstance(p, (float, int)):
        raise TypeError("p argument should be a number")
    if p < 0 or p > 1:
        raise ValueError("p argument should between 0 and 1")
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")


def fused_ffn(x,
              linear1_weight,
              linear2_weight,
              linear1_bias=None,
              linear2_bias=None,
              ln1_scale=None,
              ln1_bias=None,
              ln2_scale=None,
              ln2_bias=None,
              dropout_prob1=0.5,
              dropout_prob2=0.5,
              act_method="relu",
              epsilon1=1e-5,
              epsilon2=1e-5,
              dropout_implementation1='upscale_in_train',
              dropout_implementation2='upscale_in_train',
              normalize_pre_or_post=False,
              name=None):
    r"""
    Fused feedforward operator.
    the operator is the same as the following pseudo code:
    .. code-block:: python

        residual = x
        if normalize_pre_or_post:
            out = layer_norm(x)
        out = linear(dropout(activation(linear(out))))
        out = residual + dropout(out)
        if not normalize_pre_or_post:
            out = layer_norm(out)

    Parameters:
        x (Tensor): The input Tensor with data type float16, float32, float64.
        linear1_weight (Tensor): The weight of first linear.
        linear2_weight (Tensor): The weight of second linear.
        linear1_bias (Tensor): The bias of first linear. Default is None.
        linear2_bias (Tensor): The bias of first linear. Default is None.
        ln1_scale (Tensor): The scale of first layer_norm. Default is None.
        ln1_bias (Tensor): The bias of first layer_norm. Default is None.
        ln2_scale (Tensor): The scale of second layer_norm. Default is None.
        ln2_bias (Tensor): The bias of second layer_norm. Default is None.
        dropout_prob1 (float|int, optional): The first dropout probility of setting units to zero. Default 0.5.
        dropout_prob2 (float|int, optional): The second dropout probility of setting units to zero. Default 0.5.
        act_method (string, optional): The activation. Default is relu.
        epsilon1 (float, optional): the epsilon of first layer_norm, a small value added to the variance to prevent division by zero. Default: 1e-05
        epsilon2 (float, optional): the epsilon of first layer_norm, a small value added to the variance to prevent division by zero. Default: 1e-05
        dropout_implementation1 (string, option): ['upscale_in_train'(default) | 'dowscale_in_infer'].
            1. upscale_in_train(default), upscale the output at training time
                - train: out = input * mask / ( 1.0 - dropout_prob  )
                - inference: out = input
            2. downscale_in_infer, downscale the output at inference
                - train: out = input * mask
                - inference: out = input * (1.0 - dropout_prob)
        dropout_implementation2 (string, option): the dropout_implementation2 of second dropout2
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    """
    _verify_dropout_param(dropout_prob1, dropout_implementation1)
    _verify_dropout_param(dropout_prob2, dropout_implementation2)

    if in_dygraph_mode():
        out, _, _, _, _, _, _, _, _, _, _ = _C_ops.fused_ffn(
            x, None, None, linear1_weight, linear1_bias, linear2_weight,
            linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias,
            'normalize_pre_or_post', normalize_pre_or_post, 'epsilon1',
            epsilon1, 'epsilon2', epsilon2, 'act_method', act_method,
            'dropout_prob1', dropout_prob1, 'dropout_prob2', dropout_prob2,
            'dropout_implementation1', dropout_implementation1,
            'dropout_implementation2', dropout_implementation2)
        return out

    helper = LayerHelper("fused_ffn", **locals())
    dtype = x.dtype
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'fused_ffn')
    check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'], 'fused_ffn')

    out = helper.create_variable_for_type_inference(x.dtype)
    dropout1_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    dropout2_mask = helper.create_variable_for_type_inference(
        'uint8', stop_gradient=True)
    ln1_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_mean = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln2_variance = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    linear1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    ln1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout1_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)
    dropout2_out = helper.create_variable_for_type_inference(
        x.dtype, stop_gradient=True)

    helper.append_op(
        type='fused_ffn',
        inputs={
            'X': x,
            'Linear1Weight': linear1_weight,
            'Linear1Bias': linear1_bias,
            'Linear2Weight': linear2_weight,
            'Linear2Bias': linear2_bias,
            'Ln1Scale': ln1_scale,
            'Ln1Bias': ln1_bias,
            'Ln2Scale': ln2_scale,
            'Ln2Bias': ln2_bias,
        },
        outputs={
            'Out': out,
            'Dropout1Mask': dropout1_mask,
            'Dropout2Mask': dropout2_mask,
            'Ln1Mean': ln1_mean,
            'Ln1Variance': ln1_variance,
            'Ln2Mean': ln2_mean,
            'Ln2Variance': ln2_variance,
            'Linear1Out': linear1_out,
            'Ln1Out': ln1_out,
            'Dropout1Out': dropout1_out,
            'Dropout2Out': dropout2_out,
        },
        attrs={
            'dropout_prob1': dropout_prob1,
            'dropout_prob2': dropout_prob2,
            'act_method': act_method,
            'normalize_pre_or_post': normalize_pre_or_post,
            'epsilon1': epsilon1,
            'epsilon2': epsilon2,
            'dropout_implementation1': dropout_implementation1,
            'dropout_implementation2': dropout_implementation2,
        })

    return out
