# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import collections
import numpy as np

from paddle.framework import ParamAttr
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.fluid.core as core
from paddle.nn.layer import fused_transformer
import paddle.nn.functional as F
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout

import unittest

place = paddle.CUDAPlace(0)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedFFNOp(unittest.TestCase):
    def getDtype(self):
        self.dtype = "float32"
        self.layer_norm_dtype = "float32"

    def getShape(self):
        self.batch_size = np.random.randint(1, 64)
        self.query_length = np.random.randint(32, 256)
        self.d_model = np.random.randint(32, 1024)
        self.dim_feedforward = np.random.randint(32, 4096)

    def getDiff(self):
        self.rtol = 1e-4
        self.atol = 1e-5

    def getActivation(self):
        self.act_method = "gelu"

    def getNormalizeBefore(self):
        self.normalize_before = False

    def setUp(self):
        paddle.disable_static()
        self.getDtype()
        self.getShape()
        self.getDiff()
        self.getActivation()
        self.getNormalizeBefore()

        paddle.set_default_dtype(self.dtype)
        self.weight_attr = None
        self.bias_attr = None

        self.weight_attrs = fused_transformer._convert_param_attr_to_list(
            self.weight_attr, 2)
        self.bias_attrs = fused_transformer._convert_param_attr_to_list(
            self.bias_attr, 2)

        self.linear1 = Linear(
            self.d_model,
            self.dim_feedforward,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1])
        self.linear2 = Linear(
            self.dim_feedforward,
            self.d_model,
            self.weight_attrs[1],
            bias_attr=self.bias_attrs[1])

        paddle.set_default_dtype(self.layer_norm_dtype)
        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.dropout = Dropout(0.0, mode="upscale_in_train")
        self.dropout1 = Dropout(0.0, mode="upscale_in_train")
        self.dropout2 = Dropout(0.0, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

        self.src = np.random.random((self.batch_size, self.query_length,
                                     self.d_model)).astype(self.dtype)
        self.dout = np.random.random((self.batch_size, self.query_length,
                                      self.d_model)).astype(self.dtype)

    def Base(self):
        paddle.disable_static()
        tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
        residual = paddle.to_tensor(self.src)
        if self.normalize_before:
            ln1_out = self.norm1(tensor_src)

            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(ln1_out))))
            dropout2_out = residual + self.dropout2(linear2_out)
            paddle.autograd.backward([dropout2_out],
                                     [paddle.to_tensor(self.dout)])
        else:
            linear2_out = self.linear2(
                self.dropout(self.activation(self.linear1(tensor_src))))
            dropout2_out = residual + self.dropout2(linear2_out)
            dropout2_out = self.norm2(dropout2_out)
            paddle.autograd.backward([dropout2_out],
                                     [paddle.to_tensor(self.dout)])
        return dropout2_out, tensor_src.grad

    def FusedFFN(self):
        paddle.disable_static()
        with fluid.dygraph.guard(fluid.CUDAPlace(0)):
            linear1_weight = paddle.to_tensor(
                self.linear1.weight, stop_gradient=False)
            linear1_bias = paddle.to_tensor(
                self.linear1.bias, stop_gradient=False)
            linear2_weight = paddle.to_tensor(
                self.linear2.weight, stop_gradient=False)
            linear2_bias = paddle.to_tensor(
                self.linear2.bias, stop_gradient=False)
            ln1_scale = paddle.to_tensor(self.norm1.weight, stop_gradient=False)
            ln1_bias = paddle.to_tensor(self.norm1.bias, stop_gradient=False)
            ln2_scale = paddle.to_tensor(self.norm2.weight, stop_gradient=False)
            ln2_bias = paddle.to_tensor(self.norm2.bias, stop_gradient=False)
            seed1 = None
            seed2 = None
            x = paddle.to_tensor(self.src, stop_gradient=False)
            out = F.fused_ffn(
                x,
                linear1_weight,
                linear2_weight,
                seed1,
                seed2,
                linear1_bias,
                linear2_bias,
                ln1_scale,
                ln1_bias,
                ln2_scale,
                ln2_bias,
                0.0,
                0.0,
                act_method=self.act_method,
                normalize_pre_or_post=self.normalize_before)
            paddle.autograd.backward([out], [paddle.to_tensor(self.dout)])
            return out, x.grad

    def test_fused_ffn(self):
        base_out, base_grad = self.Base()
        fused_out, fused_grad = self.FusedFFN()

        np.testing.assert_allclose(
            base_out.numpy(), fused_out.numpy(), rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(
            base_grad.numpy(),
            fused_grad.numpy(),
            rtol=self.rtol,
            atol=self.atol)


class TestFusedFFNOpFp16(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float16"
        self.layer_norm_dtype = "float32"

    def getDiff(self):
        self.rtol = 1e-2
        self.atol = 1e-3

    def getShape(self):
        self.batch_size = 1
        self.query_length = 8
        self.d_model = 8
        self.dim_feedforward = 8


class TestFusedFFNOpFp64(TestFusedFFNOp):
    def getDtype(self):
        self.dtype = "float64"
        self.layer_norm_dtype = "float64"


class TestFusedFFNOpActivation(TestFusedFFNOp):
    def getActivation(self):
        self.act_method = "relu"


class TestFusedFFNOpNormalizeBefore(TestFusedFFNOp):
    def getNormalizeBefore(self):
        self.normalize_before = True

    def getShape(self):
        self.batch_size = 1
        self.query_length = 1
        self.d_model = 8
        self.dim_feedforward = 8


class TestFusedFFNOpApi(TestFusedFFNOp):
    def setUp(self):
        self.getDtype()
        self.getShape()
        self.getDiff()
        self.getActivation()
        self.getNormalizeBefore()
        self.weight_attr = None
        self.bias_attr = None

        self.weight_attrs = fused_transformer._convert_param_attr_to_list(
            self.weight_attr, 2)
        self.bias_attrs = fused_transformer._convert_param_attr_to_list(
            self.bias_attr, 2)
        self.ffn_layer = fused_transformer.FusedFeedForward(
            self.d_model, self.dim_feedforward, 0.0, self.act_method, 0.0,
            self.normalize_before, self.weight_attrs[1], self.bias_attrs[1])

        self.ln1_scale = self.ffn_layer._ln1_scale
        self.ln1_bias = self.ffn_layer._ln1_bias
        self.ln2_scale = self.ffn_layer._ln2_scale
        self.ln2_bias = self.ffn_layer._ln2_bias
        self.linear1_weight = self.ffn_layer._linear1_weight
        self.linear1_bias = self.ffn_layer._linear1_bias
        self.linear2_weight = self.ffn_layer._linear2_weight
        self.linear2_bias = self.ffn_layer._linear2_bias

        self.src = np.random.random((self.batch_size, self.query_length,
                                     self.d_model)).astype(self.dtype)
        self.dout = np.random.random((self.batch_size, self.query_length,
                                      self.d_model)).astype(self.dtype)

        self.dropout1 = Dropout(0.0, mode="upscale_in_train")
        self.dropout2 = Dropout(0.0, mode="upscale_in_train")
        self.activation = getattr(F, self.act_method)

    def Base(self):
        paddle.disable_static()
        tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
        residual = paddle.to_tensor(self.src)
        if self.normalize_before:
            ln1_out = F.layer_norm(tensor_src,
                                   list([self.d_model]), self.ln1_scale,
                                   self.ln1_bias)
            linear1_out = F.linear(ln1_out, self.linear1_weight,
                                   self.linear1_bias)
            act_out = self.activation(linear1_out)
            dropout1_out = self.dropout1(act_out)
            linear2_out = F.linear(dropout1_out, self.linear2_weight,
                                   self.linear2_bias)
            dropout2_out = residual + self.dropout2(linear2_out)
            paddle.autograd.backward([dropout2_out],
                                     [paddle.to_tensor(self.dout)], True)
            return dropout2_out, tensor_src.grad
        else:
            linear1_out = F.linear(tensor_src, self.linear1_weight,
                                   self.linear1_bias)
            act_out = self.activation(linear1_out)
            dropout1_out = self.dropout1(act_out)
            linear2_out = F.linear(dropout1_out, self.linear2_weight,
                                   self.linear2_bias)
            dropout2_out = residual + self.dropout2(linear2_out)
            dropout2_out = F.layer_norm(dropout2_out,
                                        list([self.d_model]), self.ln2_scale,
                                        self.ln2_bias)
            paddle.autograd.backward([dropout2_out],
                                     [paddle.to_tensor(self.dout)], True)
            return dropout2_out, tensor_src.grad

    def FusedFFN(self):
        paddle.disable_static()
        tensor_src = paddle.to_tensor(self.src, stop_gradient=False)
        out = self.ffn_layer(tensor_src)
        paddle.autograd.backward([out], [paddle.to_tensor(self.dout)])
        return out, tensor_src.grad


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class APITestStaticFusedFFN(unittest.TestCase):
    def test_static(self):
        paddle.enable_static()
        dtype = "float32"
        layer_norm_dtype = "float32"
        batch_size = 1
        d_model = 8
        dim_feedforward = 8

        x = paddle.static.data(
            name='x', shape=[batch_size, d_model, dim_feedforward], dtype=dtype)
        linear1_weight = paddle.static.data(
            name='linear1_weight',
            shape=[d_model, dim_feedforward],
            dtype=dtype)
        linear1_bias = paddle.static.data(
            name='linear1_bias', shape=[dim_feedforward])
        linear2_weight = paddle.static.data(
            name='linear2_weight',
            shape=[dim_feedforward, d_model],
            dtype=dtype)
        linear2_bias = paddle.static.data(name='linear2_bias', shape=[d_model])
        ln1_scale = paddle.static.data(name='ln1_scale', shape=[d_model])
        ln1_bias = paddle.static.data(name='ln1_scale', shape=[d_model])
        ln2_scale = paddle.static.data(name='ln2_scale', shape=[d_model])
        ln2_bias = paddle.static.data(name='ln2_scale', shape=[d_model])

        fused_out = F.fused_ffn(
            x,
            linear1_weight,
            linear2_weight,
            None,
            None,
            linear1_bias,
            linear2_bias,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            0.0,
            0.0,
            act_method="relu",
            normalize_pre_or_post=False)

        ######base ffn######
        linear1_out = F.linear(x, linear1_weight, linear1_bias)
        act_out = F.relu(linear1_out)
        dropout1_out = F.dropout(x=act_out, p=0.0, training=False)
        linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
        dropout2_out = x + F.dropout(x=linear2_out, p=0.0, training=False)
        ln_out = F.layer_norm(
            dropout2_out,
            normalized_shape=list([d_model]),
            weight=ln2_scale,
            bias=ln2_bias)
        ######base ffn######

        exe = paddle.static.Executor(place)

        x_data = np.random.random(
            (batch_size, d_model, dim_feedforward)).astype(dtype)
        linear1_weight_data = np.random.random(
            (d_model, dim_feedforward)).astype(dtype)
        linear1_bias_data = np.zeros((dim_feedforward)).astype(dtype)
        linear2_weight_data = np.random.random(
            (dim_feedforward, d_model)).astype(dtype)
        linear2_bias_data = np.zeros((d_model)).astype(dtype)

        ln1_scale_data = np.ones((d_model)).astype(layer_norm_dtype)
        ln1_bias_data = np.zeros((d_model)).astype(layer_norm_dtype)
        ln2_scale_data = np.ones((d_model)).astype(layer_norm_dtype)
        ln2_bias_data = np.zeros((d_model)).astype(layer_norm_dtype)

        res_list = [fused_out, ln_out]
        real_res = []

        for res in res_list:
            fetch = exe.run(feed={
                'x': x_data,
                'linear1_weight': linear1_weight_data,
                'linear1_bias': linear1_bias_data,
                'linear2_weight': linear2_weight_data,
                'linear2_bias': linear2_bias_data,
                'ln1_scale': ln1_scale_data,
                'ln1_bias': ln1_bias_data,
                'ln2_scale': ln2_scale_data,
                'ln2_bias': ln2_bias_data
            },
                            fetch_list=[res])
            real_res.append(fetch)
        self.assertTrue(
            np.allclose(
                real_res[0], real_res[1], atol=1e-5),
            "two value is check diff")
        print("test static success")


if __name__ == "__main__":
    unittest.main()
