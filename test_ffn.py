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
import time
import datetime

import paddle
import paddle.nn as nn
from paddle.framework import ParamAttr
from paddle.nn.layer import fused_transformer
import paddle.nn.functional as F
from paddle.nn.layer.norm import LayerNorm
from paddle.nn.layer.common import Linear, Dropout

place = paddle.CUDAPlace(0)

batch_size = 64
query_length = 128
d_model = 1024
dim_feedforward = 4096
normalize_before = False
act_method = "relu"
dropout_prob1 = 0.5
dropout_prob2 = 0.5

weight_attr = None
bias_attr = None

weight_attrs = fused_transformer._convert_param_attr_to_list(weight_attr, 2)
bias_attrs = fused_transformer._convert_param_attr_to_list(bias_attr, 2)

dtype = "float32"

rows = batch_size * query_length

linear1_weight_data = np.random.random((d_model, dim_feedforward)).astype(dtype)
linear1_bias_data = np.zeros((dim_feedforward)).astype(
    dtype)  #np.random.random((dim_feedforward)).astype(dtype)
linear2_weight_data = np.random.random((dim_feedforward, d_model)).astype(dtype)
linear2_bias_data = np.zeros(
    (d_model)).astype(dtype)  #np.random.random((d_model)).astype(dtype)

ln1_scale_data = np.ones(
    (d_model)).astype("float32")  #np.random.random((d_model)).astype("float32")
ln1_bias_data = np.zeros(
    (d_model)).astype("float32")  #np.random.random((d_model)).astype("float32")
ln2_scale_data = np.ones(
    (d_model)).astype("float32")  #np.random.random((d_model)).astype("float32")
ln2_bias_data = np.zeros(
    (d_model)).astype("float32")  #np.random.random((d_model)).astype("float32")

dropout1 = Dropout(dropout_prob1, mode="upscale_in_train")
dropout2 = Dropout(dropout_prob1, mode="upscale_in_train")
activation = getattr(F, act_method)

src_data = np.random.random((batch_size, query_length, d_model)).astype(dtype)
dout_data = np.random.random((batch_size, query_length, d_model)).astype(dtype)

iters = 100


def print_time(desc, times):
    times *= 1000
    print(desc, " total time = ", times, "ms, avg time = ", times / iters, "ms")
    print()


def Base():
    total_time = 0
    for i in range(0, iters):
        src = paddle.to_tensor(src_data, stop_gradient=False)
        residual = paddle.to_tensor(src_data, stop_gradient=False)
        dout = paddle.to_tensor(dout_data, stop_gradient=False)
        linear1_weight = paddle.to_tensor(
            linear1_weight_data, stop_gradient=False)
        linear1_bias = paddle.to_tensor(linear1_bias_data, stop_gradient=False)
        linear2_weight = paddle.to_tensor(
            linear2_weight_data, stop_gradient=False)
        linear2_bias = paddle.to_tensor(linear2_bias_data, stop_gradient=False)

        ln1_scale = paddle.to_tensor(ln1_scale_data, stop_gradient=False)
        ln1_bias = paddle.to_tensor(ln1_bias_data, stop_gradient=False)
        ln2_scale = paddle.to_tensor(ln2_scale_data, stop_gradient=False)
        ln2_bias = paddle.to_tensor(ln2_bias_data, stop_gradient=False)

        paddle.device.cuda.synchronize(place)
        t0 = time.time()

        if normalize_before:
            ln1_out = F.layer_norm(src, list([d_model]), ln1_scale, ln1_bias)
            linear1_out = F.linear(ln1_out, linear1_weight, linear1_bias)
            act_out = activation(linear1_out)
            dropout1_out = dropout1(act_out)
            linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
            dropout2_out = residual + dropout2(linear2_out)
        else:
            linear1_out = F.linear(src, linear1_weight, linear1_bias)
            act_out = activation(linear1_out)
            dropout1_out = dropout1(act_out)
            linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
            dropout2_out = residual + dropout2(linear2_out)
            dropout2_out = F.layer_norm(dropout2_out,
                                        list([d_model]), ln2_scale, ln2_bias)
        paddle.autograd.backward([dropout2_out], [dout], True)

        paddle.device.cuda.synchronize(place)
        t1 = time.time()
        total_time += (t1 - t0)

    print_time("base : ", total_time)


def FusedFFN():

    total_time = 0
    for i in range(0, iters):
        x = paddle.to_tensor(src_data, stop_gradient=False)
        dout = paddle.to_tensor(dout_data, stop_gradient=False)
        linear1_weight = paddle.to_tensor(
            linear1_weight_data, stop_gradient=False)
        linear1_bias = paddle.to_tensor(linear1_bias_data, stop_gradient=False)
        linear2_weight = paddle.to_tensor(
            linear2_weight_data, stop_gradient=False)
        linear2_bias = paddle.to_tensor(linear2_bias_data, stop_gradient=False)

        ln1_scale = paddle.to_tensor(ln1_scale_data, stop_gradient=False)
        ln1_bias = paddle.to_tensor(ln1_bias_data, stop_gradient=False)
        ln2_scale = paddle.to_tensor(ln2_scale_data, stop_gradient=False)
        ln2_bias = paddle.to_tensor(ln2_bias_data, stop_gradient=False)
        seed1_data = None
        seed2_data = None

        paddle.device.cuda.synchronize(place)
        t0 = time.time()
        out = F.fused_ffn(
            x,
            linear1_weight,
            linear2_weight,
            linear1_bias,
            linear2_bias,
            ln1_scale,
            ln1_bias,
            ln2_scale,
            ln2_bias,
            dropout_prob1=dropout_prob1,
            dropout_prob2=dropout_prob1,
            act_method=act_method,
            normalize_pre_or_post=normalize_before)

        paddle.autograd.backward([out], [dout], True)

        paddle.device.cuda.synchronize(place)
        t1 = time.time()
        total_time += (t1 - t0)
    print_time("fused: ", total_time)


def test_static_base():
    paddle.enable_static()
    layer_norm_dtype = "float32"
    x = paddle.static.data(
        name='x', shape=[batch_size, query_length, d_model], dtype=dtype)
    linear1_weight = paddle.static.data(
        name='linear1_weight', shape=[d_model, dim_feedforward], dtype=dtype)
    linear1_bias = paddle.static.data(
        name='linear1_bias', shape=[dim_feedforward], dtype=dtype)
    linear2_weight = paddle.static.data(
        name='linear2_weight', shape=[dim_feedforward, d_model], dtype=dtype)
    linear2_bias = paddle.static.data(
        name='linear2_bias', shape=[d_model], dtype=dtype)
    ln1_scale = paddle.static.data(
        name='ln1_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln1_bias = paddle.static.data(
        name='ln1_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln2_scale = paddle.static.data(
        name='ln2_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln2_bias = paddle.static.data(
        name='ln2_scale', shape=[d_model], dtype=layer_norm_dtype)

    linear1_out = F.linear(x, linear1_weight, linear1_bias)
    act_out = F.relu(linear1_out)
    dropout1_out = F.dropout(x=act_out, p=0.5, training=True)
    linear2_out = F.linear(dropout1_out, linear2_weight, linear2_bias)
    dropout2_out = x + F.dropout(x=linear2_out, p=0.5, training=True)
    ln_out = F.layer_norm(
        dropout2_out,
        normalized_shape=list([d_model]),
        weight=ln2_scale,
        bias=ln2_bias)

    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    fetch = exe.run(feed={
        'x': src_data,
        'linear1_weight': linear1_weight_data,
        'linear1_bias': linear1_bias_data,
        'linear2_weight': linear2_weight_data,
        'linear2_bias': linear2_bias_data,
        'ln1_scale': ln1_scale_data,
        'ln1_bias': ln1_bias_data,
        'ln2_scale': ln2_scale_data,
        'ln2_bias': ln2_bias_data
    },
                    fetch_list=[ln_out])
    paddle.device.cuda.synchronize(place)

    t0 = time.time()
    for i in range(0, iters):
        fetch = exe.run(feed={
            'x': src_data,
            'linear1_weight': linear1_weight_data,
            'linear1_bias': linear1_bias_data,
            'linear2_weight': linear2_weight_data,
            'linear2_bias': linear2_bias_data,
            'ln1_scale': ln1_scale_data,
            'ln1_bias': ln1_bias_data,
            'ln2_scale': ln2_scale_data,
            'ln2_bias': ln2_bias_data
        },
                        fetch_list=[ln_out])
    paddle.device.cuda.synchronize(place)
    t1 = time.time()
    print_time("base static: ", (t1 - t0))


def test_static_fused():
    paddle.enable_static()
    layer_norm_dtype = "float32"

    x = paddle.static.data(
        name='x', shape=[batch_size, query_length, d_model], dtype=dtype)
    linear1_weight = paddle.static.data(
        name='linear1_weight', shape=[d_model, dim_feedforward], dtype=dtype)
    linear1_bias = paddle.static.data(
        name='linear1_bias', shape=[dim_feedforward], dtype=dtype)
    linear2_weight = paddle.static.data(
        name='linear2_weight', shape=[dim_feedforward, d_model], dtype=dtype)
    linear2_bias = paddle.static.data(
        name='linear2_bias', shape=[d_model], dtype=dtype)
    ln1_scale = paddle.static.data(
        name='ln1_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln1_bias = paddle.static.data(
        name='ln1_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln2_scale = paddle.static.data(
        name='ln2_scale', shape=[d_model], dtype=layer_norm_dtype)
    ln2_bias = paddle.static.data(
        name='ln2_scale', shape=[d_model], dtype=layer_norm_dtype)

    fused_out = F.fused_ffn(
        x,
        linear1_weight,
        linear2_weight,
        linear1_bias,
        linear2_bias,
        ln1_scale,
        ln1_bias,
        ln2_scale,
        ln2_bias,
        dropout_prob1,
        dropout_prob1,
        act_method="relu",
        normalize_pre_or_post=False)

    exe = paddle.static.Executor(paddle.CUDAPlace(0))

    exe.run(feed={
        'x': src_data,
        'linear1_weight': linear1_weight_data,
        'linear1_bias': linear1_bias_data,
        'linear2_weight': linear2_weight_data,
        'linear2_bias': linear2_bias_data,
        'ln1_scale': ln1_scale_data,
        'ln1_bias': ln1_bias_data,
        'ln2_scale': ln2_scale_data,
        'ln2_bias': ln2_bias_data
    },
            fetch_list=[fused_out])
    paddle.device.cuda.synchronize(place)

    t0 = time.time()
    for i in range(0, iters):
        exe.run(feed={
            'x': src_data,
            'linear1_weight': linear1_weight_data,
            'linear1_bias': linear1_bias_data,
            'linear2_weight': linear2_weight_data,
            'linear2_bias': linear2_bias_data,
            'ln1_scale': ln1_scale_data,
            'ln1_bias': ln1_bias_data,
            'ln2_scale': ln2_scale_data,
            'ln2_bias': ln2_bias_data
        },
                fetch_list=[fused_out])

    paddle.device.cuda.synchronize(place)
    t1 = time.time()
    print_time("fused static: ", (t1 - t0))


#Base()
#Base()
FusedFFN()
#test_static_base()
#test_static_fused()
