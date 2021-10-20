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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle.nn.layer.fused_transformer import FusedMultiHeadAttention
from paddle import tensor
from paddle.fluid import layers
from paddle.static import Program, program_guard
import unittest


def fc(x, weight):
    return np.matmul(x, weight)


def softmax(x):
    np.seterr(invalid='ignore')
    output = np.zeros(x.shape, dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_curr = x[i, j, k, :]
                e_x = np.exp(x_curr - np.amax(x_curr))
                output[i, j, k, :] = e_x / np.sum(e_x)
    return output


def batch_matmul(x, y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    retval = np.zeros(
        (x.shape[0], x.shape[1], x.shape[2], y.shape[3]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            retval[i, j, :, :] = np.matmul(x[i, j, :, :], y[i, j, :, :])
    return retval


def layer_norm(x, has_scale, has_bias, weight, bias, epsilon=1e-05):
    batch_size, src_len, d_model = x.shape
    x = x.reshape((batch_size * src_len, d_model))
    mu = np.mean(x, axis=1, keepdims=True)
    sigma_squar = np.sum(np.square(x - mu), axis=1) / d_model
    x1_up = (x - mu)
    x1_down_1 = sigma_squar + epsilon
    x1_down = np.sqrt(x1_down_1)
    x1_down = x1_down.reshape((x1_down.shape[0], 1))
    x1 = x1_up / x1_down
    x_scaled = x1
    if (has_scale):
        x_scaled = weight * x1
    x_scaled_bias = x_scaled
    if (has_bias):
        x_scaled_bias = x_scaled + bias
    x_scaled_bias = x_scaled_bias.reshape((batch_size, src_len, d_model))
    return x_scaled_bias

def compute_reference(pre_layer_norm, num_head, query, attn_mask, ln_scale, ln_bias,
                      ln_2_scale, ln_2_bias, weight, out_linear_bias):
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    embed_dim = query.shape[2]
    head_dim = embed_dim//num_head

    print(batch_size)
    print(seq_len)
    print(embed_dim)
    print(head_dim)
    
    #[1, embed_dim, embed_dim]
    q_weight = weight[0:1, ::]
    k_weight = weight[1:2, ::]
    v_weight = weight[2:3, ::]
    out_linear_weight = weight[3:4, ::]
    print(weight.shape)
    print(q_weight.shape)
    print(k_weight.shape)
    print(v_weight.shape)
    print(out_linear_weight.shape)

    q_weight = q_weight.reshape(embed_dim, num_head*head_dim)
    k_weight = k_weight.reshape(embed_dim, num_head*head_dim)
    v_weight = v_weight.reshape(embed_dim, num_head*head_dim)
    out_linear_weight = out_linear_weight.reshape(embed_dim, embed_dim)

    if (pre_layer_norm):
        ln_out = layer_norm(query, True, True, ln_scale, ln_bias)

    if (pre_layer_norm):
        ln_out = ln_out.reshape(batch_size * seq_len, embed_dim)
        print(ln_out.shape)
        print(q_weight.shape)
        q = fc(ln_out, q_weight)
        k = fc(ln_out, k_weight)
        v = fc(ln_out, v_weight)
        ln_out = ln_out.reshape(batch_size, seq_len, embed_dim)
    else:
        query = query.reshape(batch_size * seq_len, embed_dim)
        q = fc(query, q_weight)
        k = fc(query, k_weight)
        v = fc(query, v_weight)
        query = query.reshape(batch_size, seq_len, embed_dim)

    q = q.reshape(batch_size, seq_len, num_head, head_dim)
    k = k.reshape(batch_size, seq_len, num_head, head_dim)
    v = v.reshape(batch_size, seq_len, num_head, head_dim)

    # [batch_size, num_head, seq_len, head_dim]
    q = q.transpose((0, 2, 1, 3))
    k = k.transpose((0, 2, 1, 3))
    v = v.transpose((0, 2, 1, 3))

    k = k.transpose([0, 1, 3, 2])  #[batch_size, num_head, head_dim, seq_len]
    qkt = batch_matmul(q, k / np.sqrt(head_dim, dtype=np.float64))

    # if attn_mask is not None:
    #     if attn_mask.dtype.name == 'int64':
    #         attn_mask = (attn_mask.astype(qkt.dtype) - 1.0) * 1e9
    #     else:
    #         attn_mask = attn_mask.astype(qkt.dtype)
    #     qkt += attn_mask

    # softmax
    softmax_out = softmax(qkt)
    attn_heads = batch_matmul(softmax_out, v)

    attn_heads = attn_heads.transpose(
        (0, 2, 1, 3))  # [batch_size, seq_len, num_head, head_dim]

    # out_linear
    out_linear_input = attn_heads.reshape(batch_size, seq_len,
                                          num_head * head_dim)
    out_linear_out = fc(out_linear_input, out_linear_weight)

    # bias add, dropout, residual add, layer_norm.
    out_linear_bias_out = out_linear_out + out_linear_bias
    out_linear_bias_dropout_out = out_linear_bias_out
    out_linear_bias_dropout_residual_out = query + out_linear_bias_dropout_out
    out_linear_bias_dropout_residual_ln_out = layer_norm(
        out_linear_bias_dropout_residual_out, True, True, ln_2_scale, ln_2_bias)
    #return ln_out, out_linear_out, out_linear_bias_dropout_residual_ln_out
    return out_linear_bias_dropout_residual_ln_out


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "Paddle core is not compiled with CUDA")
class TestFusedAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.config()
        self.generate_input_data()

    def config(self):
        self.x_type = np.float32
        self.attn_mask_type = np.float64
        self.pre_layer_norm = True
        self.training = True
        self.need_weight = False

        self.batch_size = 3
        self.query_length = 2
        self.head_dim = 2
        self.num_heads = 2
        self.embed_dim = self.head_dim * self.num_heads

        self.dropout_prob = 0.0
        self.attn_dropout_prob = 0.0
        self.weight_attr = None
        self.bias_attr = None

        self.kdim, self.vdim = self.embed_dim, self.embed_dim
        self.key_length, self.value_length = self.query_length, self.query_length

    def generate_input_data(self):
        self.query = np.random.rand(self.batch_size, self.query_length,
                                    self.embed_dim).astype(self.x_type)
        self.attn_mask = np.ones(
            (self.batch_size, self.num_heads, self.query_length,
             self.key_length),
            dtype=self.attn_mask_type)
        if self.attn_mask_type == np.int64:
            self.attn_mask = np.tril(self.attn_mask)
        elif self.attn_mask_type == np.float64:
            self.attn_mask = (np.tril(self.attn_mask) - 1.0) * 1e9
        else:
            raise ValueError("'attn_mask_type' should be 'int64' or 'float64'.")
        self.key, self.value = self.query, self.query

        self.seq_len = np.full((self.batch_size, ), self.query_length, dtype=np.int32)
        # lo_win = np.zeros((max_seq_len, ), dtype=np.int32)
        # hi_win = np.full(
        #     (max_seq_len, ), max_seq_len, dtype=np.int32)  # set a large number
        self.attn_low_window = np.zeros((self.query_length, ), dtype=np.int32)
        self.attn_high_window = np.full((self.query_length, ), self.query_length, dtype=np.int32)

    def run_imperative(self):
        fused_attn = FusedMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)
        out = fused_attn(
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query),
            paddle.to_tensor(self.query), 
            paddle.to_tensor(self.attn_mask),
            paddle.to_tensor(self.seq_len), 
            self.attn_low_window, 
            self.attn_high_window,
            self.seq_len)
        ref_out = compute_reference(self.pre_layer_norm, self.num_heads, self.query,
                                    self.attn_mask,
                                    fused_attn.pre_ln_scale.numpy(),
                                    fused_attn.pre_ln_bias.numpy(),
                                    fused_attn.ln_scale.numpy(),
                                    fused_attn.ln_bias.numpy(),
                                    fused_attn.weight.numpy(),
                                    fused_attn.out_linear_bias.numpy())
        # np.testing.assert_allclose(ref_ln, ln_out, rtol=1e-5, atol=1e-5)
        # np.testing.assert_allclose(ref_out_linear, linear_out, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(ref_out, out, rtol=1e-5, atol=1e-3)

    def run_static(self):
        fused_attn = FusedMultiHeadAttention(
            self.embed_dim, self.num_heads, self.dropout_prob,
            self.attn_dropout_prob, self.kdim, self.vdim, self.pre_layer_norm,
            self.need_weight, self.weight_attr, self.bias_attr)

        x = paddle.static.data(
            name='X',
            shape=[self.batch_size, self.query_length, self.embed_dim],
            dtype=self.x_type)
        attn_mask = paddle.static.data(
            name='SrcMask',
            shape=[
                self.batch_size, self.num_heads, self.query_length,
                self.key_length
            ],
            dtype=self.attn_mask_type)
        seq_len = paddle.static.data(
            name='SeqLen',
            shape=[
                self.batch_size
            ],
            dtype=np.int32)
        attn_low_window = paddle.static.data(
            name='AttnLowWin',
            shape=[
                self.query_length
            ],
            dtype=np.int32)
        attn_high_window = paddle.static.data(
            name='AttnHighWin',
            shape=[
                self.query_length
            ],
            dtype=np.int32)
        seq_len_host = paddle.static.data(
            name='SeqLenHost',
            shape=[
                self.batch_size
            ],
            dtype=np.int32)

        final_out = fused_attn(x, x, x, attn_mask, seq_len, attn_low_window, attn_high_window, seq_len_host)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        final_out, weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias = exe.run(
            paddle.static.default_main_program(),
            feed={"X": self.query,
                  "SrcMask": self.attn_mask,
                  "SeqLen": self.seq_len,
                  "AttnLowWin": self.attn_low_window,
                  "AttnHighWin": self.attn_high_window,
                  "SeqLenHost": self.seq_len},
            fetch_list=[
                final_out, fused_attn.weight, fused_attn.out_linear_bias,
                fused_attn.pre_ln_scale, fused_attn.pre_ln_bias,
                fused_attn.ln_scale, fused_attn.ln_bias
            ])

        return final_out, weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(Program()):
            out, qkv_weight, qkv_bias, linear_weight, linear_bias, ln_scale, ln_bias, ln_2_scale, ln_2_bias = self.run_static(
            )
        ref_out = compute_reference(self.pre_layer_norm, self.num_heads, self.query,
                                    self.attn_mask, ln_scale, ln_bias,
                                    ln_2_scale, ln_2_bias, qkv_weight, qkv_bias,
                                    linear_weight, linear_bias)
        # np.testing.assert_allclose(ref_ln, ln_out, rtol=1e-5, atol=1e-5)
        # np.testing.assert_allclose(ref_linear_out, linear_out, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(ref_out, out, rtol=1e-5, atol=1e-3)

    def test_dynamic_api(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()


if __name__ == "__main__":
    unittest.main()