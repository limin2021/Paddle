// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <map>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

// Class to store the keys for compiling CINN.
//
// CINN cannot handle changable shape now, so CinnRunner keeps a cache mapping
// from CinnCacheKey to CinnCompiledObject.
//
// The CinnCacheKey contains a graph serialized string and the feeded tensor
// shapes.
class CinnCacheKey {
 public:
  CinnCacheKey(const ir::Graph& graph,
               const std::map<std::string, const LoDTensor*>& feed_tensors);
  CinnCacheKey(const ir::Graph& graph,
               const std::map<std::string, DDim>& feed_shapes);

  ~CinnCacheKey() {}

  void SetKey(const ir::Graph& graph,
              const std::map<std::string, const LoDTensor*>& feed_tensors);
  void SetKey(const ir::Graph& graph,
              const std::map<std::string, DDim>& feed_shapes);

  bool operator==(const CinnCacheKey& other) const;
  bool operator!=(const CinnCacheKey& other) const;

  struct Hash {
    static size_t hash_combine(size_t seed, size_t value);
    size_t operator()(const CinnCacheKey& key) const;
  };

 private:
  std::string graph_serialize_str_;
  std::map<std::string, DDim> feed_shapes_;
};

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
