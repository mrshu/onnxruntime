// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <random>
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

struct GraphBuilder {
  GraphBuilder(Graph& graph) : graph_(graph) {
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape, const ONNX_NAMESPACE::TypeProto& type_proto) {
    OrtValue input_value;
    CreateMLValue<T>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), shape,
                     FillRandomData<T>(shape, 0, 31), &input_value);
    std::string name = graph_.GenerateNodeArgName("input");
    feeds_.insert(std::make_pair(name, input_value));

    return &graph_.GetOrCreateNodeArg(name, &type_proto);
  }

  template <typename T>
  NodeArg* MakeInput(const std::vector<int64_t>& shape) {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(utils::ToTensorProtoElementType<T>());

    for (auto& dim : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }

    return MakeInput<T>(shape, type_proto);
  }

  NodeArg* MakeOutput() {
    std::string name = graph_.GenerateNodeArgName("output");
    output_names_.push_back(name);
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  NodeArg* MakeIntermediate() {
    std::string name = graph_.GenerateNodeArgName("node");
    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, const std::vector<T>& data) {
    std::string name = graph_.GenerateNodeArgName("constant");
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_name(name);
    tensor_proto.set_data_type(utils::ToTensorProtoElementType<T>());
    tensor_proto.set_raw_data(data.data(), data.size() * sizeof(T));

    for (auto& dim : shape) {
      tensor_proto.add_dims(dim);
    }

    graph_.AddInitializedTensor(tensor_proto);

    return &graph_.GetOrCreateNodeArg(name, nullptr);
  }

  template <typename T>
  NodeArg* MakeInitializer(const std::vector<int64_t>& shape, int32_t min_value, int32_t max_value) {
    return MakeInitializer<T>(shape, FillRandomData<T>(shape, min_value, max_value));
  }

  template <typename T>
  NodeArg* MakeScalarInitializer(T data) {
    return MakeInitializer({}, std::vector<T>{data});
  }

  template <typename T>
  NodeArg* Make1DInitializer(const std::vector<T>& data) {
    return MakeInitializer({static_cast<int64_t>(data.size())}, data);
  }

  template <typename T>
  NodeArg* MakeWeightsInitializer(const std::vector<int64_t>& shape, T min_value, T max_value) {
    return MakeInitializer<T>(shape, min_value, max_value);
  }

  Node& AddNode(const std::string& op_type,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain = "") {
    return graph_.AddNode(graph_.GenerateNodeName("node"),
                          op_type,
                          "description",
                          input_args,
                          output_args,
                          nullptr,
                          domain);
  }

  Node& AddConvNode(NodeArg* input_arg,
                    NodeArg* weights_arg,
                    NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(weights_arg);

    return AddNode("Conv", input_args, {output_arg});
  }

  Node& AddQuantizeLinearNode(NodeArg* input_arg,
                              float input_scale,
                              uint8_t input_zero_point,
                              NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));

    return AddNode("QuantizeLinear", input_args, {output_arg});
  }

  Node& AddDequantizeLinearNode(NodeArg* input_arg,
                                float input_scale,
                                uint8_t input_zero_point,
                                NodeArg* output_arg) {
    std::vector<NodeArg*> input_args;
    input_args.push_back(input_arg);
    input_args.push_back(MakeScalarInitializer<float>(input_scale));
    input_args.push_back(MakeScalarInitializer<uint8_t>(input_zero_point));

    return AddNode("DequantizeLinear", input_args, {output_arg});
  }

  template <typename T>
  std::vector<T> FillRandomData(size_t count, int32_t min_value, int32_t max_value) {
    std::vector<T> random_data;
    random_data.resize(count);
    std::uniform_int_distribution<int32_t> distribution(min_value, max_value);
    for (size_t n = 0; n < count; n++) {
      random_data[n] = static_cast<T>(distribution(generator_));
    }
    return random_data;
  }

  template <typename T>
  std::vector<T> FillRandomData(const std::vector<int64_t>& shape, int32_t min_value, int32_t max_value) {
    int64_t num_elements = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>{});
    return FillRandomData<T>(static_cast<size_t>(num_elements), min_value, max_value);
  }

  Graph& graph_;
  NameMLValMap feeds_;
  std::vector<std::string> output_names_;
  std::default_random_engine generator_{2345};
};

void TransformerTester(const std::function<void(GraphBuilder& helper)>& build_test_case,
                       const std::function<void(InferenceSessionWrapper& session)>& check_nhwc_graph,
                       TransformerLevel baseline_level,
                       TransformerLevel target_level,
                       int opset_version = 12);
}  // namespace test
}  // namespace onnxruntime
