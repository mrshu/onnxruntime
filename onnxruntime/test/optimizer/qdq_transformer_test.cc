// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "graph_gen_helper.h"

namespace onnxruntime {
namespace test {

#ifndef DISABLE_CONTRIB_OPS

TEST(QDQTransformerTests, Conv) {
  auto test_case = [&](const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape) {
    auto build_test_case = [&](GraphBuilder& helper) {
      auto* input_arg = helper.MakeInput<float>(input_shape);
      auto* output_arg = helper.MakeOutput();
      auto* q_output = helper.MakeIntermediate();
      auto* dq_output = helper.MakeIntermediate();
      auto* dq_w_output = helper.MakeIntermediate();
      auto* conv_output = helper.MakeIntermediate();
      auto* weight = helper.MakeWeightsInitializer<uint8_t>(weights_shape, 0, 128);

      helper.AddQuantizeLinearNode(input_arg, .01f, 135, q_output);
      helper.AddDequantizeLinearNode(q_output, .01f, 135, dq_output);
      helper.AddDequantizeLinearNode(weight, .01f, 12, dq_w_output);
      helper.AddConvNode(dq_output, dq_w_output, conv_output);
      helper.AddQuantizeLinearNode(conv_output, .01f, 135, output_arg);
    };

    auto check_nhwc_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["QLinearConv"], 1);
    };

    TransformerTester(build_test_case, check_nhwc_graph, TransformerLevel::Level1, TransformerLevel::Level2);
  };

  // Test the basic case of a single 1D/2D/3D convolution.
  test_case({1, 12, 37}, {32, 12, 5});
  test_case({1, 23, 13, 13}, {30, 23, 3, 3});
  test_case({1, 22, 11, 13, 15}, {30, 22, 5, 3, 3});
}

#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
