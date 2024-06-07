#include "postprocess.hpp"
#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

/**
 * @brief This is a test function which makes sample input for testing.
 * We have the following information:
 * 1. Index 0: Score
 * 2. Index 1, 2: x-coordinates
 * 2. Index 3, 4: z-coordinates
 *
 * @return AnchorMat
 */

/**
 * @brief Tests NMS initialization with six anchors.
 */
TEST(TestSimpleInput, InitWithSixAnchors) {
  AnchorMat test_input = create_sample_input();
  NMS test_obj(test_input, 0.2, true);
  AnchorMat required_output = create_sample_output();
  AnchorMat output = test_obj.nms();
  std::cout << output.size() << std::endl;

  for (int i = 0; i < output.size(); ++i) {
    for (int j = 0; j < (sizeof(i) / sizeof(float)); ++j) {
      ASSERT_EQ(required_output[i][j], output[i][j]);
    }
  }
}

/**
 * @brief Tests the softmax function with a simple input.
 */
TEST(TestSoftmax, SampleSoftmaxTest) {
  std::vector<float> input_vec{0.0};
  float req_out = 1.0;
  float output = softmax(input_vec);
  ASSERT_EQ(req_out, output);
}

/**
 * @brief Tests the get_new_order function with a simple input.
 */
TEST(TestGetNewOrder, SimpleNewOrderTest) {
  std::vector<int> input_order{5, 4, 2, 1, 89};
  std::vector<int> indices{0, 1, 2, 3};
  std::vector<int> req_out{2, 1};

  get_new_order(input_order, indices);
  ASSERT_EQ(req_out, input_order);
}

/**
 * @brief Tests the distance computation between two lanes.
 */
TEST(TestDistances, SimpleDistanceTest) {
  AnchorMat test_input = create_sample_input();
  int x_coordinate_start_idx = 1;
  int z_coordinate_start_idx = 3;
  int prediction_steps = 2;
  constexpr int proposal_dim = 5;
  std::vector<int> x_indices = create_sequence_vector(
      x_coordinate_start_idx, x_coordinate_start_idx + prediction_steps - 1);
  std::vector<int> z_indices = create_sequence_vector(
      z_coordinate_start_idx, z_coordinate_start_idx + prediction_steps - 1);

  std::vector<std::vector<float>> all_x =
      extract_columns<proposal_dim>(test_input, x_indices);
  std::vector<std::vector<float>> all_z =
      extract_columns<proposal_dim>(test_input, z_indices);

  std::vector<float> expected_output = {0,        0.141421, 0.282843,
                                        0.424264, 0.565685, 0.707107};
  std::vector<float> accumulated_dis =
      compute_all_lane_distance(all_x, all_z, all_x[0], all_z[0]);
  for (int i = 0; i < accumulated_dis.size(); ++i) {
    ASSERT_TRUE(floatCompare(accumulated_dis[i], expected_output[i]));
  }
  // ASSERT_EQ(req_out, input_order);
}

/**
 * @brief Tests the filtering of proposals based on confidence threshold.
 */
TEST(TestConfFiltering, SimpleConfFilteringTest) {
  AnchorMat test_input = create_sample_input();
  AnchorMat expected_ouput = conf_testing();
  auto [proposals, scores, anchor_inds] =
      filter_proposals(test_input, 0.6, true);
  for (int i = 0; i < expected_ouput.size(); ++i) {
    for (int j = 0; j < expected_ouput[i].size(); ++j) {
      ASSERT_TRUE(floatCompare(expected_ouput[i][j], proposals[i][j]));
    }
  }
}

/**
 * @brief Tests the NMS3D function with a simple input.
 */
TEST(TestNMS3D, SimpleNMS3DTest) {
  AnchorMat test_input = create_sample_input();
  auto [proposals, scores, anchor_inds] =
      filter_proposals(test_input, 0.0, true);
  NMS test_obj(test_input, 0.2, true);
  std::vector<int> keep = test_obj.nms_3d(proposals, scores);
  std::vector<int> expected_out_indices{3, 1, 5};
  ASSERT_EQ(keep.size(), expected_out_indices.size());
  for (int i = 0; i < expected_out_indices.size(); ++i) {
    ASSERT_EQ(expected_out_indices[i], keep[i]);
  }
}

/**
 * @brief Tests filtering rows based on given indices.
 */
TEST(TestFilterRows, SimpleRowFilteringTest) {
  std::vector<std::vector<float>> input_vec, expected_ouput;

  std::vector<float> x1{1.2, 1.5};
  std::vector<float> x2{1.4, 1.3};
  std::vector<float> x3{1.7, 1.0};

  input_vec.emplace_back(x1);
  input_vec.emplace_back(x2);
  input_vec.emplace_back(x3);

  expected_ouput.emplace_back(x1);
  expected_ouput.emplace_back(x2);
  // There is an implementation detail, we skip the first elem in the
  // index_to_get in the implementation.
  // And we sort the order as well.

  std::vector<int> index_to_get{-1, 1, 0};
  std::vector<std::vector<float>> output = FilterRows(input_vec, index_to_get);
  ASSERT_EQ(expected_ouput.size(), output.size());

  for (int i = 0; i < expected_ouput.size(); ++i) {
    for (int j = 0; j < expected_ouput[i].size(); ++j) {
      ASSERT_TRUE(floatCompare(expected_ouput[i][j], output[i][j]));
    }
  }
}

/**
 * @brief Tests the full functionality of NMS using file inputs.
 */
TEST(TestFullFunctional, FullFunctionalityTest) {

  AnchorMat input_anchors = read_anchors_from_file("../proposals.txt");
  AnchorMat required_output = read_anchors_from_file("../output.txt");

  NMS test_obj(input_anchors, 2.0, false);
  AnchorMat output = test_obj.nms();
  ASSERT_EQ(required_output.size(), output.size());
  for (int i = 0; i < required_output.size(); ++i) {
    for (int j = 0; j < (sizeof(i) / sizeof(float)); ++j) {
      ASSERT_EQ(required_output[i][j], output[i][j]);
    }
  }
}
