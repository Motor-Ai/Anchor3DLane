#include "../src/postprocess.cpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
AnchorMat create_sample_input() {
  AnchorMat anchor_vec;

  Anchor anchor_1{0.5, 1.2, 1.5, 0.5, 0.7};
  Anchor anchor_2{0.7, 1.3, 1.4, 0.6, 0.6};
  Anchor anchor_3{0.8, 1.4, 1.3, 0.7, 0.5};
  Anchor anchor_4{0.9, 1.5, 1.2, 0.8, 0.4};
  Anchor anchor_5{0.3, 1.6, 1.1, 0.9, 0.3};
  Anchor anchor_6{0.4, 1.7, 1.0, 1.0, 0.2};

  anchor_vec.emplace_back(anchor_1);
  anchor_vec.emplace_back(anchor_2);
  anchor_vec.emplace_back(anchor_3);
  anchor_vec.emplace_back(anchor_4);
  anchor_vec.emplace_back(anchor_5);
  anchor_vec.emplace_back(anchor_6);

  return anchor_vec;
}

AnchorMat create_sample_output() {
  AnchorMat anchor_vec;

  Anchor anchor_4{0.9, 1.5, 1.2, 0.8, 0.4};
  Anchor anchor_2{0.7, 1.3, 1.4, 0.6, 0.6};
  Anchor anchor_6{0.4, 1.7, 1.0, 1.0, 0.2};

  anchor_vec.emplace_back(anchor_4);
  anchor_vec.emplace_back(anchor_2);
  anchor_vec.emplace_back(anchor_6);

  return anchor_vec;
}

TEST(TestSimpleInput, InitWithSixAnchors) {
  AnchorMat test_input = create_sample_input();
  NMS test_obj(test_input, 0.2, true);
  AnchorMat required_output = create_sample_output();
  AnchorMat output = test_obj.nms();

  for (int i = 0; i < required_output.size(); ++i) {
    for (int j = 0; j < (sizeof(i) / sizeof(float)); ++j) {
      ASSERT_EQ(required_output[i][j], output[i][j]);
    }
  }
}

TEST(TestSoftmax, Hello) {
  std::vector<float> input_vec{0.0};
  float req_out = 1.0;
  float output = softmax(input_vec);
  ASSERT_EQ(req_out, output);
}

TEST(TestGetNewOrder, Hello) {
  std::vector<int> input_order{5, 4, 2, 1, 89};
  std::vector<int> indices{0, 1, 2, 3};
  std::vector<int> req_out{2, 1};

  get_new_order(input_order, indices);
  ASSERT_EQ(req_out, input_order);
}

bool floatCompare(float f1, float f2) {
  static constexpr auto epsilon = 1.0e-01f;
  if (fabs(f1 - f2) <= epsilon)
    return true;
  return false;
}

TEST(TestDistances, Hello) {
  AnchorMat test_input = create_sample_input();

  std::vector<int> x_indices = create_sequence_vector(
      x_coordinate_start_idx, x_coordinate_start_idx + predication_steps - 1);
  std::vector<int> z_indices = create_sequence_vector(
      z_coordinate_start_idx, z_coordinate_start_idx + predication_steps - 1);

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

AnchorMat conf_testing() {
  AnchorMat anchor_vec;

  Anchor anchor_2{0.7, 1.3, 1.4, 0.6, 0.6};
  Anchor anchor_3{0.8, 1.4, 1.3, 0.7, 0.5};
  Anchor anchor_4{0.9, 1.5, 1.2, 0.8, 0.4};

  anchor_vec.emplace_back(anchor_2);
  anchor_vec.emplace_back(anchor_3);
  anchor_vec.emplace_back(anchor_4);

  return anchor_vec;
}

TEST(TestConfFiltering, Hello) {
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

TEST(TestNMS3D, Hello) {
  AnchorMat test_input = create_sample_input();
  auto [proposals, scores, anchor_inds] =
      filter_proposals(test_input, 0.0, true);
  NMS test_obj(test_input, 0.2, true);
  std::vector<int> keep = test_obj.nms_3d(proposals, scores, 0.2);
  std::vector<int> expected_out_indices{3, 1, 5};
  ASSERT_EQ(keep.size(), expected_out_indices.size());
  for (int i = 0; i < expected_out_indices.size(); ++i) {
    ASSERT_EQ(expected_out_indices[i], keep[i]);
  }
}

TEST(TestFilterRows, Hello) {
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

std::vector<std::vector<float>> read_anchors_from_file(std::string path) {

  std::vector<std::vector<float>> out_vec;
  std::ifstream file(path);
  std::string line;
  std::vector<float> anchor;
  while (std::getline(file, line)) {

    float num_float = std::stof(line);
    anchor.emplace_back(num_float);
    if (anchor.size() == 86) {
      out_vec.emplace_back(anchor);
      anchor.clear();
    }
  }
  std::cout << out_vec.size() << std::endl;
  return out_vec;
}

TEST(TestFullFunctional, Hello) {

  AnchorMat input_anchors = read_anchors_from_file(
      "/home/sandhu/project/LaneSeg/Anchor3DLane/proposals.txt");
  AnchorMat required_output = read_anchors_from_file(
      "/home/sandhu/project/LaneSeg/Anchor3DLane/output.txt");
  std::cout << required_output.size();
  std::cout << input_anchors.size();

  NMS test_obj(input_anchors, 2.0, false);
  AnchorMat output = test_obj.nms();
  std::cout << "output" << std::endl;
  std::cout << required_output.size();
  for (int i = 0; i < required_output.size(); ++i) {
    for (int j = 0; j < (sizeof(i) / sizeof(float)); ++j) {
      ASSERT_EQ(required_output[i][j], output[i][j]);
    }
  }
}
