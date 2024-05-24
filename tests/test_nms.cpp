#include <gtest/gtest.h>

#include "../src/postprocess.cpp"
// #include "../src/utils.cpp"

AnchorMat create_sample_input() {
  AnchorMat anchor_vec;

  Anchor anchor_1{0.5, 1.2, 1.5, 0.5, 0.7};
  Anchor anchor_2{0.7, 1.3, 1.4, 0.6, 0.6};
  Anchor anchor_3{0.8, 1.4, 1.3, 0.7, 0.5};
  Anchor anchor_4{0.9, 1.5, 1.2, 0.8, 0.4};
  Anchor anchor_5{0.3, 1.6, 1.1, 0.9, 0.3};
  Anchor anchor_6{0.4, 1.7, 1.0, 1.0, 0.2};

  // , anchor_2, anchor_3, anchor_4, anchor_5, anchor_6;

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
  NMS test_obj;
  AnchorMat required_output = create_sample_output();
  AnchorMat output = test_obj.nms(test_input, 0.2, 0.2, false, 0.5);
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
