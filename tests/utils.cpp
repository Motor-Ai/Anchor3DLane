#include "utils.hpp"
#include "postprocess.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief This is a test function which makes sample input for testing.
 We have the following information:
 1. Index 0: Score
 2. Index 1, 2: x-coordinates
 2. Index 3, 4: z-coordinates
 *
 * @return AnchorMat
 */

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

bool floatCompare(float f1, float f2) {
  static constexpr auto epsilon = 1.0e-01f;
  if (fabs(f1 - f2) <= epsilon)
    return true;
  return false;
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
  return out_vec;
}
