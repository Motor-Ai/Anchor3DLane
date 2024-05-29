#include "postprocess.cpp"

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

int main() { AnchorMat abc = create_sample_input(); }