#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

extern constexpr int num_anchors = 4431;
extern constexpr int proposal_dim = 86;
extern constexpr int dim_anchor = 65;
constexpr int x_coordinate_start_idx = 5;
constexpr int z_coordinate_start_idx = 25;
constexpr int predication_steps = 20;

// extern constexpr int x_coordinate_start_idx = 1;
// extern constexpr int z_coordinate_start_idx = 3;
// extern constexpr int predication_steps = 2;

using AnchorMat = std::vector<std::array<float, proposal_dim>>;
using Anchor = std::array<float, proposal_dim>;

template <typename T> void PrintVecVec(const std::vector<std::vector<T>> &mat) {
  for (auto i : mat) {
    for (auto j : i) {
      std::cout << std::setw(3) << j << "  ";
    }
    std::cout << std::endl;
  }
}
void PrintArr(std::array<int, num_anchors> arr) {
  for (auto i : arr) {
    std::cout << i << std::endl;
  }
}

template <typename T> void PrintVec(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << "  ";
  }
  std::cout << std::endl;
}

void PrintMat(const AnchorMat &mat) {
  for (auto i : mat) {
    for (auto j : i) {
      std::cout << std::setw(3) << j << "  ";
    }
    std::cout << std::endl;
  }
}
