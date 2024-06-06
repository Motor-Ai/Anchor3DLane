#include <algorithm>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using AnchorMat = std::vector<std::vector<float>>;
using Anchor = std::vector<float>;

float softmax(std::vector<float> row) {
  std::transform(row.begin(), row.end(), row.begin(),
                 [](float r) { return exp(r); });
  float exp_row_sum = std::accumulate(row.begin(), row.end(), 0.0);
  return row[0] / exp_row_sum;
}

std::vector<int> create_sequence_vector(int start, int end) {
  // could use directly in the function. It is one line
  //
  std::vector<int> result(end - start + 1);
  std::iota(result.begin(), result.end(), start);
  return result;
}

// template std::vector<int>
// get_items_from_indices(const std::vector<int> &data,
//                        const std::vector<int> &indices);
// template std::vector<float>
// get_items_from_indices(const std::vector<float> &data,
//                        const std::vector<int> &indices);
// template AnchorMat extract_columns<20>(const AnchorMat &arr,
//                                        const std::vector<int> &columns);
std::vector<std::vector<float>>
FilterRows(const std::vector<std::vector<float>> &data,
           const std::vector<int> &order) {
  // Create a vector to store the filtered data
  std::vector<int> order_copy = order;
  std::sort(order_copy.begin() + 1, order_copy.end());

  std::vector<std::vector<float>> other_data;

  // Iterate through the order vector
  for (int i = 1; i < order_copy.size(); ++i) {
    other_data.push_back(data[order_copy[i]]);
  }
  return other_data;
}

void get_new_order(std::vector<int> &order, std::vector<int> indices) {

  std::vector<int> new_order;
  for (auto i : order) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      /* indices contains i */
      new_order.emplace_back(i);
    } else
      continue;
  }
  order = new_order;
}

std::vector<float> distance_between_2_lanes(std::vector<float> x1,
                                            std::vector<float> x2,
                                            std::vector<float> z1,
                                            std::vector<float> z2) {
  std::vector<float> distance(x2.size());

  std::transform(x1.begin(), x1.end(), x2.begin(), x2.begin(),
                 std::minus<float>());
  std::transform(z1.begin(), z1.end(), z2.begin(), z2.begin(),
                 std::minus<float>());

  std::transform(z2.begin(), z2.end(), x2.begin(), distance.begin(),
                 [](float z, float x) { return sqrt(z * z + x * x); });
  return distance;
}

std::vector<float> test_scores(const AnchorMat &proposals_arr) {
  std::vector<float> scores;
  for (auto i : proposals_arr) {
    scores.emplace_back(i[0]);
  }
  return scores;
}

void PrintArr(std::array<int, 4431> arr) {
  for (auto i : arr) {
    std::cout << i << std::endl;
  }
}

void PrintMat(const AnchorMat &mat) {
  for (auto i : mat) {
    for (auto j : i) {
      std::cout << std::setw(3) << j << "  ";
    }
    std::cout << std::endl;
  }
}
