#ifndef UTILS_HPP
#define UTILS_HPP

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

/**
 * @brief Computes the softmax of a vector.
 *
 * @param row A vector of floats.
 * @return The softmax-transformed vector.
 */
float softmax(std::vector<float> row);

/**
 * @brief Extracts items from a vector based on a list of indices.
 *
 * @tparam T The type of elements in the vector.
 * @param data The vector to extract items from.
 * @param indices The indices of the items to extract.
 * @return A vector containing the extracted items.
 */
template <typename T>
std::vector<T> get_items_from_indices(std::vector<T> data,
                                      std::vector<int> indices) {
  std::vector<T> output;

  for (auto i : indices) {
    output.emplace_back(data[i]);
  }
  return output;
}
/**
 * @brief Creates a sequence vector from start to end.
 *
 * @param start The starting value of the sequence.
 * @param end The ending value of the sequence.
 * @return A vector containing the sequence from start to end.
 */
std::vector<int> create_sequence_vector(int start, int end);
/**
 * @brief Extracts specified columns from a 2D matrix.
 *
 * @tparam Cols The number of columns to extract.
 * @param arr The input matrix.
 * @param columns The indices of the columns to extract.
 * @return A matrix containing the extracted columns.
 */
template <int Cols>
AnchorMat extract_columns(const AnchorMat &arr,
                          const std::vector<int> &columns) {
  AnchorMat extracted_columns(arr.size(), std::vector<float>(columns.size()));

  for (size_t i = 0; i < arr.size(); ++i) {
    for (size_t j = 0; j < columns.size(); ++j) {
      extracted_columns[i][j] = arr[i][columns[j]];
    }
  }
  return extracted_columns;
}
/**
 * @brief Filters rows of a matrix based on an order vector.
 *
 * @param data The input matrix.
 * @param order The order vector.
 * @return A matrix containing the filtered rows.
 */
std::vector<std::vector<float>>
FilterRows(const std::vector<std::vector<float>> &data,
           const std::vector<int> &order);
/**
 * @brief Sorts a vector and returns the indices of the sorted elements.
 *
 * @tparam T The type of elements in the vector.
 * @param v The vector to sort.
 * @return A vector of indices of the sorted elements.
 */
template <typename T> std::vector<int> argsort(const std::vector<T> &v) {
  // Initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // Sort indices based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}
/**
 * @brief Updates the order vector based on new indices.
 *
 * @param order The order vector to update.
 * @param indices The new indices.
 */
void get_new_order(std::vector<int> &order, std::vector<int> indices);
/**
 * @brief Computes the distance between two lanes.
 *
 * @param x1 The x-coordinates of the first lane.
 * @param x2 The x-coordinates of the second lane.
 * @param z1 The z-coordinates of the first lane.
 * @param z2 The z-coordinates of the second lane.
 * @return A vector of distances between the two lanes.
 */
std::vector<float> distance_between_2_lanes(std::vector<float> x1,
                                            std::vector<float> x2,
                                            std::vector<float> z1,
                                            std::vector<float> z2);
/**
 * @brief Computes test scores for the proposals.
 *
 * @param proposals_arr The array of proposals.
 * @return A vector of test scores.
 */
std::vector<float> test_scores(const AnchorMat &proposals_arr);
/**
 * @brief Prints a vector to the standard output.
 *
 * @tparam T The type of elements in the vector.
 * @param vec The vector to print.
 */
template <typename T> void PrintVec(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << "  ";
  }
  std::cout << std::endl;
}
/**
 * @brief Prints a 2D vector (matrix) to the standard output.
 *
 * @tparam T The type of elements in the matrix.
 * @param mat The matrix to print.
 */
template <typename T> void PrintVecVec(const std::vector<std::vector<T>> &mat) {
  for (auto i : mat) {
    for (auto j : i) {
      std::cout << std::setw(3) << j << "  ";
    }
    std::cout << std::endl;
  }
}

/**
 * @brief Prints an array to the standard output.
 *
 * @param arr The array to print.
 */

void PrintArr(std::array<int, 4431> arr);

/**
 * @brief Prints a matrix to the standard output.
 *
 * @param mat The matrix to print.
 */
void PrintMat(const AnchorMat &mat);
#endif // UTILS_HPP
