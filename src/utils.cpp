#include <algorithm>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <tuple>
#include <vector>

using AnchorMat = std::vector<std::vector<float>>;
using Anchor = std::vector<float>;

float softmax(std::vector<float> row) {
  std::transform(row.begin(), row.end(), row.begin(),
                 [](float r) { return exp(r); });
  float exp_row_sum = std::accumulate(row.begin(), row.end(), 0.0);
  return row[0] / exp_row_sum;
}

template <typename T>
std::vector<T> get_items_from_indices(std::vector<T> data,
                                      std::vector<int> indices) {
  std::vector<T> output;

  for (auto i : indices) {
    output.emplace_back(data[i]);
  }
  return output;
}
// AnchorMat convert_pylist_to_arr(py::list proposals) {
//   unsigned list_len = py::len(proposals);

//   AnchorMat proposal_arr;
//   for (int i = 0; i < list_len; ++i) {
//     Anchor temp;
//     int col = i % proposal_dim;
//     temp[col] = proposals[i].cast<float>();
//     if (col == proposal_dim - 1) {
//       proposal_arr.emplace_back(temp);
//     }
//   }

//   return proposal_arr;
// }

std::vector<int> create_sequence_vector(int start, int end) {
  // could use directly in the function. It is one line 
  // 
  std::vector<int> result(end - start + 1);
  std::iota(result.begin(), result.end(), start);
  return result;
}

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

template <typename T> std::vector<int> argsort(const std::vector<T> &v) {
  // Initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // Sort indices based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
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

std::vector<int> distance_greater_indices(const std::vector<float> &acc_dis,
                                          const float &threshold,
                                          const std::vector<int> &order) {

  std::vector<int> order_copy = order;
  std::sort(order_copy.begin() + 1, order_copy.end());
  order_copy.erase(order_copy.begin());

  std::vector<int> indices;
  for (int i = 0; i < acc_dis.size(); ++i) {
    if (acc_dis[i] > threshold) {
      indices.emplace_back(order_copy[i]);
    }
  }
  return indices;
}

std::vector<float> compute_score(const AnchorMat &proposals) {
  std::vector<float> scores;

  std::vector<int> indx = create_sequence_vector(65, 85);
  std::vector<std::vector<float>> scores_before_softmax =
      extract_columns<proposal_dim>(proposals, indx);

  for (auto row : scores_before_softmax) {
    scores.emplace_back(1 - softmax(row));
  }
  return scores;
}

std::vector<float>
compute_distance(const std::vector<std::vector<float>> &all_x,
                 const std::vector<std::vector<float>> &all_z,
                 const std::vector<float> &x1, const std::vector<float> &z1) {
  std::vector<float> distance_arr;

  for (int i = 0; i < all_x.size(); ++i) {
    std::vector<float> x2 = all_x[i];
    std::vector<float> z2 = all_z[i];
    std::vector<float> distance__(x2.size());

    std::transform(x1.begin(), x1.end(), x2.begin(), x2.begin(),
                   std::minus<float>());
    std::transform(z1.begin(), z1.end(), z2.begin(), z2.begin(),
                   std::minus<float>());

    std::transform(z2.begin(), z2.end(), x2.begin(), distance__.begin(),
                   [](float z, float x) { return sqrt(z * z + x * x); });
    float accumulated_dis =
        std::accumulate(distance__.begin(), distance__.end(), 0.0);

    accumulated_dis /= (float)predication_steps;
    distance_arr.emplace_back(accumulated_dis);
  }
  return distance_arr;
}

std::vector<float> test_scores(const AnchorMat &proposals_arr) {
  std::vector<float> scores;
  for (auto i : proposals_arr) {
    scores.emplace_back(i[0]);
  }
  return scores;
}

std::tuple<AnchorMat, std::vector<float>, std::vector<int>>
filter_proposals(const AnchorMat &proposals_arr, float conf_threshold) {
  std::vector<int> anchor_inds = create_sequence_vector(0, num_anchors - 1);
  AnchorMat proposals_after_thresholding;
  std::vector<float> scores_after_thresholding;
  std::vector<int> anchor_inds_after_thresholding;
  // std::vector<float> scores = compute_score(proposals_arr);
  std::vector<float> scores = test_scores(proposals_arr);

  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > conf_threshold) {
      proposals_after_thresholding.emplace_back(proposals_arr[i]);
      scores_after_thresholding.emplace_back(scores[i]);
      anchor_inds_after_thresholding.emplace_back(anchor_inds[i]);
    }
  }

  return std::make_tuple(proposals_after_thresholding,
                         scores_after_thresholding,
                         anchor_inds_after_thresholding);
}
