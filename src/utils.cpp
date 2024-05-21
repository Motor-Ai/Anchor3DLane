// #include </home/sandhu/project/LaneSeg/Anchor3DLane/src/prints.cpp>
#include <iomanip>
#include <iostream>
#include <tuple>

constexpr int num_anchors = 4431;
constexpr int proposal_dim = 86;
constexpr int dim_anchor = 65;
constexpr int x_coordinate_start_idx = 5;
constexpr int z_coordinate_start_idx = 25;
constexpr int predication_steps = 20;

// constexpr int x_coordinate_start_idx = 1;
// constexpr int z_coordinate_start_idx = 3;
// constexpr int predication_steps = 2;

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
template <typename T> void PrintArr(std::array<T, proposal_dim> arr) {
  for (auto i : arr) {
    std::cout << i << std::endl;
  }
}

AnchorMat convert_pylist_to_arr(py::list proposals) {
  unsigned list_len = py::len(proposals);

  AnchorMat proposal_arr;
  for (int i = 0; i < list_len; ++i) {
    Anchor temp;
    int col = i % proposal_dim;
    temp[col] = proposals[i].cast<float>();
    if (col == proposal_dim - 1) {
      proposal_arr.emplace_back(temp);
    }
  }
  PrintArr(proposal_arr[0]);
  return proposal_arr;
}

template <typename T> void PrintVec(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << "  ";
  }
  std::cout << std::endl;
}

float softmax(std::vector<float> row) {
  std::transform(row.begin(), row.end(), row.begin(),
                 [](float r) { return exp(r); });
  float exp_row_sum = std::accumulate(row.begin(), row.end(), 0.0);
  return row[0] / exp_row_sum;
}
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

template <typename T>
std::vector<T> get_items_from_indices(std::vector<T> data,
                                      std::vector<int> indices) {
  std::vector<T> output;

  for (auto i : indices) {
    output.emplace_back(data[i]);
  }
  return output;
}

std::vector<int> create_sequence_vector(int start, int end) {
  std::vector<int> result(end - start + 1);
  std::iota(result.begin(), result.end(), start);
  return result;
}

template <int Cols>
std::vector<std::vector<float>>
extract_columns(const std::vector<std::array<float, Cols>> &arr,
                const std::vector<int> &columns) {
  std::vector<std::vector<float>> extracted_columns(
      arr.size(), std::vector<float>(columns.size()));

  for (size_t i = 0; i < arr.size(); ++i) {
    for (size_t j = 0; j < columns.size(); ++j) {
      extracted_columns[i][j] = arr[i][columns[j]];
    }
  }

  return extracted_columns;
}

void FilterRows(std::vector<std::vector<float>> &data,
                const std::vector<int> &order) {
  // Create a vector to store the filtered data
  std::vector<int> order_copy = order;

  std::vector<std::vector<float>> other_data;
  std::sort(order_copy.begin() + 1, order_copy.end());

  // Iterate through the order vector
  for (int i = 1; i < order_copy.size(); ++i) {
    other_data.push_back(data[order_copy[i]]);
  }
  data = other_data;
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
  std::for_each(indices.begin(), indices.end(), [](int &i) { i += 1; });
  std::vector<int> new_order;
  for (auto i : indices) {
    new_order.emplace_back(order[i]);
  }
  order = new_order;
}

std::vector<int> distance_greater_indices(const std::vector<float> &acc_dis,
                                          const float &threshold) {
  std::vector<int> indices;
  for (int i = 0; i < acc_dis.size(); ++i) {
    if (acc_dis[i] > threshold) {
      std::cout << i << "  ";

      std::cout << acc_dis[i] << "  " << threshold << std::endl;
      indices.emplace_back(i);
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
  std::vector<float> scores = compute_score(proposals_arr);
  //   std::vector<float> scores = test_scores(proposals_arr);

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

void PrintMat(const AnchorMat &mat) {
  for (auto i : mat) {
    for (auto j : i) {
      std::cout << std::setw(3) << j << "  ";
    }
    std::cout << std::endl;
  }
}
