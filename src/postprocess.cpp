#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <math.h>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
namespace py = pybind11;

constexpr int num_anchors = 4431;
constexpr int dim_proposal = 86;
constexpr int dim_anchor = 65;

void PrintArr(std::array<int, num_anchors> arr) {
  for (auto i : arr) {
    std::cout << i << std::endl;
  }
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

template <typename T> void PrintVec(std::vector<T> vec) {
  for (auto i : vec) {
    std::cout << i << "  ";
  }
  std::cout << std::endl;
}

std::vector<float> compute_distance(std::vector<std::vector<float>> all_x,
                                    std::vector<std::vector<float>> all_z,
                                    std::vector<float> x1,
                                    std::vector<float> z1) {
  std::vector<float> distance_arr;

  for (int i = 0; i < all_x.size(); ++i) {
    std::vector<float> x2 = all_x[i];
    std::vector<float> z2 = all_z[i];
    std::vector<float> distance__(x2.size());
    // if (i == 0) {
    //   std::cout << "x1: " << std::endl;
    //   PrintVec(x1);
    //   std::cout << "z1: " << std::endl;
    //   PrintVec(z1);
    //   std::cout << "x2: " << std::endl;
    //   PrintVec(x2);
    //   std::cout << "z2: " << std::endl;
    //   PrintVec(z2);
    // }
    std::transform(x1.begin(), x1.end(), x2.begin(), x2.begin(),
                   std::minus<float>());
    std::transform(z1.begin(), z1.end(), z2.begin(), z2.begin(),
                   std::minus<float>());

    // if (i == 0) {
    //   std::cout << "diff x: " << std::endl;
    //   PrintVec(x2);
    //   std::cout << "diff z: " << std::endl;
    //   PrintVec(z2);
    // }
    std::transform(z2.begin(), z2.end(), x2.begin(), distance__.begin(),
                   [](float z, float x) { return sqrt(z * z + x * x); });

    // if (i == 0) {
    //   std::cout << "distance: " << std::endl;
    //   PrintVec(distance__);
    // }

    float accumulated_dis =
        std::accumulate(distance__.begin(), distance__.end(), 0.0);

    accumulated_dis /= 20.0;
    distance_arr.emplace_back(accumulated_dis);
    // std::cout << distance_arr.size() << " ";
  }
  //   std::cout << "hello" << std::endl;
  //   std::cout << "Size: " << distance_arr.size();
  return distance_arr;
}

std::vector<int> inds_where(std::vector<float> acc_dis, float threshold,
                            std::vector<int> order, int counter) {
  std::vector<int> order_;
  //   std::cout << order.size() << std::endl;
  auto check_threshold = [threshold](float a) { return a > threshold; };
  auto it = std::find_if(acc_dis.begin(), acc_dis.end(), check_threshold);
  for (; it != acc_dis.end();
       it = std::find_if(++it, acc_dis.end(), check_threshold)) {
    if (it - acc_dis.begin() + 1 == order.size()) {
      break;
    }
    order_.push_back(order[it - acc_dis.begin() + 1]);
    // std::cout << it - acc_dis.begin() + 1 << " ";
  }
  //   std::cout << std::endl;
  return order_;
}

// std::vector<int> inds_where(std::vector<float> acc_dis, float threshold,
//                             std::vector<int> order, int counter) {
//   std::vector<int> order_;
//   std::vector<int> inds;
//   for (int i = 0; i < acc_dis.size(); ++i) {
//     if (acc_dis[i] > threshold) {
//       inds.emplace_back(i);
//     }
//   }
//   for (auto i : inds) {
//     order_.emplace_back(order[i + 1]);
//   }
//   return order_;
// }

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
  std::vector<std::vector<float>> other_data;

  // Iterate through the order vector
  for (int i = 1; i < order.size(); i++) {
    other_data.push_back(data[order[i]]);
  }
  data = other_data;
}

std::vector<int>
nms_3d(std::vector<std::array<float, 86>> proposals_after_thresholding,
       std::vector<float> scores_after_thresholding,
       std::vector<std::vector<float>> vises, int nms_thres, int anchor_len) {

  std::vector<int> x_indices = create_sequence_vector(5, 24);
  std::vector<int> z_indices = create_sequence_vector(25, 44);
  std::vector<std::vector<float>> all_x =
      extract_columns<86>(proposals_after_thresholding, x_indices);
  std::vector<std::vector<float>> all_z =
      extract_columns<86>(proposals_after_thresholding, z_indices);
  std::vector<int> order, keep;
  order = argsort(scores_after_thresholding);

  std::cout << "C++  " << std::endl;
  std::cout << std::endl;

  std::vector<float> accumulated_dis;
  int count = 0;

  while (order.size() > 0) {
    // std::cout << order.size() << " Before Order" << std::endl;

    int i = order[0];
    keep.emplace_back(i);
    std::vector<float> x1 = all_x[i];
    std::vector<float> z1 = all_z[i];

    FilterRows(all_x, order);
    FilterRows(all_z, order);

    std::vector<float> vis1 = vises[i];
    accumulated_dis = compute_distance(all_x, all_z, x1, z1);

    order = inds_where(accumulated_dis, nms_thres, order, count);

    if (count == 1) {
      std::cout << std::endl;

      std::cout << all_x.size() << " allx: " << std::endl;
      std::cout << order.size() << " Order: " << std::endl;

      std::cout << i << " I: " << std::endl;

      std::cout << "C++ X1" << std::endl;
      PrintVec(x1);
      std::cout << "C++ Z1" << std::endl;
      PrintVec(z1);
      std::cout << "C++ all X" << std::endl;
      std::cout << all_x.size() << std::endl;
      PrintVec(all_x[0]);
      std::cout << "C++ all Z" << std::endl;
      std::cout << all_z.size() << std::endl;
      PrintVec(all_z[0]);
      std::cout << "C++ acc distance" << std::endl;

      sort(accumulated_dis.begin(), accumulated_dis.end());
      PrintVec(accumulated_dis);
      std::cout << accumulated_dis.size() << std::endl;
      break;
    }
    count += 1;
    }

  std::cout << "End is here... " << std::endl;
  return keep;
}

std::vector<std::array<float, 86>> convert_pylist_to_arr(py::list proposals) {
  unsigned list_len = py::len(proposals);
  std::vector<std::array<float, 86>> proposal_arr;
  for (int i = 0; i < list_len; ++i) {
    std::array<float, 86> temp;
    int col = i % 86;
    temp[col] = proposals[i].cast<float>();
    if (col == 85) {
      proposal_arr.emplace_back(temp);
    }
  }
  return proposal_arr;
}

float softmax(std::vector<float> row) {
  std::transform(row.begin(), row.end(), row.begin(),
                 [](float r) { return exp(r); });
  float exp_row_sum = std::accumulate(row.begin(), row.end(), 0.0);
  return row[0] / exp_row_sum;
}

std::vector<float> compute_score(std::vector<std::array<float, 86>> proposals) {
  std::vector<float> scores;

  std::vector<int> indx = create_sequence_vector(65, 85);
  std::vector<std::vector<float>> scores_before_softmax =
      extract_columns<86>(proposals, indx);

  for (auto row : scores_before_softmax) {
    scores.emplace_back(1 - softmax(row));
  }
  return scores;
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

std::vector<std::array<float, 86>> nms(py::list &proposals, int nms_thres,
                                       float conf_threshold, bool refine_vis,
                                       float vis_thresh) {
  int anchor_len = 20;
  std::vector<std::array<float, 86>> proposals_arr;
  proposals_arr = convert_pylist_to_arr(proposals);

  std::vector<int> anchor_inds = create_sequence_vector(0, 4430);
  std::vector<std::array<float, 86>> proposals_after_thresholding;

  std::vector<float> scores = compute_score(proposals_arr);
  std::vector<float> scores_after_thresholding;
  std::vector<int> anchor_inds_after_thresholding;

  if (conf_threshold > 0) {
    for (int i = 0; i < num_anchors; ++i) {
      if (scores[i] > conf_threshold) {
        proposals_after_thresholding.emplace_back(proposals_arr[i]);
        scores_after_thresholding.emplace_back(scores[i]);
        anchor_inds_after_thresholding.emplace_back(anchor_inds[i]);
      }
    }
  }

  if (proposals_after_thresholding.size() == 0) {
    // [TODO: Raise Error!]
    std::cout << "Sorry" << std::endl;
  }
  std::vector<int> vises_indices = create_sequence_vector(45, 64);
  std::vector<std::vector<float>> vises =
      extract_columns<86>(proposals_arr, vises_indices);
  std::vector<int> keep;
  keep = nms_3d(proposals_after_thresholding, scores_after_thresholding, vises,
                nms_thres, anchor_len);

  std::vector<std::array<float, 86>> out_proposals;
  std::vector<int> out_anchors;
  out_proposals = get_items_from_indices(proposals_after_thresholding, keep);
  out_anchors = get_items_from_indices(anchor_inds_after_thresholding, keep);
  //   std::cout << "C++" << std::endl;

  //   PrintVec(keep);
  return out_proposals;
}

PYBIND11_MODULE(postprocess, m) { m.def("nms", &nms); }