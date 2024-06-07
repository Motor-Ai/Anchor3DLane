#include "postprocess.hpp"
#include "utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using AnchorMat = std::vector<std::vector<float>>;
using Anchor = std::vector<float>;

NMS::NMS(py::list &proposals, float nms_threshold, bool is_test) {
  proposals_array_ = ConvertPyList2Array(proposals);
  nms_threshold_ = nms_threshold;
  is_test_ = is_test;
}

std::vector<int> NMS::NMS3D(AnchorMat proposals_after_thresholding,
                            std::vector<float> scores_after_thresholding) {

  std::vector<int> x_indices = create_sequence_vector(
      x_coordinate_start_idx, x_coordinate_start_idx + predication_steps - 1);
  std::vector<int> z_indices = create_sequence_vector(
      z_coordinate_start_idx, z_coordinate_start_idx + predication_steps - 1);
  std::vector<std::vector<float>> all_x =
      extract_columns<proposal_dim>(proposals_after_thresholding, x_indices);

  std::vector<std::vector<float>> all_z =
      extract_columns<proposal_dim>(proposals_after_thresholding, z_indices);

  std::vector<int> order, keep;
  order = argsort(scores_after_thresholding);

  while (order.size() > 0) {

    int i = order[0];
    if (i < 0 || i >= all_x.size() || i >= all_z.size()) {
      throw std::out_of_range("Index out of bounds");
    }
    keep.emplace_back(i);
    std::vector<float> x1 = all_x[i];
    std::vector<float> z1 = all_z[i];

    std::vector<std::vector<float>> new_x = FilterRows(all_x, order);
    std::vector<std::vector<float>> new_z = FilterRows(all_z, order);

    std::vector<float> accumulated_dis =
        ComputeAllLaneDistance(new_x, new_z, x1, z1);
    std::vector<int> indices_greter_than_threshold =
        DistanceGreaterIndices(accumulated_dis, nms_threshold_, order);

    get_new_order(order, indices_greter_than_threshold);
  }
  return keep;
}

AnchorMat NMS::PerformNMS() {

  auto [proposals_after_thresholding, scores_after_thresholding,
        anchor_inds_after_thresholding] =
      FilterProposals(proposals_array_, conf_threshold_, is_test_);

  if (proposals_after_thresholding.size() == 0) {
    throw std::range_error("Proposal length 0 after confidence filtering.");
  }

  std::vector<int> keep =
      NMS3D(proposals_after_thresholding, scores_after_thresholding);
  AnchorMat out_proposals;
  std::vector<int> out_anchors;
  out_proposals = get_items_from_indices(proposals_after_thresholding, keep);
  out_anchors = get_items_from_indices(anchor_inds_after_thresholding, keep);

  return out_proposals;
}

std::tuple<AnchorMat, std::vector<float>, std::vector<int>>
NMS::FilterProposals(const AnchorMat &proposals_arr, float conf_threshold,
                     bool is_test) {
  std::vector<int> anchor_inds = create_sequence_vector(0, num_anchors - 1);
  AnchorMat proposals_after_thresholding;
  std::vector<float> scores_after_thresholding;
  std::vector<int> anchor_inds_after_thresholding;
  std::vector<float> scores;
  if (is_test) {
    scores = test_scores(proposals_arr);
  } else {
    scores = ComputeScore(proposals_arr);
  }

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

std::vector<float>
NMS::ComputeAllLaneDistance(const std::vector<std::vector<float>> &all_x,
                            const std::vector<std::vector<float>> &all_z,
                            const std::vector<float> &x1,
                            const std::vector<float> &z1) {
  std::vector<float> distance_arr;

  for (int i = 0; i < all_x.size(); ++i) {
    std::vector<float> x2 = all_x[i];
    std::vector<float> z2 = all_z[i];
    std::vector<float> dis = distance_between_2_lanes(x1, x2, z1, z2);
    float accumulated_dis = std::accumulate(dis.begin(), dis.end(), 0.0);

    accumulated_dis /= (float)predication_steps;
    distance_arr.emplace_back(accumulated_dis);
  }
  return distance_arr;
}
std::vector<float> NMS::ComputeScore(const AnchorMat &proposals) {
  std::vector<float> scores;

  std::vector<int> indx = create_sequence_vector(65, 85);
  std::vector<std::vector<float>> scores_before_softmax =
      extract_columns<proposal_dim>(proposals, indx);
  for (auto row : scores_before_softmax) {
    scores.emplace_back(1 - softmax(row));
  }
  return scores;
}

std::vector<int> NMS::DistanceGreaterIndices(const std::vector<float> &acc_dis,
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

AnchorMat NMS::ConvertPyList2Array(py::list proposals) {
  unsigned list_len = py::len(proposals);

  AnchorMat proposal_arr;
  Anchor temp;

  for (int i = 0; i < list_len; ++i) {
    int col = i % proposal_dim;

    temp.emplace_back(proposals[i].cast<float>());
    if (col == proposal_dim - 1) {
      proposal_arr.emplace_back(temp);
      temp.clear();
    }
  }
  return proposal_arr;
}

PYBIND11_MODULE(postprocess, m) {
  py::class_<NMS>(m, "NMS")
      .def(py::init<AnchorMat, float, bool>())
      .def(py::init<py::list &, float, bool>())
      .def("PerformNMS", &NMS::PerformNMS);
}
