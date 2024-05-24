
#include <algorithm>
#include <array>
#include <math.h>
#include <numeric>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <vector>
// namespace py = pybind11;
#include "prints.cpp"
#include "utils.cpp"


class NMS {

  // NMS() {

  // }

public:
  std::vector<int> nms_3d(AnchorMat proposals_after_thresholding,
                          std::vector<float> scores_after_thresholding,
                          float nms_thres) {

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
          compute_distance(new_x, new_z, x1, z1);

      std::vector<int> indices_greter_than_threshold =
          distance_greater_indices(accumulated_dis, nms_thres, order);

      get_new_order(order, indices_greter_than_threshold);
    }
    return keep;
  }

  // AnchorMat nms(py::list &proposals, float nms_thres, float conf_threshold,
  //               bool refine_vis, float vis_thresh) {

  AnchorMat nms(AnchorMat proposals_arr, float nms_thres, float conf_threshold,
                bool refine_vis, float vis_thresh) {
    // AnchorMat proposals_arr = convert_pylist_to_arr(proposals);

    auto [proposals_after_thresholding, scores_after_thresholding,
          anchor_inds_after_thresholding] =
        filter_proposals(proposals_arr, conf_threshold);

    if (proposals_after_thresholding.size() == 0) {
      // TODO: Raise Error!
      std::cout << "Sorry" << std::endl;
    }

    std::vector<int> keep = nms_3d(proposals_after_thresholding,
                                   scores_after_thresholding, nms_thres);

    AnchorMat out_proposals;
    std::vector<int> out_anchors;
    out_proposals = get_items_from_indices(proposals_after_thresholding, keep);
    out_anchors = get_items_from_indices(anchor_inds_after_thresholding, keep);

    return out_proposals;
  }
};
// PYBIND11_MODULE(postprocess, m) { m.def("nms", &nms); }
