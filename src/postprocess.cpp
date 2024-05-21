
#include <algorithm>
#include <array>
#include <math.h>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
namespace py = pybind11;
#include </home/sandhu/project/LaneSeg/Anchor3DLane/src/utils.cpp>

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

  int count = 0;

  while (order.size() > 0) {

    int i = order[0];
    keep.emplace_back(i);
    std::vector<float> x1 = all_x[i];
    std::vector<float> z1 = all_z[i];

    FilterRows(all_x, order);
    FilterRows(all_z, order);

    std::vector<float> accumulated_dis = compute_distance(all_x, all_z, x1, z1);
    PrintVec(order);

    std::vector<int> indices_greter_than_threshold =
        distance_greater_indices(accumulated_dis, nms_thres);
    get_new_order(order, indices_greter_than_threshold);
  }
  PrintVec(keep);
  std::cout << "End is here... " << std::endl;
  return keep;
}

AnchorMat nms(py::list &proposals, float nms_thres, float conf_threshold,
              bool refine_vis, float vis_thresh) {
  unsigned list_len = py::len(proposals);
  //   std::cout << "hello sbjdfhsbdghb" << list_len;

  AnchorMat proposals_arr;
  proposals_arr = convert_pylist_to_arr(proposals);
  std::cout << proposals_arr.size() << std::endl;

  std::cout << proposals_arr[0][0];

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

PYBIND11_MODULE(postprocess, m) { m.def("nms", &nms); }

// int main() {
//   AnchorMat test_input = create_sample_input();
//   nms(test_input, 0.2, 0.2, true, 0.5);
//   std::cout << std::endl;
// }