/**
 * @file postprocess.hpp
 * @author vardeep singh sandhu
 * @brief This file contains implementaiton of NMS ofr lanes.
 * @date 2024-06-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include "utils.hpp"
#include <algorithm>
#include <array>
#include <math.h>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using AnchorMat = std::vector<std::vector<float>>;
using Anchor = std::vector<float>;

/**
 * @brief Anchor
 An anchor is a 86 dims vector where we have from:
 1. Index [6 - 25] x-coordinate information.
 2. Index [26 - 45] z-coordinate information.
 2. Index [46 - 86] category information.
 *
 */

/**
 * @brief The NMS (Non-Maximum Suppression) class used for object detection
 * post-processing.
 */
class NMS {

private:
  int num_anchors = 4431;
  static constexpr int proposal_dim = 86;
  int dim_anchor = 65;
  int x_coordinate_start_idx = 5;
  int z_coordinate_start_idx = 25;
  int predication_steps = 20;

  //   int num_anchors = 6;
  //   static constexpr int proposal_dim = 5;
  //   int dim_anchor = 65;
  //   int x_coordinate_start_idx = 1;
  //   int z_coordinate_start_idx = 3;
  //   int predication_steps = 2;

  float nms_threshold_;
  float conf_threshold_ = 0.2;
  bool refine_vis_ = false;
  float vis_threshold_ = 0.5;
  AnchorMat proposals_array_;
  bool is_test_;

public:
  /**
   * @brief Constructs an NMS object from a Python list.
   * @param proposals The proposals list.
   * @param nms_threshold The NMS threshold.
   * @param is_test Indicates if this is a test.
   */
  NMS(py::list &proposals, float nms_threshold, bool is_test);

  /**
   * @brief Constructs an NMS object.
   * @param proposals The proposals array.
   * @param nms_threshold The NMS threshold.
   * @param is_test Indicates if this is a test.
   */
  NMS(AnchorMat proposals, float nms_threshold, bool is_test)
      : proposals_array_(proposals), nms_threshold_(nms_threshold),
        is_test_(is_test){};

  /**
   * @brief Performs Non-Maximum Suppression (NMS) on the proposals.
   * @return The proposals after applying NMS.
   */
  AnchorMat PerformNMS();

  /**
   * @brief Performs 3D NMS on the given proposals and scores.
   * @param proposals_after_thresholding The proposals after thresholding.
   * @param scores_after_thresholding The scores after thresholding.
   * @return A vector of indices representing the selected proposals.
   */
  std::vector<int> NMS3D(AnchorMat proposals_after_thresholding,
                         std::vector<float> scores_after_thresholding);

  /**
   * @brief Computes scores for the given proposals.
   * @param proposals The proposals to compute scores for.
   * @return A vector of scores.
   */
  std::vector<float> ComputeScore(const AnchorMat &proposals);

  /**
   * @brief Computes the distance between all lanes and a given lane.
   * @param all_x The x-coordinates of all lanes.
   * @param all_z The z-coordinates of all lanes.
   * @param x1 The x-coordinates of the reference lane.
   * @param z1 The z-coordinates of the reference lane.
   * @return A vector of distances between each lane and the reference lane.
   */
  std::vector<float>
  ComputeAllLaneDistance(const std::vector<std::vector<float>> &all_x,
                         const std::vector<std::vector<float>> &all_z,
                         const std::vector<float> &x1,
                         const std::vector<float> &z1);

  /**
   * @brief Filters the proposals based on confidence threshold and test flag.
   * @param proposals_arr The array of proposals.
   * @param conf_threshold The confidence threshold.
   * @param is_test Indicates if this is a test.
   * @return A tuple containing the filtered proposals, their scores, and their
   * indices.
   */
  std::tuple<AnchorMat, std::vector<float>, std::vector<int>>
  FilterProposals(const AnchorMat &proposals_arr, float conf_threshold,
                  bool is_test);

  /**
   * @brief Gets the indices of elements in a vector where the distance is
   * greater than a threshold.
   * @param acc_dis The vector of accumulated distances.
   * @param threshold The distance threshold.
   * @param order The vector of indices to consider.
   * @return A vector of indices where the distance is greater than the
   * threshold.
   */
  std::vector<int> DistanceGreaterIndices(const std::vector<float> &acc_dis,
                                          const float &threshold,
                                          const std::vector<int> &order);
  /**
   * @brief Converts a Python list of proposals to an array.
   * @param proposals The Python list of proposals.
   * @return The converted array of proposals.
   */
  AnchorMat ConvertPyList2Array(py::list proposals);
};

#endif // POSTPROCESS_HPP
