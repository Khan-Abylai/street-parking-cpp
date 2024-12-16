#include "Slicing.h"

Slicing::Slicing(int numSlices, float paddingRatio)
    : numSlices(numSlices), paddingRatio(paddingRatio) {}

std::pair<int,int> Slicing::calculatePadding(int sliceH, int sliceW) {
    int paddingH = static_cast<int>(sliceH * paddingRatio);
    int paddingW = static_cast<int>(sliceW * paddingRatio);
    return {paddingH, paddingW};
}

std::vector<std::tuple<cv::Mat,int,int>> Slicing::sliceImage(const cv::Mat &img) {
    std::vector<std::tuple<cv::Mat,int,int>> slices;
    int h = img.rows;
    int w = img.cols;

    int sliceH = h / numSlices;
    int sliceW = w / numSlices;

    auto [paddingH, paddingW] = calculatePadding(sliceH, sliceW);

    // int count = 0;

    for (int y = 0; y < h; y += sliceH) {
        for (int x = 0; x < w; x += sliceW) {
            int x_raw_start = x;
            int y_raw_start = y;
            int x_raw_end = std::min(w, x + sliceW);
            int y_raw_end = std::min(h, y + sliceH);

            int x_start = std::max(0, x - paddingW);
            int y_start = std::max(0, y - paddingH);
            int x_end = std::min(w, x + sliceW + paddingW);
            int y_end = std::min(h, y + sliceH + paddingH);

            cv::Mat crop = img(cv::Rect(x_start, y_start, x_end - x_start, y_end - y_start)).clone();
            slices.push_back({crop, x_start, y_start});

            cv::Mat originalClip = img(cv::Rect(x_raw_start, y_raw_start,
                                                x_raw_end - x_raw_start, y_raw_end - y_raw_start)).clone();

            // if(DEBUG) {
            //     cv::imwrite("../test/slice_original" + std::to_string(count) + ".jpg", originalClip);
            //     cv::imwrite("../test/slice_padded" + std::to_string(count) + ".jpg", crop);
            // }
            // count++;
        }
    }

    return slices;
}

float Slicing::iou(const std::array<float,4> &a, const std::array<float,4> &b) {
    float inter_xmin = std::max(a[0], b[0]);
    float inter_ymin = std::max(a[1], b[1]);
    float inter_xmax = std::min(a[2], b[2]);
    float inter_ymax = std::min(a[3], b[3]);

    float inter_w = std::max(0.0f, inter_xmax - inter_xmin);
    float inter_h = std::max(0.0f, inter_ymax - inter_ymin);
    float inter_area = inter_w * inter_h;

    float area_a = (a[2]-a[0]) * (a[3]-a[1]);
    float area_b = (b[2]-b[0]) * (b[3]-b[1]);

    float denom = area_a + area_b - inter_area;
    if (denom > 0.0f) {
        return inter_area / denom;
    }
    return 0.0f;
}

std::tuple<std::vector<std::array<float,4>>, std::vector<float>>
Slicing::weightedBoxesFusion(const std::vector<std::vector<std::array<float,4>>> &boxes_list,
                             const std::vector<std::vector<float>> &scores_list,
                             float iou_thr,
                             float skip_box_thr) {
    std::vector<std::array<float,4>> all_boxes;
    std::vector<float> all_scores;

    for (size_t i = 0; i < boxes_list.size(); i++) {
        for (size_t j = 0; j < boxes_list[i].size(); j++) {
            float sc = scores_list[i][j];
            if (sc >= skip_box_thr) {
                all_boxes.push_back(boxes_list[i][j]);
                all_scores.push_back(sc);
            }
        }
    }

    if (all_boxes.empty()) {
        return {{},{}};
    }

    std::vector<int> indices(all_boxes.size());
    for (int i = 0; i < (int)all_boxes.size(); i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int a, int b){
        return all_scores[a] > all_scores[b];
    });

    std::vector<std::array<float,4>> fused_boxes;
    std::vector<float> fused_scores;
    std::vector<bool> used(all_boxes.size(), false);

    for (int i_idx = 0; i_idx < (int)indices.size(); i_idx++) {
        int i_box_index = indices[i_idx];
        if (used[i_box_index]) continue;

        std::vector<std::array<float,4>> cluster_boxes;
        std::vector<float> cluster_scores;

        cluster_boxes.push_back(all_boxes[i_box_index]);
        cluster_scores.push_back(all_scores[i_box_index]);
        used[i_box_index] = true;

        for (int j_idx = i_idx + 1; j_idx < (int)indices.size(); j_idx++) {
            int j_box_index = indices[j_idx];
            if (used[j_box_index]) continue;

            float current_iou = iou(all_boxes[i_box_index], all_boxes[j_box_index]);
            if (current_iou > iou_thr) {
                cluster_boxes.push_back(all_boxes[j_box_index]);
                cluster_scores.push_back(all_scores[j_box_index]);
                used[j_box_index] = true;
            }
        }

        float total_score = 0.f;
        float x_min = 0.f, y_min = 0.f, x_max = 0.f, y_max = 0.f;

        for (size_t c = 0; c < cluster_boxes.size(); c++) {
            float sc = cluster_scores[c];
            total_score += sc;
            x_min += cluster_boxes[c][0] * sc;
            y_min += cluster_boxes[c][1] * sc;
            x_max += cluster_boxes[c][2] * sc;
            y_max += cluster_boxes[c][3] * sc;
        }

        x_min /= total_score;
        y_min /= total_score;
        x_max /= total_score;
        y_max /= total_score;

        float fused_score = total_score / (float)cluster_boxes.size();
        fused_boxes.push_back({x_min, y_min, x_max, y_max});
        fused_scores.push_back(fused_score);
    }

    return {fused_boxes, fused_scores};
}

std::tuple<std::vector<std::array<float,4>>, std::vector<float>>
Slicing::applyWBF(const std::vector<std::vector<std::array<float,4>>> &boxes_list,
                  const std::vector<std::vector<float>> &scores_list,
                  float iou_thr,
                  float skip_box_thr) {
    return weightedBoxesFusion(boxes_list, scores_list, iou_thr, skip_box_thr);
}
