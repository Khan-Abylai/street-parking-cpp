#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

#include "Utils.h"
#include <filesystem>
#include <iostream>
#include <vector>
#include <array>
#include <tuple>

class Slicing {
public:
    Slicing(int numSlices = 4, float paddingRatio = 0.15f);

    std::vector<std::tuple<cv::Mat,int,int>> sliceImage(const cv::Mat &img);

     /**
     * Применение Weighted Boxes Fusion к нескольким наборам боксов.
     * boxes_list и scores_list:
     * - Размер: N источников (срезов), каждый — вектор боксов.
     * - Координаты боксов в нормализованном формате [0,1], [x_min, y_min, x_max, y_max].
     * Все боксы считаем одного класса.
     *
     * Возвращает объединенные боксы и их оценки.
     */
    static std::tuple<std::vector<std::array<float,4>>, std::vector<float>>
    applyWBF(const std::vector<std::vector<std::array<float,4>>> &boxes_list,
             const std::vector<std::vector<float>> &scores_list,
             float iou_thr=0.5f,
             float skip_box_thr=0.5f);

private:
    int numSlices;
    float paddingRatio;

    std::pair<int,int> calculatePadding(int sliceH, int sliceW);

    static float iou(const std::array<float,4> &a, const std::array<float,4> &b);
    static std::tuple<std::vector<std::array<float,4>>, std::vector<float>>
    weightedBoxesFusion(const std::vector<std::vector<std::array<float,4>>> &boxes_list,
                        const std::vector<std::vector<float>> &scores_list,
                        float iou_thr,
                        float skip_box_thr);
};
