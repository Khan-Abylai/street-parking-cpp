
//
// Created by kartykbayev on 5/30/24.
//
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <vector>

#include "net.h"

#include "LicensePlate.h"
#include "Constants.h"

#define NUM_KEY_POINTS 5
#define INPUT_WIDTH  640
#define INPUT_HEIGHT 640

struct Objects
{
    cv::Rect_<float> rect;
    int class_id;
    float prob;
    std::vector<cv::Point2f> pts;
};

class DetectorYoloV5NCNN {
public:
    explicit DetectorYoloV5NCNN();
    ~DetectorYoloV5NCNN();
    std::vector<std::shared_ptr<LicensePlate>> detect(const cv::Mat &rgb);
private:
    int NUM_THREADS = 4;
    std::string labels[2] = {"plate", "car"};
    const float PROB_THRESHOLD = 0.7, NMS_THRESHOLD = 0.4;
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    const float stride[3] = { 8.0, 16.0, 32.0 };
    const float anchors[3][6] = { {4,5,  8,10,  13,16}, {23,29,  43,55,  73,105},{146,217,  231,300,  335,433} };
    ncnn::Net yolo{};

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }


};


