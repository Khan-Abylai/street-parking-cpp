//
// Created by artyk on 9/20/2023.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>
#include "Constants.h"
#include "Utils.h"
#include "TemplateMatching.h"
#include "LicensePlate.h"
#include <cmath>
#include "net.h"


class LPRecognizerNCNN {
public:
    LPRecognizerNCNN();

    ~LPRecognizerNCNN();

    std::vector <std::pair<std::string, float>> predict(const std::vector <cv::Mat> &frames);
private:
    ncnn::Net recognizer{};

    std::vector<float> makeFlattened(ncnn::Mat &val);

    std::vector<float> softmax(std::vector<float> &score_vec);

    const std::string
            ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz.";

    const int
            SEQUENCE_SIZE = 31,
            ALPHABET_SIZE = 38,
            BLANK_INDEX = 0,
            IMG_WIDTH = 128,
            IMG_HEIGHT = 32,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            MAX_BATCH_SIZE = Constants::RECOGNIZER_MAX_BATCH_SIZE,
            MAX_PLATE_SIZE = 12;
};
