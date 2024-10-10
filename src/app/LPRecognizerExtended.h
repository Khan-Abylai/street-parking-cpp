//
// Created by kartykbayev on 9/5/24.
//

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "Constants.h"
#include "TensorRTDeleter.h"
#include "TrtLogger.h"
#include "TensorRTEngine.h"


class LPRecognizerExtended {
public:
    LPRecognizerExtended();

    ~LPRecognizerExtended();

    std::tuple<std::string, double, std::string> makePrediction(const std::vector<cv::Mat> &frames);

private:
    const int
            SEQUENCE_SIZE = Constants::EXTENDED_SEQUENCE_SIZE,
            ALPHABET_SIZE = Constants::EXTENDED_ALPHABET_SIZE,
            BLANK_INDEX = 0,
            IMG_WIDTH = Constants::EXTENDED_RECT_LP_W,
            IMG_HEIGHT = Constants::EXTENDED_RECT_LP_H,
            IMG_CHANNELS = Constants::IMG_CHANNELS,
            INPUT_SIZE = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH,
            OUTPUT_SIZE = SEQUENCE_SIZE * ALPHABET_SIZE,
            REGION_SIZE = 37,
            OUTPUT_2_SIZE = REGION_SIZE,
            MAX_BATCH_SIZE = Constants::RECOGNIZER_MAX_BATCH_SIZE,
            MAX_PLATE_SIZE = 12;
    const std::string
            ALPHABET = Constants::EXTENDED_ALPHABET,
            NETWORK_INPUT_NAME = "INPUT",
            NETWORK_DIM_NAME = "DIMENSIONS",
            NETWORK_OUTPUT_NAME = "OUTPUT",
            NETWORK_OUTPUT_2_NAME = "OUTPUT_2",
            ENGINE_NAME = "recognizer_eur_marocco.engine",
            WEIGHTS_FILENAME = "recognizer_eur_marocco.np";
    std::vector<int> dimensions;

    void createEngine();


    // std::vector<std::string> REGIONS{"dubai", "abu-dhabi", "sharjah", "ajman", "ras-al-khaimah", "fujairah",
    //                                  "alquwain", "bahrein", "oman", "saudi", "quatar", "kuwait", "others"};

    std::vector<std::string> REGIONS{"albania", "andorra", "austria", "belgium", "bosnia", "bulgaria", "croatia", "cyprus", "czech", "estonia",
                                     "finland", "france", "germany", "greece", "hungary", "ireland", "italy", "latvia",
                                     "licht", "lithuania", "luxemburg", "makedonia", "malta", "monaco", "montenegro", "netherlands", "poland",
                                     "portugal", "romania", "san_marino", "serbia", "slovakia", "slovenia", "spain", "sweden", "swiss", "marocco"};



    std::pair<std::vector<float>, std::vector<float>> executeInferEngine(const std::vector<cv::Mat> &frames);

    std::vector<float> prepareImage(const std::vector<cv::Mat> &frames) const ;

    void *cudaBuffer[4];

    nvinfer1::IExecutionContext *executionContext;
    nvinfer1::ICudaEngine *engine = nullptr;
    cudaStream_t stream;


};


