#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace Constants {
    enum class CountryCode {
        KZ,
        KG,
        UZ,
        RU,
        BY,
        GE,
        AM,
        AZ
    };

    const std::string UNIDENTIFIED_COUNTRY;
    const std::string JPG_EXTENSION = ".jpg";
    const std::string IMAGE_DIRECTORY = "./images/";
    const std::string MODEL_FOLDER{"./models/"};

    static const int BLACK_IMG_WIDTH = 12;

    const int DETECTION_IMG_W = 512;
    const int DETECTION_IMG_H = 512;

    const int IMG_CHANNELS = 3;

    const int PLATE_COORDINATE_SIZE = 13;
    const int DETECTION_BATCH_SIZE = 1;
    const float confThreshold = 0.8f;
    const float iouThreshold = 0.4f;

    const int STANDARD_RECT_LP_H = 32;
    const int STANDARD_RECT_LP_W = 128;

    const int EXTENDED_RECT_LP_H = 64;
    const int EXTENDED_RECT_LP_W = 160;

    const int SQUARE_LP_H = 64;
    const int SQUARE_LP_W = 64;

    const int RECOGNIZER_MAX_BATCH_SIZE = 1;

    const int STANDARD_SEQUENCE_SIZE = 30;
    const int STANDARD_ALPHABET_SIZE = 38;
    const std::string STANDARD_ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz.";

    const int EXTENDED_SEQUENCE_SIZE = 38;
    const int EXTENDED_ALPHABET_SIZE = 37;
    const std::string EXTENDED_ALPHABET = "-0123456789abcdefghijklmnopqrstuvwxyz";


    constexpr float PIXEL_MAX_VALUE = 255;

    const int LP_WHITENESS_MAX = 200;
    const int LP_WHITENESS_MIN = 90;

    constexpr float PIXEL_MEAN_1_VALUE = 104;
    constexpr float PIXEL_MEAN_2_VALUE = 117;
    constexpr float PIXEL_MEAN_3_VALUE = 123;

    const std::vector<cv::Point2f> RECT_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 31),
            cv::Point2f(127, 0),
            cv::Point2f(127, 31),
    };

    const std::vector<cv::Point2f> SQUARE_LP_COORS{
            cv::Point2f(0, 0),
            cv::Point2f(0, 63),
            cv::Point2f(63, 0),
            cv::Point2f(63, 63),
    };
}

