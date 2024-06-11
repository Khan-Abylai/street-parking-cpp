//
// Created by kartykbayev on 5/27/24.
//
#pragma once


#include <string>
#include <opencv2/opencv.hpp>

#include "../ITimer.h"

class FrameData: public ITimer {
public:
    FrameData(std::string ip, std::string presetID, cv::Mat frame,
              std::chrono::high_resolution_clock::time_point startTime);

    const std::string &getPresetID();

    const std::string &getIp();

    const cv::Mat &getFrame();


    [[nodiscard]] float getFrameWidth() const;

    [[nodiscard]] float getFrameHeight() const;

private:
    std::string ip, presetID;
    cv::Mat frame;
};


