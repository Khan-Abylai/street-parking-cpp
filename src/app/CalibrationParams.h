//
// Created by kartykbayev on 5/31/24.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <climits>
#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "LicensePlate.h"
#include <utility>


class CalibrationParams: public ILogger {
public:
    CalibrationParams(const std::string &nodeIp, std::string cameraIp,
                      float calibrationWidth, float calibrationHeight);

    bool isLicensePlateInSelectedArea(const std::shared_ptr<LicensePlate> &licensePlate);

    void getMask();

    [[nodiscard]] const std::string &getCameraIp() const;

    [[nodiscard]] float getFrameWidth() const;
    [[nodiscard]] float getFrameHeight() const;

private:
    float FRAME_WIDTH, FRAME_HEIGHT;
    const int WHITE_COLOR = 255;
    const int timeout  = 3000;
    std::string cameraIp;
    cv::Mat mask;
    std::vector<cv::Point2i> maskPoints;
    std::string calibParamsUrl;
    std::mutex maskAccessChangeMutex;

    bool isPointInTheMask(const cv::Point2i &point);

    std::string sendRequestForMaskPoints();

    [[nodiscard]] cv::Point2i getRelatedPoint(const cv::Point2f &point, const cv::Size &imageSize) const;


    [[nodiscard]] std::vector<cv::Point2i>
    getPolygonPoints(const std::string &polygonPointsStr, const std::string &maskType) const;

    [[nodiscard]] std::vector<cv::Point2i> getDefaultPolygonPoints() const;
};


