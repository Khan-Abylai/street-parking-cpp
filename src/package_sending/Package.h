//
// Created by kartykbayev on 5/31/24.
//
#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <utility>

#include "../app/Utils.h"
#include "../app/LicensePlate.h"
#include "../ITimer.h"

class Package: public ITimer{
public:
    Package(std::string cameraIp, std::vector<std::string> licensePlateLabelsParam, cv::Mat carImage, std::vector<std::string> licensePlateBBoxesParam);

    [[nodiscard]] std::string getPackageJson() const;

    [[nodiscard]] const std::string &getCameraIp() const;
    [[nodiscard]] const std::string &getPlateLabelsRaw() const;
    static std::string convertVectorToRawString(std::vector<std::string> lps);

    [[nodiscard]] std::string getEventTime() const;
private:
    std::vector<std::string> licensePlateLabels, licensePlateBBoxes;
    cv::Mat carImage;
    std::string cameraIp, licensePlateLabelsRaw, licensePlateBBoxesRaw;
    time_t eventTime;

};


