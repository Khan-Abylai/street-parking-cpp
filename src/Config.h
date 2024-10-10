//
// Created by kartykbayev on 5/21/24.
//
#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

#include "app/Utils.h"


class Config {

public:
    static bool parseJson(const std::string &filename);

    static const std::vector<std::string> &getCameras();

    static const float &getRecognizerThreshold();

    static const float &getDetectorThreshold();

    static double getCalibrationWidth();

    static double getCalibrationHeight();

    static const std::string &getUsername();

    static const std::string &getPassword();

    static const std::string &getCalibrationEndPoint();

    static const std::string &getEventEndpoint();

    static const int &getEventInterval();

    static const std::string &getKafkaBrokers();

    static const std::string &getKafkaTopicName();
};


