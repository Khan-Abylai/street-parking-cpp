//
// Created by kartykbayev on 5/21/24.
//

#include "Config.h"

using namespace std;
using json = nlohmann::json;

std::vector<std::string> cameraVector;
float recognizerThreshold = 0.95;
float detectorThreshold = 0.85;

double calibrationWidth = 1920;
double calibrationHeight = 1080;

std::string username = "admin";
std::string password = "campas123.";

string eventEndpoint, calibrationEndpoint;


bool Config::parseJson(const std::string &filename) {
    try {
        ifstream configFile(filename);
        if (!configFile.is_open())
            throw runtime_error("Config file not found");

        json configs = json::parse(configFile);
        if (configs.find("camera_ips") == configs.end())
            throw runtime_error("Camera IP Entities not defined");

        if (configs.find("calibration") == configs.end())
            throw runtime_error("calibration node not defined");

        if (configs.find("event") == configs.end())
            throw runtime_error("event sending node not defined");

        cameraVector = Utils::splitString(configs["camera_ips"].get<string>(), ",");
        eventEndpoint = configs["event"].get<string>();
        calibrationEndpoint = configs["calibration"].get<string>();

        if (configs.find("username") != configs.end())
            username = configs["username"].get<string>();

        if (configs.find("password") != configs.end())
            password = configs["password"].get<string>();

        if (configs.find("detection_threshold") != configs.end())
            detectorThreshold = configs["detection_threshold"].get<float>();

        if (configs.find("recognition_threshold") != configs.end())
            recognizerThreshold = configs["recognition_threshold"].get<float>();

        if (configs.find("calibration_height") != configs.end())
            calibrationHeight = configs["calibration_height"].get<double>();

        if (configs.find("calibration_width") != configs.end())
            calibrationWidth = configs["calibration_width"].get<double>();

    } catch (exception &e) {
        cout << "Exception occurred during config parse: " << e.what() << endl;
        return false;
    }
    return true;
}

const std::vector<std::string> &Config::getCameras() {
    return cameraVector;
}

const float &Config::getRecognizerThreshold() {
    return recognizerThreshold;
}

const float &Config::getDetectorThreshold() {
    return detectorThreshold;
}


const std::string &Config::getUsername() {
    return username;
}

const std::string &Config::getPassword() {
    return password;
}

const std::string &Config::getCalibrationEndPoint() {
    return calibrationEndpoint;
}

const std::string &Config::getEventEndpoint() {
    return eventEndpoint;
}

double Config::getCalibrationWidth() {
    return calibrationWidth;
}

double Config::getCalibrationHeight() {
    return calibrationHeight;
}
