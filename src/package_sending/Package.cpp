//
// Created by kartykbayev on 5/31/24.
//

#include "Package.h"

using namespace std;
using json = nlohmann::json;
Package::Package(std::string cameraIp, std::string presetId, std::vector<std::string> licensePlateLabelsParam,
                 cv::Mat carImage, std::vector<std::string> licensePlateBBoxesParam):cameraIp{std::move(cameraIp)}, presetId{std::move(presetId)},
                 licensePlateLabels{std::move(licensePlateLabelsParam)}, carImage{std::move(carImage)}, licensePlateBBoxes{std::move(licensePlateBBoxesParam)} {
    eventTime = time_t(nullptr);
    licensePlateLabelsRaw= convertVectorToRawString(licensePlateLabels);
    licensePlateBBoxesRaw = convertVectorToRawString(licensePlateBBoxes);
}

std::string Package::getEventTime() const {
    return Utils::dateTimeToStr(eventTime);
}

std::string Package::convertVectorToRawString(std::vector<std::string> lps) {
    std::string delimiter = ";";
    std::ostringstream oss;
    for (auto it = lps.begin(); it != lps.end(); ++it) {
        if (it != lps.begin()) {
            oss << delimiter;
        }
        oss << *it;
    }
    return oss.str();
}

const std::string &Package::getPresetId() const {
    return presetId;
}

const std::string &Package::getCameraIp() const {
    return cameraIp;
}

std::string Package::getPackageJson() const {
    json packageJson;
    packageJson["camera_ip"] = cameraIp;
    packageJson["preset_id"] = presetId;
    packageJson["license_plate_labels"] = convertVectorToRawString(licensePlateLabels);
    packageJson["image"] =  Utils::encodeImgToBase64(carImage);
    packageJson["event_time"] = Utils::dateTimeToStr(eventTime);
    packageJson["license_plate_bboxes"] = convertVectorToRawString(licensePlateBBoxes);
    return packageJson.dump();
}

const std::string &Package::getPlateLabelsRaw() const {
    return licensePlateLabelsRaw;
}

