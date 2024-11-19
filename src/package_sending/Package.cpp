//
// Created by kartykbayev on 5/31/24.
//

#include "Package.h"

using namespace std;
using json = nlohmann::json;
Package::Package(std::string cameraIp, std::vector<std::string> licensePlateLabelsParam,
                 cv::Mat carImage, std::vector<std::string> licensePlateBBoxesParam):cameraIp{std::move(cameraIp)},
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

std::vector<std::map<std::string, std::string>> convertToDictList(
    const std::vector<std::string>& plateNumbers, const std::vector<std::string>& licensePlateBBoxes){
    std::vector<std::map<std::string, std::string>> result;

    size_t maxSize = plateNumbers.size();

    for (size_t i = 0; i < maxSize; ++i)
    {
        std::map<std::string, std::string> entry;
        entry["car_number"] = plateNumbers[i];
        entry["lp_rect"] = licensePlateBBoxes[i];

        result.push_back(entry);
    }

    return result;
}

std::string convertDictToRawString(const std::vector<std::map<std::string, std::string>>& cars) {
    std::ostringstream oss;
    json packageJson;
    oss << "[";
    for (size_t i = 0; i < cars.size(); ++i) {
        const auto& car = cars[i];
        oss << "{";
        oss << "\"car_number\": \"" << car.at("car_number") << "\" ";
        // oss << ", ";
        // oss << "\"lp_rect\": \"" << car.at("lp_rect") << "\" ";
        oss << "}";
        if (i < cars.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}


const std::string &Package::getCameraIp() const {
    return cameraIp;
}

std::string Package::getPackageJson() const {
    json packageJson;
    packageJson["ip_address"] = cameraIp;
    packageJson["cars"] =  convertDictToRawString(convertToDictList(licensePlateLabels, licensePlateBBoxes));
    packageJson["event_time"] = Utils::dateTimeToStr(eventTime);
    packageJson["picture"] =  Utils::encodeImgToBase64(carImage);
    // packageJson["license_plate_labels"] = convertVectorToRawString(licensePlateLabels);
    // packageJson["license_plate_bboxes"] = convertVectorToRawString(licensePlateBBoxes);

    return packageJson.dump();
}

const std::string &Package::getPlateLabelsRaw() const {
    return licensePlateLabelsRaw;
}

