//
// Created by kartykbayev on 5/31/24.
//

#include "CalibrationParams.h"
using namespace std;
using json = nlohmann::json;

CalibrationParams::CalibrationParams(const std::string &nodeIp, std::string cameraIp,
                                     float calibrationWidth, float calibrationHeight):ILogger("Calibration Params"), cameraIp{std::move(cameraIp)} {
    FRAME_WIDTH = calibrationWidth;
    FRAME_HEIGHT = calibrationHeight;
    calibParamsUrl = nodeIp+ getCameraIp();
    LOG_INFO("Calibration Parameters URL:%s", calibParamsUrl.data());
    getMask();
}

bool CalibrationParams::isLicensePlateInSelectedArea(const std::shared_ptr<LicensePlate> &licensePlate) {
    auto centerPoint = getRelatedPoint(licensePlate->getCenter(), licensePlate->getCarImageSize());
    return isPointInTheMask(centerPoint);
}

cv::Point2i CalibrationParams::getRelatedPoint(const cv::Point2f &point, const cv::Size &imageSize) const {
    auto xPoint = (point.x * FRAME_WIDTH) / (float) imageSize.width;
    auto yPoint = (point.y * FRAME_HEIGHT) / (float) imageSize.height;
    return cv::Point2i{(int) xPoint, (int) yPoint};}

void CalibrationParams::getMask() {
    auto responseText = sendRequestForMaskPoints();
    auto polygonPoints = getPolygonPoints(responseText, "mask");

    lock_guard<mutex> guard(maskAccessChangeMutex);
    mask = cv::Mat::zeros((int) FRAME_HEIGHT, (int) FRAME_WIDTH, CV_8UC1);

    cv::fillConvexPoly(mask, polygonPoints, WHITE_COLOR);
}

const std::string &CalibrationParams::getCameraIp() const {
    return this->cameraIp;
}

float CalibrationParams::getFrameWidth() const {
    return FRAME_WIDTH;
}

float CalibrationParams::getFrameHeight() const {
    return FRAME_HEIGHT;
}

bool CalibrationParams::isPointInTheMask(const cv::Point2i &point) {
    lock_guard<mutex> guard(maskAccessChangeMutex);
    return mask.at<uchar>(point.y, point.x) == WHITE_COLOR;
}

std::string CalibrationParams::sendRequestForMaskPoints() {
    cpr::Response response = cpr::Get(cpr::Url{calibParamsUrl}, cpr::VerifySsl(false), cpr::Timeout(timeout));
    if (response.status_code >= 400 || response.status_code == 0) {
        LOG_ERROR("%s Error [%d] making request for mask", cameraIp.data(), response.status_code);
        return "";
    }
    return response.text;
}

std::vector<cv::Point2i>
CalibrationParams::getPolygonPoints(const string &polygonPointsStr, const string &maskType) const {
    vector<cv::Point2i> polygonPoints;
    if (polygonPointsStr.empty() || polygonPointsStr.length() <= 2) {
        polygonPoints = getDefaultPolygonPoints();
    } else {
        auto polygonPointsJson = json::parse(polygonPointsStr);
        if(polygonPointsJson.empty()){
            polygonPoints = getDefaultPolygonPoints();
        }else{
            for (auto &point: polygonPointsJson[maskType])
                polygonPoints.emplace_back(point["x"].get<int>(), point["y"].get<int>());
            if (polygonPoints.empty())
                polygonPoints = getDefaultPolygonPoints();
        }
    }
    return polygonPoints;
}

std::vector<cv::Point2i> CalibrationParams::getDefaultPolygonPoints() const {
    vector<cv::Point2i> polygonPoints;

    polygonPoints.emplace_back(0, 0);
    polygonPoints.emplace_back(static_cast<int>(FRAME_WIDTH), 0);
    polygonPoints.emplace_back(static_cast<int>(FRAME_WIDTH), static_cast<int>(FRAME_HEIGHT));
    polygonPoints.emplace_back(0, static_cast<int>(FRAME_HEIGHT));

    return polygonPoints;}
