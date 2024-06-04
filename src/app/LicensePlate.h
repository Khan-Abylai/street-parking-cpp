#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "Constants.h"
#include "Utils.h"
#include "../ITimer.h"

class LicensePlate : public ITimer {
public:

    LicensePlate(int x, int y, float w, float h, int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4, float prob);

    [[nodiscard]] const cv::Point2i &getCenter() const;

    [[nodiscard]] const cv::Point2f &getLeftTop() const;

    [[nodiscard]] const cv::Point2f &getRightBottom() const;

    [[nodiscard]] const cv::Point2f &getLeftBottom() const;

    [[nodiscard]] const cv::Point2f &getRightTop() const;

    [[nodiscard]] bool isSquare() const;

    [[nodiscard]] float getArea() const;

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] cv::Size getCarImageSize() const;

    [[nodiscard]] const cv::Mat &getPlateImage() const;

    void setPlateImage(const cv::Mat &frame);

    void setFakePlateImage(const cv::Mat &frame);

    const cv::Mat &getFakePlateImage() const;

    [[nodiscard]] const std::string &getPlateLabel() const;

    void setLicensePlateLabel(std::string lpLabel);

    [[nodiscard]] const std::string &getCameraIp() const;

    void setCameraIp(std::string ip);

    [[nodiscard]] const cv::Mat &getCarImage() const;

    void setCarImage(cv::Mat image);

    void setRTPtimestamp(double timestamp);

    [[nodiscard]] double getRTPtimestamp() const;

    void setCarModel(std::string carModelParam);

    [[nodiscard]] const std::string &getCarModel() const;

    void setDirection(std::string direction);

    [[nodiscard]] const std::string &getDirection() const;

    [[nodiscard]] const std::string &getResultSendUrl() const;

    [[nodiscard]] const std::string &getVerificationSendUrl() const;

    void setResultSendUrl(const std::string &url);

    void setVerificationSendUrl(const std::string &url);

    [[nodiscard]] double getSharpness() const ;

    [[nodiscard]] double getDFTSharpness() const ;

    [[maybe_unused]] [[nodiscard]] double getQuality() const ;

    [[nodiscard]] double getWhiteness() const;

    void setRealTimeOfEvent(double time);

    [[nodiscard]] double getRealTimeOfEvent() const;


private:
    static double calculateLaplacian(const cv::Mat &imageCrop);

    static double calculateWhiteScore(const cv::Mat &imageCrop);

    static double calculateQualityMetric(double laplacianValue, double whiteScore);

    static double calculateSharpness(const cv::Mat &licensePlateImg);

    static double calculateDFTSharpness(const cv::Mat &image);

    static double calculateBlurCoefficient(const cv::Mat &image);

    const int CROP_PADDING = 3;
    const float SQUARE_LP_RATIO = 2.4; // 2.6;
    cv::Mat plateImage;
    cv::Mat carImage;

    cv::Mat fakePlateImage;

    std::string licensePlateLabel;
    std::string carModel;
    std::string cameraIp;
    std::string direction;
    std::string cameraPreset;
    std::string resultSendUrl;
    std::string verificationSendUrl;

    double rtpTimestamp{};

    cv::Point2i center;
    cv::Point2f leftTop;
    cv::Point2f leftBottom;
    cv::Point2f rightTop;
    cv::Point2f rightBottom;
    float width, height;
    bool square = false;

    double laplacianValue{};
    double dftValue{};
    double qualityValue{};
    double whitenessValue{};
    double realTimeOfPackage{};


};