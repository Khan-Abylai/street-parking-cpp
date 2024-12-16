#include "LicensePlate.h"

#include <utility>

using namespace std;

LicensePlate::LicensePlate(int x, int y, float w, float h, int x1, int y1,
                           int x2, int y2, int x3, int y3, int x4, int y4, float prob) {

    width = w;
    height = h;
    center = cv::Point2i(x, y);
    leftTop = cv::Point2f(x1, y1);
    leftBottom = cv::Point2f(x2, y2);
    rightTop = cv::Point2f(x3, y3);
    rightBottom = cv::Point2f(x4, y4);

    if (width / height < SQUARE_LP_RATIO) {
        square = true;
    } else {
        square = false;
    }

    setCarModel("NotDefined");
}

float LicensePlate::getArea() const {
    return (float) width * (float) height;
}

void LicensePlate::setFakePlateImage(const cv::Mat &frame) {
    cv::Mat transformationMatrix;
    cv::Size lpSize;
    transformationMatrix = cv::getPerspectiveTransform(vector<cv::Point2f>{
            leftTop, leftBottom, rightTop, rightBottom
    }, Constants::RECT_LP_COORS);

    lpSize = cv::Size(Constants::STANDARD_RECT_LP_W, Constants::STANDARD_RECT_LP_H);
    cv::warpPerspective(frame, fakePlateImage, transformationMatrix, lpSize);
}

void LicensePlate::move(int dx, int dy) {
    center.x += dx;
    center.y += dy;
    leftTop.x += dx; leftTop.y += dy;
    rightBottom.x += dx; rightBottom.y += dy;
    leftBottom.x += dx; leftBottom.y += dy;
    rightTop.x += dx; rightTop.y += dy;
}

void LicensePlate::setPlateImage(const cv::Mat &frame) {

    cv::Mat transformationMatrix;
    cv::Size lpSize;

    if (square) {
        transformationMatrix = cv::getPerspectiveTransform(vector<cv::Point2f>{
                leftTop, leftBottom, rightTop, rightBottom
        }, Constants::SQUARE_LP_COORS);

        lpSize = cv::Size(Constants::SQUARE_LP_W, Constants::SQUARE_LP_H);

    } else {
        transformationMatrix = cv::getPerspectiveTransform(vector<cv::Point2f>{
                leftTop, leftBottom, rightTop, rightBottom
        }, Constants::RECT_LP_COORS);

        lpSize = cv::Size(Constants::STANDARD_RECT_LP_W, Constants::STANDARD_RECT_LP_H);
    }
    cv::warpPerspective(frame, plateImage, transformationMatrix, lpSize);


    auto imageCrop = cv::Mat(plateImage.clone(),
                             cv::Rect(CROP_PADDING, CROP_PADDING, plateImage.cols - 2 * CROP_PADDING,
                                      plateImage.rows - 2 * CROP_PADDING));

//    laplacianValue = calculateLaplacian(imageCrop);
    laplacianValue = calculateBlurCoefficient(imageCrop);
    whitenessValue = calculateWhiteScore(imageCrop);
//    qualityValue = calculateQualityMetric(laplacianValue, whitenessValue);
    dftValue = calculateSharpness(imageCrop);
//    dftValue = calculateDFTSharpness(imageCrop);
}

const cv::Mat &LicensePlate::getPlateImage() const {
    return plateImage;
}

const cv::Point2i &LicensePlate::getCenter() const {
    return center;
}

const cv::Point2f &LicensePlate::getLeftTop() const {
    return leftTop;
}

const cv::Point2f &LicensePlate::getRightBottom() const {
    return rightBottom;
}

int LicensePlate::getWidth() const {
    return (int) width;
}

int LicensePlate::getHeight() const {
    return (int)height;
}

const cv::Point2f &LicensePlate::getLeftBottom() const {
    return leftBottom;
}

const cv::Point2f &LicensePlate::getRightTop() const {
    return rightTop;
}

bool LicensePlate::isSquare() const {
    return square;
}

const string &LicensePlate::getPlateLabel() const {
    return licensePlateLabel;
}

void LicensePlate::setLicensePlateLabel(string lpLabel) {
    licensePlateLabel = std::move(lpLabel);
}

void LicensePlate::setCameraIp(string ip) {
    cameraIp = std::move(ip);
}

const string &LicensePlate::getCameraIp() const {
    return cameraIp;
}

void LicensePlate::setCarImage(cv::Mat image) {
    carImage = std::move(image);
}

const cv::Mat &LicensePlate::getCarImage() const {
    return carImage;
}

void LicensePlate::setRTPtimestamp(double timestamp) {
    rtpTimestamp = timestamp;
}

double LicensePlate::getRTPtimestamp() const {
    return rtpTimestamp;
}

cv::Size LicensePlate::getCarImageSize() const {
    return carImage.size();
}


const string &LicensePlate::getDirection() const {
    return direction;
}

void LicensePlate::setDirection(std::string directions) {
    direction = std::move(directions);
}

const std::string &LicensePlate::getResultSendUrl() const {
    return resultSendUrl;
}

const std::string &LicensePlate::getVerificationSendUrl() const {
    return verificationSendUrl;
}

void LicensePlate::setResultSendUrl(const std::string &url) {
    resultSendUrl = url;
}

void LicensePlate::setVerificationSendUrl(const std::string &url) {
    verificationSendUrl = url;
}

double LicensePlate::calculateLaplacian(const cv::Mat &imageCrop) {
    cv::Mat laplacianImg;
    cv::Laplacian(imageCrop, laplacianImg, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImg, mean, stddev, cv::Mat());
    return (stddev.val[0] * stddev.val[0]);
}

double LicensePlate::calculateWhiteScore(const cv::Mat &imageCrop) {
    cv::Mat blurImg;
    cv::Mat otsuImg;
    cv::GaussianBlur(imageCrop, blurImg, {5, 5}, 0);
    cv::cvtColor(blurImg, blurImg, cv::COLOR_RGB2GRAY);
    cv::threshold(blurImg, otsuImg, 0, 255, cv::THRESH_OTSU);
    int whitePixels = cv::countNonZero(otsuImg);
    int totalPixels = imageCrop.rows * imageCrop.cols;
    auto whitePercentage = (whitePixels * 100 / totalPixels);
    return whitePercentage;
}

double LicensePlate::calculateQualityMetric(double laplacianValue, double whiteScore) {
    auto whiteValue = 0;
    if (whiteScore > 50) {
        whiteValue = abs((whiteScore / 50) - 2);
    } else {
        whiteValue = whiteScore / 50;
    }
    return whiteValue * laplacianValue;
}

double LicensePlate::calculateSharpness(const cv::Mat &licensePlateImg) {
    cv::Mat imageFloat;
    cv::cvtColor(licensePlateImg, imageFloat, cv::COLOR_BGR2GRAY);
    imageFloat = cv::Mat_<float>(imageFloat);
    cv::Scalar DC = mean(imageFloat);
    subtract(imageFloat, DC[0], imageFloat);

    //expand input image to optimal size
    int m = cv::getOptimalDFTSize(imageFloat.rows);
    int n = cv::getOptimalDFTSize(imageFloat.cols); // on the border add zero values
    cv::copyMakeBorder(imageFloat, imageFloat, 0, m - imageFloat.rows, 0,
                       n - imageFloat.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(imageFloat), cv::Mat::zeros(imageFloat.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix
    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude

    cv::Mat magI = planes[0];
    double M; // the maximum absolute value
    cv::minMaxIdx(magI, NULL, &M, NULL, NULL);

    double thresh = M / 1000.0;
    threshold(magI, magI, thresh, 1, cv::THRESH_BINARY);
    cv::Scalar TH = sum(magI);
    cv::Scalar FM = TH / ((double) m * n);
    return FM[0];// This is the final value
}


double LicensePlate::getSharpness() const {
    return laplacianValue;
}

double LicensePlate::getDFTSharpness() const {
    return dftValue;
}

[[maybe_unused]] double LicensePlate::getQuality() const {
    return qualityValue;
}

double LicensePlate::getWhiteness() const {
    return whitenessValue;
}

double LicensePlate::calculateDFTSharpness(const cv::Mat &image) {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    grayImage.convertTo(grayImage, CV_32FC1); // Convert to

    cv::Mat dftImage;
    cv::dft(grayImage, dftImage, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat magnitude;
    cv::magnitude(dftImage, dftImage, magnitude);

    double totalEnergy = cv::sum(magnitude.mul(magnitude))[0];

    // Calculate the sharpness metric by focusing on the high-frequency energy
    // You might need to adjust the threshold based on your image content
    double highFrequencyEnergy = totalEnergy * 0.1; // Example threshold

    double sharpness = 1.0 - (1.0 / (1.0 + highFrequencyEnergy));
    return sharpness;
}

double LicensePlate::calculateBlurCoefficient(const cv::Mat &image) {
    cv::Mat laplacianImage;
    cv::Laplacian(image, laplacianImage, CV_64F);

    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImage, mean, stddev);

    double blurCoefficient = stddev.val[0];
    return blurCoefficient;
}

void LicensePlate::setCarModel(std::string carModelParam) {
    this->carModel = std::move(carModelParam);
}

const std::string &LicensePlate::getCarModel() const {
    return this->carModel;
}


double LicensePlate::getRealTimeOfEvent() const {
    return this->realTimeOfPackage;
}

void LicensePlate::setRealTimeOfEvent(double time) {
    this->realTimeOfPackage = time;
}

const cv::Mat &LicensePlate::getFakePlateImage() const {
    return fakePlateImage;
}






