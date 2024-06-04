//
// Created by kartykbayev on 5/29/24.
//
#pragma once
#include <utility>

#include "../RandomStringGenerator.h"
#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../client/FrameData.h"
#include "DetectorYoloV5NCNN.h"
#include "LPRecognizerNCNN.h"
#include "CalibrationParams.h"
#include "CalibrationParamsUpdater.h"
#include "../package_sending/Package.h"

class ANPRService: public IThreadLauncher, public ILogger{
public:
    ANPRService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue, std::string cameraIp,
                const std::string& nodeIp, float calibrationWidth, float calibrationHeight);

    void run() override;

    void shutdown() override;
private:
    const float RECOGNIZER_PROB_THRESHOLD = 0.70;

    std::string cameraIP;
    std::unique_ptr<DetectorYoloV5NCNN> detectionNCNN;
    std::unique_ptr<LPRecognizerNCNN> recognizerNCNN;
    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;
    std::unique_ptr<TemplateMatching> templateMatching;
    std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue;


    bool isValidLicensePlate(const std::string &lpLabel, float probability);

    static std::vector<cv::Mat> getLicensePlateImages(const std::shared_ptr<LicensePlate> &licensePlate);
    static cv::Mat combineInOneLineSquareLpPlate(const cv::Mat &lpImage);
    std::pair<std::string, float>
    getLicensePlateLabel(const std::vector<std::pair<std::string, float>> &recognizerResult, bool isSquarePlate);

    std::shared_ptr<CalibrationParams> calibParams;
    std::unique_ptr<CalibrationParamsUpdater> calibParamsUpdater;

    static std::vector<std::string> string_split(const std::string &s, const char delimiter) {
        size_t start = 0;
        size_t end = s.find_first_of(delimiter);

        std::vector<std::string> output;

        while (end <= std::string::npos) {
            output.emplace_back(s.substr(start, end - start));

            if (end == std::string::npos)
                break;

            start = end + 1;
            end = s.find_first_of(delimiter, start);
        }

        return output;
    }

    static void saveFrame(const std::shared_ptr<LicensePlate> &plate);
    void createAndPushEventVerification(const std::vector<std::string>& licensePlateLabels, const std::string &cameraIp, const std::string &presetId, const cv::Mat &frame, const std::vector<std::string> &licensePlateBBoxes);
    static std::string convertBoundingBoxToStr(const std::shared_ptr<LicensePlate> &licensePlate);
};


