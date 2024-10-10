//
// Created by kartykbayev on 5/29/24.
//
#pragma once
#include <utility>
#include <filesystem>
#include <sys/stat.h>
#include "../RandomStringGenerator.h"
#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../client/FrameData.h"
#include "CalibrationParams.h"
#include "CalibrationParamsUpdater.h"
#include "../package_sending/Package.h"
#include "../app/Detection.h"
#include "../app/TemplateMatching.h"
#include "../app/LPRecognizer.h"
#include "../app/LPRecognizerExtended.h"

class ANPRService: public IThreadLauncher, public ILogger{
public:
    ANPRService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                std::shared_ptr<Detection> detection,
                const float &recognizerThreshold,
                std::string cameraIp,
                const std::string& nodeIp, float calibrationWidth, float calibrationHeight);

    void run() override;

    void shutdown() override;
private:
    const float RECOGNIZER_PROB_THRESHOLD = 0.5;

    std::string cameraIP;
    std::shared_ptr<Detection> detection;
    // std::unique_ptr<LPRecognizer> recognizer;
    std::unique_ptr<LPRecognizerExtended> recognizer_ext;
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


    static bool createDirectoryIfNotExists(const std::string& path) {
        struct stat info{};

        if (stat(path.c_str(), &info) != 0) {
            // Directory does not exist, try to create it
            return std::filesystem::create_directory(path);
        } else if (info.st_mode & S_IFDIR) {
            // Directory exists
            return true;
        } else {
            // Path exists but is not a directory
            return false;
        }
    }

    static void saveImage(const cv::Mat& image, const std::string& preset_id, const std::string& event_time, const std::string& ip_address) {
        std::string folder_path = Constants::IMAGE_DIRECTORY + preset_id;
        if (createDirectoryIfNotExists(folder_path)) {
            std::string file_name = event_time + "-" + preset_id + "-" + ip_address + ".jpeg";
            std::string full_path = folder_path + "/" + file_name;

            // Save the image
            if (cv::imwrite(full_path, image)) {
                std::cout << "Image saved successfully at " << full_path << std::endl;
            } else {
                std::cerr << "Error saving image at " << full_path << std::endl;
            }
        } else {
            std::cerr << "Error creating directory " << folder_path << std::endl;
        }
    }

    static void saveFrame(const std::shared_ptr<LicensePlate> &plate);
    void createAndPushEventVerification(const std::vector<std::string>& licensePlateLabels, const std::string &cameraIp, const cv::Mat &frame, const std::vector<std::string> &licensePlateBBoxes);
    static std::string convertBoundingBoxToStr(const std::shared_ptr<LicensePlate> &licensePlate);
};


