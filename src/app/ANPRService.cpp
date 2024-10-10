//
// Created by kartykbayev on 5/29/24.
//

#include "ANPRService.h"

using namespace std;

ANPRService::ANPRService(std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue,
                         std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                         std::shared_ptr<Detection> detection,
                         const float &recognizerThreshold,
                         std::string cameraIp, const std::string& nodeIp, float calibrationWidth, float calibrationHeight) : ILogger("ANPR Service"),
                         detection{std::move(detection)}, RECOGNIZER_PROB_THRESHOLD{recognizerThreshold}, frameQueue{std::move(frameQueue)},
                          packageQueue{std::move(packageQueue)}, cameraIP{std::move(cameraIp)} {
    LOG_INFO("Initialize Detection Service for camera %s", cameraIP.data());
    // this->recognizer = make_unique<LPRecognizer>();
    this->recognizer_ext = make_unique<LPRecognizerExtended>();
    this->templateMatching = make_unique<TemplateMatching>();

    calibParams = make_shared<CalibrationParams>(nodeIp, cameraIP, calibrationWidth, calibrationHeight);
    calibParamsUpdater = make_unique<CalibrationParamsUpdater>(calibParams);
    calibParamsUpdater->run();
}

void ANPRService::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
}

void ANPRService::run() {
    while (!shutdownFlag) {
        auto frameData = frameQueue->wait_and_pop();
        if (frameData == nullptr) continue;
        auto frame = frameData->getFrame();

        auto startTime = chrono::high_resolution_clock::now();
        auto detectionResult = detection->detect(frame);
        auto endTime = chrono::high_resolution_clock::now();
        double execTime = (double) chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
        LOG_INFO("received frame, size:[%d, %d]. overall found plates %d, exec time [%f ms]", frame.cols, frame.rows, detectionResult.size(), execTime);

        if (detectionResult.empty()) continue;

        std::vector<std::string> licensePlateLabels;
        std::vector<std::string> licensePlateBBoxes;
        for (const auto &lp:detectionResult) {
            lp->setCarImage(frame);

            if(!calibParams->isLicensePlateInSelectedArea(lp))
                continue;
            lp->setPlateImage(frame);

            vector<cv::Mat> lpImages = getLicensePlateImages(lp);
            auto rec_startTime = chrono::high_resolution_clock::now();
            // auto recognizerResult = recognizer->predict(lpImages);
            auto recognizerResult = recognizer_ext->makePrediction(lpImages);
            vector<std::pair<std::string, float>> lp_vector;
            lp_vector.emplace_back(make_pair(std::get<0>(recognizerResult), std::get<1>(recognizerResult)));
            auto rec_endTime = chrono::high_resolution_clock::now();
            double rec_execTime = (double) chrono::duration_cast<chrono::milliseconds>(rec_endTime - rec_startTime).count();
            auto [licensePlateLabel, probability] = getLicensePlateLabel(lp_vector, lp->isSquare());
            bool isValid = isValidLicensePlate(licensePlateLabel, probability);

            if(!isValid && lp->isSquare()){
                lp->setFakePlateImage(frame.clone());
                vector<cv::Mat> fakeLPImages = {lp->getFakePlateImage()};
                // auto fakeRecognizerResult = recognizer->predict(fakeLPImages);
                auto fakeRecognizerResult = recognizer_ext->makePrediction(lpImages);

                vector<std::pair<std::string, float>> fake_lp_vector;
                fake_lp_vector.emplace_back(make_pair(std::get<0>(recognizerResult), std::get<1>(recognizerResult)));
                auto [fakeLabel, fakeProb] = getLicensePlateLabel(fake_lp_vector, false);
                bool newIsValid = isValidLicensePlate(fakeLabel, fakeProb);
                if(!newIsValid)
                    continue;
                licensePlateLabel = fakeLabel;
                probability = fakeProb;
                isValid = newIsValid;
            }

            if(!isValid)
                continue;
            if(DEBUG)
                LOG_INFO(" lp: %s, prob: %f, exec time: %f",  licensePlateLabel.data(), probability, rec_execTime);
            auto bboxes = convertBoundingBoxToStr(lp);
            licensePlateLabels.emplace_back(licensePlateLabel);
            licensePlateBBoxes.emplace_back(bboxes);
        }
//        if(DEBUG)
            // saveImage(frame, "1", Utils::dateTimeToStrAnother(time_t(nullptr)), frameData->getIp());
        if(!licensePlateLabels.empty()){
            createAndPushEventVerification(licensePlateLabels, frameData->getIp(), frame, licensePlateBBoxes);
        }

    }
}

std::vector<cv::Mat> ANPRService::getLicensePlateImages(const shared_ptr<LicensePlate> &licensePlate) {
    vector<cv::Mat> lpImages;
    if (licensePlate->isSquare()) {
        auto combinedImage = combineInOneLineSquareLpPlate(licensePlate->getPlateImage());
        lpImages.push_back(std::move(combinedImage));
    } else {
        lpImages.push_back(licensePlate->getPlateImage());
    }
    return lpImages;}

cv::Mat ANPRService::combineInOneLineSquareLpPlate(const cv::Mat &lpImage) {
    // auto blackImage = cv::Mat(Constants::STANDARD_RECT_LP_H, Constants::BLACK_IMG_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    auto topHalf = lpImage(cv::Rect(0, 0, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2));
    auto bottomHalf = lpImage(
            cv::Rect(0, Constants::SQUARE_LP_H / 2, Constants::SQUARE_LP_W, Constants::SQUARE_LP_H / 2));

    cv::Mat combinedPlateImage;
    // cv::hconcat(topHalf, blackImage, topHalf);
    cv::hconcat(topHalf, bottomHalf, combinedPlateImage);
    return combinedPlateImage;}

std::pair<std::string, float>
ANPRService::getLicensePlateLabel(const vector<std::pair<std::string, float>> &recognizerResult, bool isSquarePlate) {
    float probability;
    string licensePlateLabel;
    if (isSquarePlate) {
        string label = recognizerResult.front().first;
        string sep = ".";
        vector<string> parts = string_split(label, '.');

        if (parts.size() == 2) {
            licensePlateLabel = templateMatching->processSquareLicensePlate(parts[0], parts[1]);
        } else {
            licensePlateLabel = label;
        }
        probability = recognizerResult.front().second;
    } else {
        licensePlateLabel = recognizerResult.front().first;
        probability = recognizerResult.front().second;
    }
    return make_pair(licensePlateLabel, probability);}

void ANPRService::saveFrame(const shared_ptr<LicensePlate> &plate) {
    string fileName = RandomStringGenerator::generate(30, Constants::IMAGE_DIRECTORY, Constants::JPG_EXTENSION);
    auto frame = plate->getCarImage();
    cv::imwrite(fileName, frame);
}

bool ANPRService::isValidLicensePlate(const string &lpLabel, float probability) {
    auto plateCountry = templateMatching->getCountryCode(lpLabel);
    auto isTemplateMatched = plateCountry != Constants::UNIDENTIFIED_COUNTRY;
    return probability > RECOGNIZER_PROB_THRESHOLD; //&& isTemplateMatched;
}

void
ANPRService::createAndPushEventVerification(const std::vector<std::string>& licensePlateLabels, const string &cameraIp,
                                            const cv::Mat &frame, const std::vector<std::string> &licensePlateBBoxes) {
    auto package = make_shared<Package>(cameraIp,  licensePlateLabels, frame, licensePlateBBoxes);
    packageQueue->push(std::move(package));
}

std::string ANPRService::convertBoundingBoxToStr(const shared_ptr<LicensePlate> &licensePlate) {
    auto frameSize = licensePlate->getCarImageSize();
    auto frameWidth = (float) frameSize.width;
    auto frameHeight = (float) frameSize.height;
    return Utils::pointToStr(licensePlate->getLeftTop().x / frameWidth,
                             licensePlate->getLeftTop().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getLeftBottom().x / frameWidth,
                             licensePlate->getLeftBottom().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getRightTop().x / frameWidth,
                             licensePlate->getRightTop().y / frameHeight) + ", " +
           Utils::pointToStr(licensePlate->getRightBottom().x / frameWidth,
                             licensePlate->getRightBottom().y / frameHeight);}
