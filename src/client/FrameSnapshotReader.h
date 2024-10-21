//
// Created by kartykbayev on 5/27/24.
//

#pragma once

#include <future>
#include <string>
#include <ctime>
#include <shared_mutex>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "cpr/cpr.h"
#include <utility>
#include <nlohmann/json.hpp>
#include <openssl/md5.h>

#include "../SharedQueue.h"
#include "FrameData.h"
#include "../IThreadLauncher.h"
#include "../ILogger.h"

class FrameSnapshotReader:public ILogger {

public:
    FrameSnapshotReader(const std::string &cameraIP, std::string username, std::string password,
                        std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>>& frameQueue, int eventInterval,
                        const std::string &cameraSnapshotUrl, const std::string &cameraTimeUrl);

    void launchStream();

    void shutdown();

    bool checkCameraAvailability();

private:
    const int REQUEST_TIMEOUT = 5000;
    int OVERALL_SECONDS_TO_CHANGE = 60;
    bool PRESETS_CONFIGURED = false;

    std::atomic<bool> shutdownFlag = false;
    std::condition_variable shutdownEvent;
    std::mutex shutdownMutex;
    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;

    std::string cameraIp, username, password;
    std::string currentTimeUrl = "/LAPI/V1.0/System/Time";
    std::string snapshotUrl = "/LAPI/V1.0/Channels/0/Media/Video/Streams/0/Snapshot";

    void fillRequest(cpr::Session &session, const std::string &url, cpr::AuthMode authMode) const;

    [[nodiscard]] bool wasRequestSuccessful(const cpr::Response &response) const;

    [[nodiscard]] cv::Mat snapshotGetter() const;


};


