//
// Created by kartykbayev on 5/27/24.
//

#include "FrameSnapshotReader.h"

using namespace std;
using namespace nlohmann;


FrameSnapshotReader::FrameSnapshotReader(const std::string &cameraIP, std::string username,
                                         std::string password,
                                         std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> &frameQueue, int eventInterval,
                                         const std::string &cameraSnapshotUrl, const std::string &cameraTimeUrl) :
        ILogger("Camera " + cameraIP), username{std::move(username)}, password{std::move(password)}, cameraIp{cameraIP}, frameQueue{std::move(frameQueue)}, OVERALL_SECONDS_TO_CHANGE{eventInterval} {
    this->currentTimeUrl = "http://" + cameraIP + cameraTimeUrl;
    this->snapshotUrl = "http://" + cameraIP + cameraSnapshotUrl;

    auto result = checkCameraAvailability();
}

bool FrameSnapshotReader::checkCameraAvailability() {
    LOG_INFO("Checking camera availability");
    cpr::Session session;
    fillRequest(session, currentTimeUrl, cpr::AuthMode::DIGEST);
    auto response = session.Get();

    if (response.elapsed >= REQUEST_TIMEOUT / 1000 || response.status_code == 0) {
        LOG_ERROR("Camera is not available");
        return false;
    }

    cpr::Session sessionNew;
    fillRequest(sessionNew, currentTimeUrl, cpr::AuthMode::DIGEST);
    auto responseNew = sessionNew.Get();
    if (responseNew.elapsed >= REQUEST_TIMEOUT / 1000 || responseNew.status_code == 0 ||
        responseNew.status_code == 401) {
        LOG_ERROR("Camera is not available");
        return false;
    }

    LOG_INFO("Camera is available");
    return true;
}

void FrameSnapshotReader::fillRequest(cpr::Session &session, const std::string &url, cpr::AuthMode authMode) const {
    session.SetUrl(url);
    session.SetVerifySsl(cpr::VerifySsl(false));
    session.SetTimeout(cpr::Timeout(REQUEST_TIMEOUT));
    session.SetAuth(cpr::Authentication(username, password, authMode));
}

bool FrameSnapshotReader::wasRequestSuccessful(const cpr::Response &response) const {
    bool isSuccess = false;
    if (response.status_code >= 200 && response.status_code < 300) {
        try {
            auto responseBody = json::parse(response.text);
            if (responseBody.contains("Response") && responseBody["Response"].contains("ResponseCode")) {
                isSuccess = (responseBody["Response"]["ResponseCode"].get<int>() == 0);
            }
        } catch (exception e) {
            LOG_ERROR("Cannot parse response from camera API, %s", e.what());
            return false;
        }
    }
    return !(response.elapsed > REQUEST_TIMEOUT / 1000 || response.status_code < 200 ||
             response.status_code > 299 || !isSuccess);
}

void FrameSnapshotReader::launchStream() {

    while (!shutdownFlag) {
            unique_lock<mutex> lock(shutdownMutex);
            auto timeout = chrono::seconds(OVERALL_SECONDS_TO_CHANGE);
            if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag.load(); })) {
                auto snapshot = snapshotGetter();
                auto startTime = chrono::high_resolution_clock::now();
                frameQueue->push(make_unique<FrameData>(cameraIp, std::move(snapshot.clone()), startTime));
            }
    }
}

void FrameSnapshotReader::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
    shutdownEvent.notify_one();
}

cv::Mat FrameSnapshotReader::snapshotGetter() const {
    cpr::Session session;
    session.SetUrl(cpr::Url(this->snapshotUrl));
    session.SetVerifySsl(cpr::VerifySsl(false));
    session.SetTimeout(cpr::Timeout(REQUEST_TIMEOUT));
    session.SetAuth(cpr::Authentication(username,password, cpr::AuthMode::DIGEST));

    auto response = session.Get();
    if (response.elapsed >= REQUEST_TIMEOUT / 1000 || response.status_code == 0 || response.status_code == 401) {
        LOG_ERROR("Problem with getting snapshot from camera");
        return cv::Mat();
    }

    std::vector<uchar> data(response.text.begin(), response.text.end());
    cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);

    if (image.empty()) {
        LOG_ERROR("Failed to decode image from response" );
        return cv::Mat();
    }
    return image;
}
