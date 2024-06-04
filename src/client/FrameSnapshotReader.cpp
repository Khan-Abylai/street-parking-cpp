//
// Created by kartykbayev on 5/27/24.
//

#include "FrameSnapshotReader.h"

using namespace std;
using namespace nlohmann;


FrameSnapshotReader::FrameSnapshotReader(const std::string &cameraIP, std::string username,
                                         std::string password,
                                         const std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> &frameQueue) :
        ILogger("Camera " + cameraIP), username{std::move(username)}, password{std::move(password)}, cameraIp{cameraIP} {
    this->getAllPresetsURL = "http://" + cameraIP + this->getAllPresetsURL;
    this->currentTimeUrl = "http://" + cameraIP + this->currentTimeUrl;
    this->snapshotUrl = "http://" + cameraIP + this->snapshotUrl;

    auto result = checkCameraAvailability();
    auto presets = getAllPresets();
    if (!presets.empty()) {
        presetToPresetURL = presets;
        PRESETS_CONFIGURED = true;
    }
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

    int CHECK_CURRENT_CAR_SECONDS = 60 / presetToPresetURL.size();
    pair<string, string> currentPresetInfo = !presetToPresetURL.empty() ? make_pair(presetToPresetURL.begin()->first, presetToPresetURL.begin()->second) : make_pair("", "");
    while (!shutdownFlag) {
        while (PRESETS_CONFIGURED) {
            unique_lock<mutex> lock(shutdownMutex);
            auto timeout = chrono::seconds(CHECK_CURRENT_CAR_SECONDS);

            if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag.load(); })) {
                LOG_INFO("Taking snapshots every %d seconds", CHECK_CURRENT_CAR_SECONDS);
                LOG_INFO("Now we are working with %s preset",currentPresetInfo.first.data());

                auto snapshot = snapshotGetter();
                LOG_INFO("image with size :%d %d", snapshot.cols, snapshot.rows);

                LOG_INFO("moving to another preset...");
                auto presetChanged = changeToPreset("http://"+cameraIp + currentPresetInfo.second);
                LOG_INFO("preset changed %d", presetChanged);

                auto it = presetToPresetURL.find(currentPresetInfo.first);
                if(it!= presetToPresetURL.end()){
                    ++it;
                    if(it != presetToPresetURL.end())
                        currentPresetInfo = make_pair(it->first, it->second);
                    else
                        currentPresetInfo = make_pair(presetToPresetURL.begin()->first, presetToPresetURL.begin()->second);
                }else{
                    currentPresetInfo = make_pair(presetToPresetURL.begin()->first, presetToPresetURL.begin()->second);
                }
            }
        }
    }
}

void FrameSnapshotReader::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
    shutdownEvent.notify_one();
}

std::map<std::string, std::string> FrameSnapshotReader::getAllPresets() const {
    cpr::Session session;
    fillRequest(session, getAllPresetsURL, cpr::AuthMode::DIGEST);
    auto response = session.Get();


    if (!wasRequestSuccessful(response)) {
        return {};
    }
    auto presets = json::parse(response.text)["Response"]["Data"];

    if (presets["Nums"].get<int>() == 0) {
        LOG_INFO("Number of presets or presets are empty");
        return {};
    }

    auto presetNums = presets["Nums"].get<int>();
    auto presetsInfos = presets["PresetInfos"];

    std::map<std::string, std::string> preset2url;

    for (int i = 0; i < presetNums; ++i) {
        auto presetInfo = presetsInfos[i];
        std::string presetID = to_string(presetInfo["ID"].get<int>());
        auto presetName = presetInfo["Name"].get<string>();
        auto newPresetUrl(presetCtrlURL);
        newPresetUrl.replace(newPresetUrl.find("<ID>"), 4, presetID);
        preset2url.insert(make_pair(presetName, newPresetUrl));
    }
    return preset2url;
}

cv::Mat FrameSnapshotReader::snapshotGetter() const {
    cpr::Session session;
    session.SetUrl(cpr::Url("http://10.66.117.40/LAPI/V1.0/Channels/0/Media/Video/Streams/0/Snapshot"));
    session.SetVerifySsl(cpr::VerifySsl(false));
    session.SetTimeout(cpr::Timeout(1000));
    session.SetAuth(cpr::Authentication("admin","campas123.", cpr::AuthMode::DIGEST));

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

bool FrameSnapshotReader::changeToPreset(const std::string &url) {
    cpr::Session putSession;
    fillRequest(putSession, url, cpr::AuthMode::DIGEST);
    putSession.SetHeader(cpr::Header{{"Content-Type", "application/json"}});
    return wasRequestSuccessful(putSession.Put());
}
