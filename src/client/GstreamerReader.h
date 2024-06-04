#pragma once

#include <future>
#include <string>
#include <ctime>
#include <shared_mutex>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cpr/session.h>
#include "opencv2/highgui.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
extern "C" {
#include <gst/gst.h>
#include <gst/app/app.h>
}

#include "../SharedQueue.h"
#include "FrameData.h"
#include "../IThreadLauncher.h"
#include "../ILogger.h"

class GstreamerReader : public ::ILogger {

public:

    GstreamerReader(const std::string &cameraIp, bool useGPUDecode,
                    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue);

    void launchStream();

    void shutdown();

private:
    std::string getAllPresetsURL = "/LAPI/V1.0/Channels/0/PTZ/Presets";
    std::string currentTimeUrl = "/LAPI/V1.0/System/Time";
    std::string presetCtrlURL = "/LAPI/V1.0/Channels/0/PTZ/Presets/<ID>/Goto";

    std::string username="admin";
    std::string password="campas123.";
    std::map<std::string, std::string> presetToPresetURL;
    std::pair<std::string, std::string> currentPresetInfo;

    bool PRESETS_CONFIGURED = false;
    bool READING_FROM_RTSP_ENABLED = false;
    std::thread backgroundThread;
    std::condition_variable updateCurrentPresetEvent;
    std::mutex updateCurrentPresetMutex;
    int CHANGE_PRESET_PERIOD = 150;

    const int REQUEST_TIMEOUT = 5000;
    const double DURATION_BETWEEN_FRAMES_TH = 1.0;
    static constexpr std::chrono::seconds CAMERA_CONNECTION_WAIT_MAX = std::chrono::seconds(60);

    std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>> frameQueue;
    std::string cameraIp;
    std::string rtspUrl;
    bool useGPUDecode = false;
    int frameNumber = 0;
    double lastRtpTimestamp = 0;
    double startTime = 0;

    GstElement *decodingStreamPipeline = nullptr;
    std::atomic<bool> connected = false;

    std::atomic<bool> shutdownFlag = false;
    std::condition_variable shutdownEvent;
    std::mutex shutdownMutex;

    static GstFlowReturn newDecodedSample(GstAppSink *appSink, gpointer data);

    void createStreamDecodingPipeline();

    void closePipeline();

    void startPipeline();

    static double getRTPtimestamp(GstBuffer *buffer);

    static double getRTPtimestamp(GstSample *sample);

    void resetCameraState();

    static bool isKeyFrame(GstSample *sample);

    double getTimeDiffBetweenFrames(double rtpTimestamp);

    bool checkCameraAvailability();

    void fillRequest(cpr::Session &session, const std::string &url) const;

    [[nodiscard]] bool wasRequestSuccessful(const cpr::Response &response) const;

    [[nodiscard]] std::map<std::string, std::string> getAllPresets() const;

    void periodicallySetFrameReading();

    bool changeToPreset(const std::string &url);


};
