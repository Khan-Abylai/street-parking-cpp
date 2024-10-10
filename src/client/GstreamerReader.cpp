#include "GstreamerReader.h"
#include "string"
#include <filesystem>


using namespace std;
using namespace nlohmann;

GstreamerReader::GstreamerReader(const std::string &cameraIp, bool useGPUDecode,
                                 shared_ptr<SharedQueue<unique_ptr<FrameData>>> frameQueue)
        : ILogger("Camera " + cameraIp),
          cameraIp{cameraIp},
          useGPUDecode{useGPUDecode}, frameQueue{std::move(frameQueue)} {
    rtspUrl = "rtsp://admin:campas123.@" + cameraIp + "/media/video1";
    LOG_INFO("RTSP URL: %s ", rtspUrl.c_str());

    this->getAllPresetsURL = "http://" + cameraIp + this->getAllPresetsURL;
    this->currentTimeUrl = "http://" + cameraIp + this->currentTimeUrl;

    auto result = checkCameraAvailability();
    auto presets = getAllPresets();
    if (!presets.empty()) {
        presetToPresetURL = presets;

        PRESETS_CONFIGURED = true;
        CHANGE_PRESET_PERIOD /= (int) presetToPresetURL.size();
        LOG_INFO("All presets configured, number of presets %d. Interval between checking each preset is %d ",
                 presetToPresetURL.size(), CHANGE_PRESET_PERIOD);
        currentPresetInfo = make_pair(presetToPresetURL.begin()->first, presetToPresetURL.begin()->second);
        changeToPreset("http://"+cameraIp+currentPresetInfo.second);
    }

    gst_init(nullptr, nullptr);
    createStreamDecodingPipeline();
    backgroundThread = thread(&GstreamerReader::periodicallySetFrameReading, this);
}

void GstreamerReader::createStreamDecodingPipeline() {
    string protocol = "protocols=tcp";
    string decode = (useGPUDecode) ? "avdec_h264 ! gldownload" : "decodebin";
    auto streamingPipelineString = "rtspsrc " + protocol + " location=" + rtspUrl +
                                   " name=source latency=0 !  rtph264depay ! h264parse ! " + decode +
                                   " ! videoconvert ! video/x-raw, format=(string)BGR !  appsink name=sink emit-signals=true ";

    decodingStreamPipeline = gst_parse_launch(streamingPipelineString.c_str(), nullptr);
    if (!GST_IS_PIPELINE(decodingStreamPipeline))
        LOG_ERROR("rtsp pipeline not initialized");

    auto sink = gst_bin_get_by_name(GST_BIN(decodingStreamPipeline), "sink");
    GstAppSinkCallbacks callbacks = {nullptr, nullptr, newDecodedSample};
    gst_app_sink_set_callbacks(GST_APP_SINK(sink), &callbacks, this, nullptr);
    gst_object_unref(sink);
}

void GstreamerReader::resetCameraState() {
    connected = false;
    frameNumber = 0;
}

void GstreamerReader::closePipeline() {
    if (decodingStreamPipeline) {
        gst_element_set_state(decodingStreamPipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(decodingStreamPipeline));
    }
}

void GstreamerReader::startPipeline() {
    gst_element_set_state(decodingStreamPipeline, GST_STATE_PLAYING);
}

void GstreamerReader::launchStream() {
    auto cameraConnectionWait = 1s;

    while (!shutdownFlag) {
        LOG_INFO("Launching camera");
        startPipeline();

        auto msg = gst_bus_timed_pop_filtered(decodingStreamPipeline->bus, GST_CLOCK_TIME_NONE,
                                              static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (msg) gst_message_unref(msg);

        closePipeline();
        createStreamDecodingPipeline();

        LOG_INFO("Disconnected");
        if (!connected.load()) {
            unique_lock<mutex> shutdownLock(shutdownMutex);

            if (shutdownEvent.wait_for(shutdownLock, cameraConnectionWait,
                                       [this] { return shutdownFlag.load(); }))
                break;

            if (cameraConnectionWait < CAMERA_CONNECTION_WAIT_MAX)
                cameraConnectionWait *= 2;
        } else cameraConnectionWait = 1s;

        resetCameraState();
    }
}

void GstreamerReader::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    frameQueue->push(nullptr);
    if (decodingStreamPipeline) {
        gst_element_set_state(decodingStreamPipeline, GST_STATE_NULL);
        gst_object_unref(GST_OBJECT(decodingStreamPipeline));
    }

    updateCurrentPresetEvent.notify_one();
    if (backgroundThread.joinable())
        backgroundThread.join();

}

GstFlowReturn GstreamerReader::newDecodedSample(GstAppSink *appSink, gpointer data) {
    auto thisPointer = reinterpret_cast<GstreamerReader *>(data);
    GstSample *sample = gst_app_sink_pull_sample(appSink);

    if (thisPointer->frameNumber == 0) {
        thisPointer->LOG_INFO("Start time changed was %f now %ld %f", thisPointer->startTime, time(nullptr),
                              getRTPtimestamp(sample));
        thisPointer->startTime = time(nullptr) - getRTPtimestamp(sample);
    }
    thisPointer->frameNumber++;

    if (!thisPointer->connected.load()) {
        if (isKeyFrame(sample))
            thisPointer->connected = true;
        else {
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
    }

    if (!sample) {
        g_print("Error sample");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *structure = gst_caps_get_structure(caps, 0);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    const int width = g_value_get_int(gst_structure_get_value(structure, "width"));
    const int height = g_value_get_int(gst_structure_get_value(structure, "height"));

    cv::Mat currentFrame(cv::Size(width, height), CV_8UC3, (char *) map.data);

    auto rtpTimestamp = getRTPtimestamp(buffer);
    auto difference = thisPointer->getTimeDiffBetweenFrames(rtpTimestamp);
    if (difference > thisPointer->DURATION_BETWEEN_FRAMES_TH) {
        thisPointer->LOG_INFO("was delay: %lf", difference);
    }

    auto startTime = chrono::high_resolution_clock::now();

    if (!currentFrame.empty() && thisPointer->PRESETS_CONFIGURED && thisPointer->READING_FROM_RTSP_ENABLED) {
        thisPointer->frameQueue->push(
                make_unique<FrameData>(thisPointer->cameraIp,
                                       std::move(currentFrame.clone()),startTime));
        thisPointer->READING_FROM_RTSP_ENABLED = false;
        auto it = thisPointer->presetToPresetURL.find(thisPointer->currentPresetInfo.first);
        if (it != thisPointer->presetToPresetURL.end()) {
            ++it;
            if (it != thisPointer->presetToPresetURL.end())
                thisPointer->currentPresetInfo = make_pair(it->first, it->second);
            else
                thisPointer->currentPresetInfo = make_pair(thisPointer->presetToPresetURL.begin()->first,
                                                           thisPointer->presetToPresetURL.begin()->second);
        } else {
            thisPointer->currentPresetInfo = make_pair(thisPointer->presetToPresetURL.begin()->first,
                                                       thisPointer->presetToPresetURL.begin()->second);
        }

        auto presetChangeResult = thisPointer->changeToPreset(
                "http://" + thisPointer->cameraIp + thisPointer->currentPresetInfo.second);
        if(DEBUG){
            if (presetChangeResult)
                thisPointer->LOG_INFO("Changing preset to the %s is done", thisPointer->currentPresetInfo.first.data());
            else
                thisPointer->LOG_ERROR("Changing preset to the %s is not done",
                                       thisPointer->currentPresetInfo.first.data());
        }
    }
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

double GstreamerReader::getTimeDiffBetweenFrames(double rtpTimestamp) {
    auto difference = rtpTimestamp - lastRtpTimestamp;
    lastRtpTimestamp = rtpTimestamp;
    return difference;
}

double GstreamerReader::getRTPtimestamp(GstBuffer *buffer) {
    return static_cast<double>(GST_BUFFER_PTS (buffer)) / 1000000000;
}

double GstreamerReader::getRTPtimestamp(GstSample *sample) {
    return static_cast<double>(GST_BUFFER_PTS (gst_sample_get_buffer(sample))) / 1000000000;
}

bool GstreamerReader::isKeyFrame(GstSample *sample) {
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    return !GST_BUFFER_FLAG_IS_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);
}

bool GstreamerReader::checkCameraAvailability() {
    LOG_INFO("Checking camera availability");
    cpr::Session session;
    fillRequest(session, currentTimeUrl);
    auto response = session.Get();

    if (response.elapsed >= REQUEST_TIMEOUT / 1000 || response.status_code == 0) {
        LOG_ERROR("Camera is not available");
        return false;
    }

    cpr::Session sessionNew;
    fillRequest(sessionNew, currentTimeUrl);
    auto responseNew = sessionNew.Get();
    if (responseNew.elapsed >= REQUEST_TIMEOUT / 1000 || responseNew.status_code == 0 ||
        responseNew.status_code == 401) {
        LOG_ERROR("Camera is not available");
        return false;
    }

    LOG_INFO("Camera is available");
    return true;
}

void GstreamerReader::fillRequest(cpr::Session &session, const string &url) const {
    session.SetUrl(url);
    session.SetVerifySsl(cpr::VerifySsl(false));
    session.SetTimeout(cpr::Timeout(REQUEST_TIMEOUT));
    session.SetAuth(cpr::Authentication(username, password, cpr::AuthMode::DIGEST));
}

bool GstreamerReader::wasRequestSuccessful(const cpr::Response &response) const {
    bool isSuccess = false;
    if (response.status_code >= 200 && response.status_code < 300) {
        try {
            auto responseBody = json::parse(response.text);
            if (responseBody.contains("Response") && responseBody["Response"].contains("ResponseCode")) {
                isSuccess = (responseBody["Response"]["ResponseCode"].get<int>() == 0);
            }
        } catch (std::exception e) {
            LOG_ERROR("Cannot parse response from camera API, %s", e.what());
            return false;
        }
    }
    return !(response.elapsed > REQUEST_TIMEOUT / 1000 || response.status_code < 200 ||
             response.status_code > 299 || !isSuccess);
}

std::map<std::string, std::string> GstreamerReader::getAllPresets() const {
    cpr::Session session;
    fillRequest(session, getAllPresetsURL);
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

void GstreamerReader::periodicallySetFrameReading() {
    unique_lock<mutex> lock(updateCurrentPresetMutex);
    auto timeout = chrono::seconds(CHANGE_PRESET_PERIOD);
    while (!shutdownFlag) {
        if (!updateCurrentPresetEvent.wait_for(lock, timeout, [this] { return shutdownFlag.load(); })) {
            if(DEBUG)
                LOG_INFO("provide reading from rtsp stream.....");
            READING_FROM_RTSP_ENABLED = true;
        }
    }
}

bool GstreamerReader::changeToPreset(const string &url) {
    cpr::Session putSession;
    fillRequest(putSession, url);
    putSession.SetHeader(cpr::Header{{"Content-Type", "application/json"}});
    return wasRequestSuccessful(putSession.Put());
}
