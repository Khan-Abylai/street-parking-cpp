#include "FrameData.h"

FrameData::FrameData(std::string ip, std::string presetID, cv::Mat frame,
                     std::chrono::high_resolution_clock::time_point startTime) :
        ip{std::move(ip)}, presetID{std::move(presetID)}, frame{std::move(frame)} {
    setStartTime(startTime);

}

const std::string &FrameData::getPresetID() {
    return presetID;
}

const std::string &FrameData::getIp() {
    return ip;
}

const cv::Mat &FrameData::getFrame() {
    return frame;
}


float FrameData::getFrameWidth() const {
    return (float) frame.cols;
}

float FrameData::getFrameHeight() const {
    return (float) frame.rows;
}
