//
// Created by kartykbayev on 5/31/24.
//



#include "CalibrationParamsUpdater.h"

using namespace std;

CalibrationParamsUpdater::CalibrationParamsUpdater(std::shared_ptr<CalibrationParams> calibParams) : ILogger(
        "Calib Params Updater: "), calibParams{std::move(calibParams)} {

}

void CalibrationParamsUpdater::run() {
    backgroundThread = thread(&CalibrationParamsUpdater::periodicallyUpdateMask, this);

}

void CalibrationParamsUpdater::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    shutdownEvent.notify_one();
    if (backgroundThread.joinable())
        backgroundThread.join();
}

void CalibrationParamsUpdater::periodicallyUpdateMask() {
    unique_lock<mutex> lock(updateMaskMutex);
    auto timeout = chrono::minutes(SLEEP_TIME_MINUTES);
    while (!shutdownFlag) {
        if (!shutdownEvent.wait_for(lock, timeout, [this] { return shutdownFlag; })) {
            LOG_INFO("updating mask for %s.....", calibParams->getCameraIp().c_str());
            calibParams->getMask();
        }
    }
}
