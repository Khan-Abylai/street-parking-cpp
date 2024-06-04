//
// Created by kartykbayev on 5/31/24.

#pragma once
#include <utility>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "CalibrationParams.h"

class CalibrationParamsUpdater: public IThreadLauncher, public ILogger{

public:
    explicit CalibrationParamsUpdater(std::shared_ptr<CalibrationParams> calibParams);

    void run() override;

    void shutdown() override;
private:
    const int SLEEP_TIME_MINUTES = 5;

    std::condition_variable shutdownEvent;
    std::mutex updateMaskMutex;

    std::shared_ptr<CalibrationParams> calibParams;
    std::thread backgroundThread;

    void periodicallyUpdateMask();
};


