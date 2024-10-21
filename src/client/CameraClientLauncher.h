//
// Created by kartykbayev on 5/23/24.
//
#pragma once

#include <thread>
#include "../SharedQueue.h"
#include "../ILogger.h"
#include "../IThreadLauncher.h"
#include "FrameData.h"
#include "FrameSnapshotReader.h"

class CameraClientLauncher: public IThreadLauncher, public ILogger{
public:
    CameraClientLauncher(const std::vector<std::string > &cameras,
                         std::vector<std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>>> &frameQueues,
                         const std::string &username, const std::string &password, int eventInterval,
                         const std::string &cameraSnapshotUrl, const std::string &cameraTimeUrl);

    void run() override;

    void shutdown() override;

protected:
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<FrameSnapshotReader>> cameraSnapshotReaders;

};


