//
// Created by kartykbayev on 5/23/24.
//

#include "CameraClientLauncher.h"

using namespace std;

CameraClientLauncher::CameraClientLauncher(const std::vector<std::string> &cameras,
                                           std::vector<std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>>> &frameQueues,
                                           const std::string &username, const std::string &password, int eventInterval,
                                           const std::string &cameraSnapshotUrl, const std::string &cameraTimeUrl)
        : ILogger("Camera Client Launcher ") {
    int index = 0;
    for (
        const auto &camera: cameras) {
        auto cameraFrameReader = make_shared<FrameSnapshotReader>(camera, username, password,frameQueues[index], eventInterval, cameraSnapshotUrl, cameraTimeUrl);
        cameraSnapshotReaders.push_back(std::move(cameraFrameReader));
        index++;
    }
}


void CameraClientLauncher::run() {
    for (const auto &frameStreamer: cameraSnapshotReaders)
        threads.emplace_back(&FrameSnapshotReader::launchStream, frameStreamer);
}

void CameraClientLauncher::shutdown() {
    for (int i = 0; i < cameraSnapshotReaders.size(); i++) {
        cameraSnapshotReaders[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
}
