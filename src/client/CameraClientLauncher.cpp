//
// Created by kartykbayev on 5/23/24.
//

#include "CameraClientLauncher.h"

using namespace std;

CameraClientLauncher::CameraClientLauncher(const std::vector<std::string> &cameras,
                                           const std::vector<std::shared_ptr<SharedQueue<std::unique_ptr<FrameData>>>> &frameQueues,
                                           const std::string &username, const std::string &password)
        : ILogger("Camera Client Launcher ") {
    int index = 0;
    for (
        const auto &camera: cameras) {
        auto cameraFrameReader = make_shared<GstreamerReader>(camera, true,frameQueues[index]);
        cameraFrameReaders.push_back(std::move(cameraFrameReader));
    }
}


void CameraClientLauncher::run() {
    for (const auto &gstreamer: cameraFrameReaders)
        threads.emplace_back(&GstreamerReader::launchStream, gstreamer);
}

void CameraClientLauncher::shutdown() {
    for (int i = 0; i < cameraFrameReaders.size(); i++) {
        cameraFrameReaders[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
}
