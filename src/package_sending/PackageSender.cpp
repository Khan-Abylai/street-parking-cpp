//
// Created by kartykbayev on 5/31/24.
//

#include "PackageSender.h"

using namespace std;

PackageSender::PackageSender(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                             std::string serverUrl): ILogger("Package Sender --------------------"),
                                                            packageQueue{std::move(packageQueue)}, serverEndpoint{std::move(serverUrl)}
{
    LOG_INFO("Server URL configured as %s", serverEndpoint.data());
}

void PackageSender::run() {
    std::queue<cpr::AsyncResponse> responses;
    while (!shutdownFlag){
        auto package = packageQueue->wait_and_pop();
        if (package == nullptr) continue;

        LOG_INFO("Camera %s, preset %s, plate numbers: %s, event time: %s", package->getCameraIp().data(), package->getPresetId().data(),
                 package->getPlateLabelsRaw().data(), package->getEventTime().data());
        responses.push(sendRequests(package->getPackageJson()));
    }
}

void PackageSender::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    packageQueue->push(nullptr);
}

cpr::AsyncResponse PackageSender::sendRequests(const std::string &jsonString) {
    return cpr::PostAsync(cpr::Url{serverEndpoint}, cpr::VerifySsl(false), cpr::Body{jsonString},
                          cpr::Timeout{SEND_REQUEST_TIMEOUT}, cpr::Header{{"Content-Type", "application/json"}});}
