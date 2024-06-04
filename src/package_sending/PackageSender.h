//
// Created by kartykbayev on 5/31/24.
//
#pragma once

#include <ctime>
#include <future>
#include <unordered_map>
#include <utility>

#include <cpr/cpr.h>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "../app/LicensePlate.h"
#include "Package.h"
#include "../app/Utils.h"

class PackageSender: public IThreadLauncher,public ILogger{
public:
    PackageSender(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                  std::string serverUrl);

    void run() override;

    void shutdown() override;
private:

    const int SEND_REQUEST_TIMEOUT = 10000;
    const int MAX_FUTURE_RESPONSES = 30;
    std::string serverEndpoint;

    std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue;
    cpr::AsyncResponse sendRequests(const std::string &jsonString);

};


