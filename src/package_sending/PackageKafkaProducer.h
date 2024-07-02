//
// Created by kartykbayev on 7/2/24.
//

#pragma once

#include <ctime>
#include <future>
#include <unordered_map>
#include <utility>

#include <kafka/KafkaProducer.h>

#include "../IThreadLauncher.h"
#include "../ILogger.h"
#include "../SharedQueue.h"
#include "Package.h"

class PackageKafkaProducer : public IThreadLauncher, public ILogger {
public:
    PackageKafkaProducer(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                         const std::string &kafkaBrokers,
                         std::string topicName);

    void run() override;

    void shutdown() override;

private:
    std::string topicName;
    std::string serverEndpoint;

    std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue;
    std::unique_ptr<kafka::clients::producer::KafkaProducer> kafkaProducer;

};


