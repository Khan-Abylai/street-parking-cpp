//
// Created by kartykbayev on 7/2/24.
//

#include "PackageKafkaProducer.h"

#include <utility>


using namespace std;
using namespace kafka;
using namespace kafka::clients::producer;

PackageKafkaProducer::PackageKafkaProducer(std::shared_ptr<SharedQueue<std::shared_ptr<Package>>> packageQueue,
                                           const std::string &kafkaBrokers, std::string topicName) : ILogger(
        "Kafka Producer"), topicName{std::move(topicName)}, packageQueue{std::move(packageQueue)} {
    kafka::Properties props;
    props.put("bootstrap.servers", kafkaBrokers);
    props.put("log_level", "0");

    kafkaProducer = make_unique<KafkaProducer>(props);

}

void PackageKafkaProducer::run() {
    while (!shutdownFlag) {
        auto package = packageQueue->wait_and_pop();
        if (package == nullptr) continue;

        auto deliveryCb = [this](const RecordMetadata &metadata, const Error &error) {
            if (!error) {
                LOG_DEBUG("%s", metadata.toString().data());
            } else {
                LOG_ERROR("%s", error.message().data());
            }
        };

        LOG_INFO("Camera %s, preset %s, plate numbers: %s, event time: %s", package->getCameraIp().data(),
                 package->getPresetId().data(),
                 package->getPlateLabelsRaw().data(), package->getEventTime().data());
        std::string message = package->getPresetId();
        ProducerRecord record(topicName, NullKey, Value(message.c_str(), message.size()));

        kafkaProducer->send(record, deliveryCb);
    }
}

void PackageKafkaProducer::shutdown() {
    LOG_INFO("service is shutting down");
    shutdownFlag = true;
    packageQueue->push(nullptr);
    kafkaProducer->close();
}

