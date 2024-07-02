#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <mutex>
#include <atomic>

#include "Config.h"
#include "IThreadLauncher.h"
#include "client/FrameData.h"
#include "SharedQueue.h"
#include "client/CameraClientLauncher.h"
#include "app/ANPRService.h"
#include "package_sending/Package.h"
#include "package_sending/PackageSender.h"
#include "package_sending/PackageKafkaProducer.h"

using namespace std;


atomic<bool> shutdownFlag = false;
condition_variable shutdownEvent;
mutex shutdownMutex;

void signalHandler(int signum) {
    cout << "signal is to shutdown" << endl;
    shutdownFlag = true;
    shutdownEvent.notify_all();
}

int main(int argc, char *argv[]) {

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGHUP, signalHandler);
    signal(SIGABRT, signalHandler);

    string configFileName;

    if (argc <= 1)
        configFileName = "config.json";
    else
        configFileName = argv[1];

    if (!Config::parseJson(configFileName))
        return -1;

    vector<shared_ptr<IThreadLauncher>> services;
    auto packageQueue = make_shared<SharedQueue<shared_ptr<Package>>>();
    vector<shared_ptr<SharedQueue<unique_ptr<FrameData>>>> frameQueues;

    auto cameras = Config::getCameras();

    for (const auto &camera: cameras) {
        auto frameQueue = make_shared<SharedQueue<unique_ptr<FrameData>>>();
        auto anprService = make_shared<ANPRService>(frameQueue, packageQueue, camera, Config::getCalibrationEndPoint(),
                                                    Config::getCalibrationWidth(), Config::getCalibrationHeight());
        frameQueues.push_back(std::move(frameQueue));
        services.emplace_back(anprService);
    }

    shared_ptr<IThreadLauncher> clientStarter;
    clientStarter = make_shared<CameraClientLauncher>(cameras, frameQueues,
                                                      Config::getUsername(), Config::getPassword());
    services.emplace_back(clientStarter);

//    auto packageSender = make_shared<PackageSender>(packageQueue, Config::getEventEndpoint());
//    services.emplace_back(packageSender);

    auto kafkaPackageProducer = make_shared<PackageKafkaProducer>(packageQueue, Config::getKafkaBrokers(),
                                                                  Config::getKafkaTopicName());
    services.emplace_back(kafkaPackageProducer);


    vector<thread> threads;
    for (const auto &service: services) {
        threads.emplace_back(&IThreadLauncher::run, service);
    }

    unique_lock<mutex> shutdownLock(shutdownMutex);
    while (!shutdownFlag) {
        auto timeout = chrono::hours(24);
        if (shutdownEvent.wait_for(shutdownLock, timeout, [] { return shutdownFlag.load(); })) {
            cout << "shutting all services" << endl;
        }
    }

    for (int i = 0; i < services.size(); i++) {
        services[i]->shutdown();
        if (threads[i].joinable())
            threads[i].join();
    }
    return 0;

}