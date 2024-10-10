cmake_minimum_required(VERSION 3.15)
set(CMAKE_C_COMPILER /usr/bin/gcc-9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)

project(streetParking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})

option(DEBUG_LOG "Debug Mode" OFF)
option(IMSHOW "Show Images" OFF)
option(IMSAVE "Save Images" OFF)

if (DEBUG_LOG)
    add_definitions(-DDEBUG=1)
else ()
    add_definitions(-DDEBUG=0)
endif ()

if (IMSHOW)
    add_definitions(-DIMSHOW=1)
else ()
    add_definitions(-DIMSHOW=0)
endif ()

if (IMSAVE)
    add_definitions(-DIMSAVE=1)
else ()
    add_definitions(-DIMSAVE=0)
endif ()

add_executable(streetParking
        src/StreetParking.cpp
        src/Config.cpp
        src/Config.h
        src/app/Utils.cpp
        src/app/Utils.h
        src/app/Constants.h
        src/ILogger.cpp
        src/ILogger.h
        src/ITimer.h
        src/IThreadLauncher.h
        src/RandomStringGenerator.cpp
        src/RandomStringGenerator.h
        src/SharedQueue.h
        src/client/CameraClientLauncher.cpp
        src/client/CameraClientLauncher.h
        src/client/FrameData.cpp
        src/client/FrameData.h
        src/client/FrameSnapshotReader.cpp
        src/client/FrameSnapshotReader.h
        src/app/ANPRService.cpp
        src/app/ANPRService.h
        src/client/GstreamerReader.cpp
        src/client/GstreamerReader.h
        src/app/LicensePlate.cpp
        src/app/LicensePlate.h
        src/app/TemplateMatching.cpp
        src/app/TemplateMatching.h
        src/app/CalibrationParams.cpp
        src/app/CalibrationParams.h
        src/app/CalibrationParamsUpdater.cpp
        src/app/CalibrationParamsUpdater.h
        src/package_sending/PackageSender.cpp
        src/package_sending/PackageSender.h
        src/package_sending/Package.cpp
        src/package_sending/Package.h
        src/app/Detection.cpp
        src/app/DetectionEngine.cpp
        src/app/DetectionEngine.h
        src/app/LPRecognizer.cpp
        src/app/LPRecognizer.h
        src/app/LPRecognizerEngine.cpp
        src/app/LPRecognizerEngine.h
        src/app/LPRecognizerExtended.cpp
        src/app/LPRecognizerExtended.h
        src/app/TensorRTEngine.cpp
)

set(MODELS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
set(MODELS_DESTINATION_DIR "${CMAKE_BINARY_DIR}/models")

# Debug output
message(STATUS "Models source dir: ${MODELS_SOURCE_DIR}")
message(STATUS "Models destination dir: ${MODELS_DESTINATION_DIR}")

# Using add_custom_command instead of add_custom_target
add_custom_command(
        TARGET streetParking POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${MODELS_SOURCE_DIR}
        ${MODELS_DESTINATION_DIR}
        COMMENT "Copying models directory to build directory")

target_link_libraries(streetParking nvinfer ${OpenCV_LIBS}  ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES}
        nlohmann_json::nlohmann_json /usr/local/cuda/lib64/libcudart.so
        ${CURL_LIBRARIES} /usr/local/lib/libcpr.so)