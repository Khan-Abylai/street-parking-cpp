cmake_minimum_required(VERSION 3.25)
project(streetParking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(ncnn REQUIRED)

pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GST_RTP REQUIRED gstreamer-rtp-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(LIBONNXRUNTIME REQUIRED IMPORTED_TARGET libonnxruntime)

include_directories(${CURL_INCLUDE_DIR})
include_directories(${GST_INCLUDE_DIRS} ${GST_APP_INCLUDE_DIRS} ${GST_RTP_INCLUDE_DIRS})
include_directories(${LIBONNXRUNTIME_INCLUDE_DIRS})

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /tmp/onnxruntime/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /tmp/onnxruntime/build/Linux/MinSizeRel)

if (NOT ONNX_RUNTIME_SESSION_INCLUDE_DIRS)
    set(ONNX_RUNTIME_SESSION_INCLUDE_DIRS /home/orangepi/onnxruntime/include/onnxruntime/core/session/)
endif ()

if (NOT ONNX_RUNTIME_LIB)
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /home/orangepi/onnxruntime/build/Linux/MinSizeRel)
endif ()

option(DEBUG_LOG "Debug Mode" ON)
if (DEBUG_LOG)
    add_definitions(-DDEBUG=1)
else ()
    add_definitions(-DDEBUG=0)
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
        src/app/DetectorYoloV5NCNN.cpp
        src/app/DetectorYoloV5NCNN.h
        src/app/LicensePlate.cpp
        src/app/LicensePlate.h
        src/app/LPRecognizerNCNN.cpp
        src/app/LPRecognizerNCNN.h
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
)

target_include_directories(streetParking PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ncnn_INCLUDE_DIRS})
target_link_libraries(streetParking PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBS} ${GST_LIBRARIES} ${GST_APP_LIBRARIES} ${GST_RTP_LIBRARIES} nlohmann_json::nlohmann_json ${CURL_LIBRARIES} /usr/local/lib/libcpr.so ${ALL_INTERFACE_TARGET} ncnn)
target_compile_definitions(streetParking PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING>)