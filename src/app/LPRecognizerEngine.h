//
// Created by kartykbayev on 9/5/24.
//
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

#include "Constants.h"
#include "TensorRTDeleter.h"
#include "TrtLogger.h"
#include "TensorRTEngine.h"
#include "Utils.h"

class LPRecognizerEngine {

public:
    explicit LPRecognizerEngine();

};