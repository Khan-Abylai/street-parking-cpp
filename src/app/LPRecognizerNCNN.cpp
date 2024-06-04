//
// Created by artyk on 9/20/2023.
//

#include "LPRecognizerNCNN.h"
using namespace std;
using namespace cv;
LPRecognizerNCNN::LPRecognizerNCNN() {
    recognizer.opt.use_vulkan_compute = true;
    if (recognizer.load_param(Constants::recognizerParam.data()))
        exit(-1);
    if (recognizer.load_model(Constants::recognizerBin.data()))
        exit(-1);
}

LPRecognizerNCNN::~LPRecognizerNCNN() {
    recognizer.clear();

}


std::vector<std::pair<std::string, float>> LPRecognizerNCNN::predict(const std::vector<cv::Mat> &frames) {
    vector <pair<string, float>> labels;


    for (const Mat &frame: frames) {
        ncnn::Extractor ex = recognizer.create_extractor();

        cv::Mat copyImage;
        frame.copyTo(copyImage);
        cv::Mat resized;
        resize(frame, resized, cv::Size(IMG_WIDTH, IMG_HEIGHT));

        ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, IMG_WIDTH, IMG_HEIGHT);

        const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
        const float norm_vals[3] = {1 / 255.0f, 1 / 255.0f, 1 / 255.0f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto flattenedInput = makeFlattened(in);
        ex.input("actual_input", in);

        ncnn::Mat output1Raw;
        ex.extract("output", output1Raw);

        auto output = makeFlattened(output1Raw);

        vector<float> predictions;
        predictions.resize(OUTPUT_SIZE);
        int batchSize = 1;

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {

            vector<float> preds(OUTPUT_SIZE);

            for (int i = 0; i < SEQUENCE_SIZE; i++) {
                vector<float> temp_out(ALPHABET_SIZE);
                for (int j = 0; j < ALPHABET_SIZE; j++) {
                    temp_out[j] = output[ALPHABET_SIZE * i + j];
                }

                vector<float> result = softmax(temp_out);
                temp_out.clear();
                int count = 0;
                for (int j = i * ALPHABET_SIZE; j < ALPHABET_SIZE * (i + 1); j++) {
                    preds[j] = result[count];
                    count++;
                }
                result.clear();
            }
            output.clear();
            int c = 0;
            for (int i = batchIndex * SEQUENCE_SIZE * ALPHABET_SIZE;
                 i < (batchIndex + 1) * ALPHABET_SIZE * SEQUENCE_SIZE; i++) {
                predictions[i] = preds[c];
                c++;
            }
            preds.clear();
        }

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            float prob = 1.0;
            string currentLabel;
            currentLabel.reserve(MAX_PLATE_SIZE);

            int currentChar = BLANK_INDEX;
            float currentProb = 1.0;

            for (int i = 0; i < SEQUENCE_SIZE; i++) {
                float maxProb = 0.0;
                int maxIndex = 0;

                for (int j = 0; j < ALPHABET_SIZE; j++) {
                    if (maxProb < predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j]) {
                        maxIndex = j;
                        maxProb = predictions[batchIndex * ALPHABET_SIZE * SEQUENCE_SIZE + i * ALPHABET_SIZE + j];
                    }
                }

                if (maxIndex == currentChar) {
                    currentProb = max(maxProb, currentProb);
                } else {
                    if (currentChar != BLANK_INDEX) {
                        currentLabel += ALPHABET[currentChar];
                        prob *= currentProb;
                    }
                    currentProb = maxProb;
                    currentChar = maxIndex;
                }
            }

            if (currentChar != BLANK_INDEX) {
                currentLabel += ALPHABET[currentChar];
                prob *= currentProb;
            }

            if (currentLabel.empty()) {
                currentLabel += ALPHABET[BLANK_INDEX];
                prob = 0.0;
            }

            labels.emplace_back(make_pair(currentLabel, prob));
        }
    }
    return labels;
}

std::vector<float> LPRecognizerNCNN::makeFlattened(ncnn::Mat &val) {
    ncnn::Mat outFlattened = val.reshape(val.w * val.h * val.c);
    std::vector<float> output;
    output.resize(outFlattened.w);
    for (int j = 0; j < outFlattened.w; j++) {
        output[j] = outFlattened[j];
    }

    return output;
}

std::vector<float> LPRecognizerNCNN::softmax(std::vector<float> &score_vec) {
    vector<float> softmax_vec(ALPHABET_SIZE);
    double score_max = *(max_element(score_vec.begin(), score_vec.end()));
    double e_sum = 0;
    for (int j = 0; j < ALPHABET_SIZE; j++) {
        softmax_vec[j] = exp((double) score_vec[j] - score_max);
        e_sum += softmax_vec[j];
    }
    for (int k = 0; k < ALPHABET_SIZE; k++)
        softmax_vec[k] /= e_sum;
    return softmax_vec;
}

