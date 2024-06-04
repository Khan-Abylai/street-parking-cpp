//
// Created by kartykbayev on 5/30/24.
//

#include "DetectorYoloV5NCNN.h"

using namespace std;
using namespace cv;
using namespace dnn;

DetectorYoloV5NCNN::DetectorYoloV5NCNN() {
    yolo.opt.use_vulkan_compute = true;

    if (yolo.load_param(Constants::detParam.data()))
        exit(-1);
    if (yolo.load_model(Constants::detBin.data()))
        exit(-1);
}

DetectorYoloV5NCNN::~DetectorYoloV5NCNN() {
    yolo.clear();
}

std::vector<std::shared_ptr<LicensePlate>> DetectorYoloV5NCNN::detect(const Mat &rgb) {
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, INPUT_WIDTH,
                                                 INPUT_HEIGHT);

    in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("input", in);

    ncnn::Mat outs;
    ex.extract("output", outs);

    auto a = cv::Mat(outs.h, outs.w, CV_32FC1);
    std::memcpy((uchar *) a.data, outs.data, outs.w * outs.h * sizeof(float));

    vector<float> confidences;
    vector<cv::Rect> boxes;
    vector<vector<int>> landmarks;
    vector<int> class_indexes;

    float ratioh = (float) rgb.rows / INPUT_WIDTH, ratiow = (float) rgb.cols / INPUT_HEIGHT;
    int n = 0, q = 0, i = 0, j = 0, nout = 17, row_ind = 0, k = 0;
    for (n = 0; n < 3; n++) {
        int num_grid_x = (int) (INPUT_WIDTH / stride[n]);
        int num_grid_y = (int) (INPUT_HEIGHT / stride[n]);
        for (q = 0; q < 3; q++) {
            const float anchor_w = anchors[n][q * 2];
            const float anchor_h = anchors[n][q * 2 + 1];
            for (i = 0; i < num_grid_y; i++) {
                for (j = 0; j < num_grid_x; j++) {
                    float *pdata = (float *) a.data + row_ind * nout;

                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < 2; k++) {
                        float score = pdata[15 + k];
                        if (score > class_score) {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    float box_score = sigmoid(pdata[4]);

                    if (box_score >= PROB_THRESHOLD) {
                        float cx = (sigmoid(pdata[0]) * 2.f - 0.5f + j) * stride[n];
                        float cy = (sigmoid(pdata[1]) * 2.f - 0.5f + i) * stride[n];
                        float w = powf(sigmoid(pdata[2]) * 2.f, 2.f) * anchor_w;
                        float h = powf(sigmoid(pdata[3]) * 2.f, 2.f) * anchor_h;

                        int left = (cx - 0.5 * w) * ratiow;
                        int top = (cy - 0.5 * h) * ratioh;

                        confidences.push_back(box_score);
                        class_indexes.push_back(class_index);
                        boxes.push_back(cv::Rect(left, top, (int) (w * ratiow), (int) (h * ratioh)));
                        vector<int> landmark(10);
                        for (k = 5; k < 15; k += 2) {
                            const int ind = k - 5;
                            landmark[ind] = (int) (pdata[k] * anchor_w + j * stride[n]) * ratiow;
                            landmark[ind + 1] = (int) (pdata[k + 1] * anchor_h + i * stride[n]) * ratioh;
                        }
                        landmarks.push_back(landmark);
                    }
                    row_ind++;
                }
            }
        }
    }


    if (boxes.empty()) return {};

    vector<int> indices;
    NMSBoxes(boxes, confidences, PROB_THRESHOLD, NMS_THRESHOLD, indices);

    int count = indices.size();
    std::vector<Objects> objects;
    objects.resize(count);

    for (size_t i = 0; i < count; i++) {
        int idx = indices[i];
        Rect box = boxes[idx];
        objects[i].rect.x = box.x;
        objects[i].rect.y = box.y;
        objects[i].rect.width = box.width;
        objects[i].rect.height = box.height;
        objects[i].prob = confidences[idx];
        objects[i].class_id = class_indexes[idx];
        for (int j = 0; j < 5; j++) {
            float x = landmarks[idx][j * 2];
            float y = landmarks[idx][j * 2 + 1];
            objects[i].pts.push_back(cv::Point2f(x, y));
        }
    }

    std::vector<std::shared_ptr<LicensePlate>> lp_vector;

    for (auto &obj: objects) {
        float LT_x = obj.pts[0].x;
        float LT_y = obj.pts[0].y;
        float RT_x = obj.pts[1].x;
        float RT_y = obj.pts[1].y;
        float CP_x = obj.pts[2].x;
        float CP_y = obj.pts[2].y;
        float LB_x = obj.pts[3].x;
        float LB_y = obj.pts[3].y;
        float RB_x = obj.pts[4].x;
        float RB_y = obj.pts[4].y;
        float plate_width = ((RT_x - LT_x) + (RB_x - LB_x)) / 2;
        float plate_height = ((LB_y - LT_y) + (RB_y - RT_y)) / 2;
        float prob = obj.prob;

        if(obj.class_id ==0){
            auto lp = make_shared<LicensePlate>(static_cast<int>(CP_x), static_cast<int>(CP_y),
                                                plate_width, plate_height,
                                                static_cast<int>(floor(LT_x)),
                                                static_cast<int>(floor(LT_y)),
                                                static_cast<int>(floor(LB_x)),
                                                static_cast<int>(ceil(LB_y)),
                                                static_cast<int>(ceil(RT_x)),
                                                static_cast<int>(floor(RT_y)),
                                                static_cast<int>(ceil(RB_x)),
                                                static_cast<int>(ceil(RB_y)),
                                                static_cast<float>(prob));
            lp_vector.push_back(lp);
        }
    }

    ex.clear();

    return lp_vector;
}
