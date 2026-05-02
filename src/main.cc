/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Copyright (c) 2024-2025, WuChao && MaChao D-Robotics. (Original Author)
 * Copyright (c) 2026, A7bert777. (Modifications & RDK X5 Adaptation)
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * -----------------------------------------------------------------------
 *
 * Modified by: A7bert777
 * Contact:
 * - Email: 2506245294@qq.com
 * - QQ: 2506245294
 *
 * Description:
 * Adapted and optimized for Horizon RDK X5 deployment.
 * Key modifications include:
 * 1. Added robust handling for BPU quantized tensors with/without
 * dequantization nodes.
 * 2. Implemented absolute memory offset calculation to correctly handle
 * hardware memory padding/alignment.
 * 3. Integrated `<dirent.h>` for efficient batch image processing.
 * 4. Added YOLOv8-Pose/YOLO11-Pose keypoint estimation support.
 * 5. Customized for knob pose detection (2 keypoints: head and tail).
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径 (硬编码为相对路径或绝对路径)
#define MODEL_PATH "../model/yolov8npose_knob.bin"

// 测试图片输入文件夹路径
#define INPUT_FOLDER_PATH "../inputimage"

// 结果保存文件夹路径
#define OUTPUT_FOLDER_PATH "../outputimage"

// ----------------------------------------------------------------------------
// 核心参数：选择 .bin 模型的类型
// 1 = 模型去除了反量化节点 (使用修改后的绝对偏移+动态类型转换逻辑)
// 0 = 模型未去除反量化节点 (使用原始相对偏移+Float逻辑)
// ----------------------------------------------------------------------------
#define REMOVE_DEQUANT_NODE 0

// 前处理方式: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型的类别数量 (knob: 1个类别)
#define CLASSES_NUM 1

// NMS的阈值
#define NMS_THRESHOLD 0.45

// 分数阈值
#define SCORE_THRESHOLD 0.25

// 关键点置信度阈值
#define KPT_SCORE_THRESHOLD 0.5

// 控制回归部分离散化程度的超参数, DFL
#define REG 16

// 关键点数量 (knob: head和tail共2个关键点)
#define KPT_NUM 2

// 关键点编码维度 (x, y) — 注意: kpt_shape: [2, 2] 每个关键点只有x,y两个值，没有独立的confidence
#define KPT_ENCODE 2

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

// ============================================================================
// Knob Keypoint Names and Skeleton
// ============================================================================

const std::vector<std::string> KEYPOINT_NAMES = {
    "head", "tail"
};

// Skeleton connections (pairs of keypoint indices)
// head(0) -> tail(1): 连接head和tail两个关键点
const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}   // head -> tail
};

const cv::Scalar KEYPOINT_COLOR = cv::Scalar(0, 0, 255);      // Red
const cv::Scalar SKELETON_COLOR = cv::Scalar(255, 0, 0);      // Blue
const cv::Scalar BBOX_COLOR = cv::Scalar(0, 255, 0);          // Green

// ============================================================================
// Pose Detection Result Structure
// ============================================================================

struct PoseDetection {
    cv::Rect2d bbox;                          // Bounding box
    float score;                              // Confidence score
    std::vector<cv::Point2f> keypoints;       // KPT_NUM keypoints (x, y)
    std::vector<float> keypoint_scores;       // KPT_NUM keypoint confidences
};

// ============================================================================
// Utility Functions
// ============================================================================

std::string extractFileNameWithoutExtension(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    pos = filename.find_last_of(".");
    if (pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }
    return filename;
}

/**
 * @brief Convert BGR image to NV12 format
 */
cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    int height = bgr_img.rows;
    int width = bgr_img.cols;
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();
    cv::Mat nv12_img(height * 3 / 2, width, CV_8UC1);
    uint8_t* nv12 = nv12_img.ptr<uint8_t>();
    int y_size = height * width;
    memcpy(nv12, yuv, y_size);
    int uv_height = height / 2;
    int uv_width = width / 2;
    uint8_t* nv12_uv = nv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12_uv++ = *u_data++;
        *nv12_uv++ = *v_data++;
    }
    return nv12_img;
}

/**
 * @brief Preprocess image with letterbox or resize
 */
cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift) {
    cv::Mat result;
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;
        if (x_scale <= 0 || y_scale <= 0) {
            throw std::runtime_error("Invalid scale factor");
        }
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        int x_other = input_w - new_w - x_shift;
        int y_other = input_h - new_h - y_shift;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, y_other, x_shift, x_other,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;
    }
    return result;
}

/**
 * @brief Sigmoid function
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Softmax function for DFL
 */
void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

/**
 * @brief Draw pose detection results (knob: head and tail)
 */
void draw_pose(cv::Mat& img, const PoseDetection& det, float kpt_threshold_raw) {
    int x1 = static_cast<int>(det.bbox.x);
    int y1 = static_cast<int>(det.bbox.y);
    int x2 = static_cast<int>(det.bbox.x + det.bbox.width);
    int y2 = static_cast<int>(det.bbox.y + det.bbox.height);
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), BBOX_COLOR, 2);

    std::string label = "knob: " + std::to_string(det.score).substr(0, 4);
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int label_y;
    if (y1 - label_size.height < 0) {
        label_y = y2 + label_size.height;
        cv::rectangle(img, cv::Point(x1, y2),
                     cv::Point(x1 + label_size.width, y2 + label_size.height + baseline),
                     BBOX_COLOR, cv::FILLED);
        cv::putText(img, label, cv::Point(x1, y2 + label_size.height),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    } else {
        label_y = y1;
        cv::rectangle(img, cv::Point(x1, label_y - label_size.height),
                     cv::Point(x1 + label_size.width, label_y + baseline),
                     BBOX_COLOR, cv::FILLED);
        cv::putText(img, label, cv::Point(x1, label_y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Draw skeleton connections (head -> tail)
    for (const auto& connection : SKELETON) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        if (det.keypoint_scores[idx1] >= kpt_threshold_raw &&
            det.keypoint_scores[idx2] >= kpt_threshold_raw) {
            cv::Point pt1(static_cast<int>(det.keypoints[idx1].x),
                         static_cast<int>(det.keypoints[idx1].y));
            cv::Point pt2(static_cast<int>(det.keypoints[idx2].x),
                         static_cast<int>(det.keypoints[idx2].y));
            cv::line(img, pt1, pt2, SKELETON_COLOR, 2);
        }
    }

    // Draw keypoints with labels
    for (int i = 0; i < KPT_NUM; i++) {
        if (det.keypoint_scores[i] >= kpt_threshold_raw) {
            int x = static_cast<int>(det.keypoints[i].x);
            int y = static_cast<int>(det.keypoints[i].y);
            cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            cv::circle(img, cv::Point(x, y), 5, color, -1);
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);
            cv::putText(img, KEYPOINT_NAMES[i], cv::Point(x + 5, y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 2);
            cv::putText(img, KEYPOINT_NAMES[i], cv::Point(x + 5, y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        }
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) 
{
    LOG_INFO("=== Ultralytics YOLO Knob Pose Batch Demo (RDK X5) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);

    std::string model_path = MODEL_PATH;
    std::string input_folder = INPUT_FOLDER_PATH;
    std::string output_folder = OUTPUT_FOLDER_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) input_folder = argv[2];
    if (argc >= 4) output_folder = argv[3];

    // ========================================================================
    // 1. Load BPU model
    // ========================================================================
    LOG_INFO("Loading model: " << model_path);
    hbPackedDNNHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "Failed to initialize model from file");

    const char** model_name_list;
    int model_count = 0;
    CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "Failed to get model name list");

    const char* model_name = model_name_list[0];
    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "Failed to get model handle");

    // ========================================================================
    // 2. Check model input
    // ========================================================================
    int32_t input_count = 0;
    CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "Failed to get input count");

    hbDNNTensorProperties input_properties;
    CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "Failed to get input tensor properties");

    int32_t input_h = input_properties.validShape.dimensionSize[2];
    int32_t input_w = input_properties.validShape.dimensionSize[3];
    LOG_INFO("Input shape: (1, 3, " << input_h << ", " << input_w << ")");

    // ========================================================================
    // 3. Check model outputs
    // ========================================================================
    int32_t output_count = 0;
    CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "Failed to get output count");
    LOG_INFO("Model has " << output_count << " outputs");

    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties output_properties;
        CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "Failed to get output tensor properties");
        std::cout << "output[" << i << "] shape: ("
                 << output_properties.validShape.dimensionSize[0] << ", "
                 << output_properties.validShape.dimensionSize[1] << ", "
                 << output_properties.validShape.dimensionSize[2] << ", "
                 << output_properties.validShape.dimensionSize[3] << "), ";
        if (output_properties.quantiType == SHIFT) std::cout << "SHIFT";
        else if (output_properties.quantiType == SCALE) std::cout << "SCALE";
        else if (output_properties.quantiType == NONE) std::cout << "NONE";
        std::cout << std::endl;
    }

    // ========================================================================
    // 4. Allocate System Memory
    // ========================================================================
    hbDNNTensor input;
    input.properties = input_properties;
    int input_memSize = input_h * input_w * 3 / 2;
    hbSysAllocCachedMem(&input.sysMem[0], input_memSize);

    hbDNNTensor* output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int out_size = output[i].properties.alignedByteSize;
        hbSysAllocCachedMem(&output[i].sysMem[0], out_size);
    }

    // ========================================================================
    // 5. Batch Process Images in Folder
    // ========================================================================
    DIR *dir = opendir(input_folder.c_str());
    if (dir == nullptr) {
        LOG_ERROR("Failed to open input directory: " << input_folder);
        return -1;
    }

    struct dirent *entry;
    int image_idx = 0;

    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName = entry->d_name;
        std::string fullPath = input_folder + "/" + fileName;

        if ((fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".jpg") == 0) ||
            (fileName.size() >= 5 && strcmp(fileName.c_str() + fileName.size() - 5, ".jpeg") == 0) ||
            (fileName.size() >= 4 && strcmp(fileName.c_str() + fileName.size() - 4, ".png") == 0)) {

            image_idx++;
            std::string outputFileName = output_folder + "/" + extractFileNameWithoutExtension(fullPath) + "_out.jpg";

            LOG_INFO("---------------------------------------------------------");
            LOG_INFO("[" << image_idx << "] Processing: " << fileName);

            cv::Mat img = cv::imread(fullPath);
            if (img.empty()) {
                LOG_ERROR("Failed to load image: " << fullPath);
                continue;
            }

            float x_scale, y_scale;
            int x_shift, y_shift;
            cv::Mat preprocessed = preprocess_image(img, input_h, input_w, x_scale, y_scale, x_shift, y_shift);
            cv::Mat nv12_img = bgr2nv12(preprocessed);

            memcpy(input.sysMem[0].virAddr, nv12_img.ptr<uint8_t>(), input_memSize);
            hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

            hbDNNTaskHandle_t task_handle = nullptr;
            hbDNNInferCtrlParam infer_ctrl_param;
            HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

            auto start_time = std::chrono::high_resolution_clock::now();
            hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
            hbDNNWaitTaskDone(task_handle, 0);

            auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            LOG_TIME("  Inference time", infer_duration);
            hbDNNReleaseTask(task_handle);

            // ========================================================================
            // 6. Post-process
            // ========================================================================
            float CONF_THRES_RAW = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);
            float KPT_THRES_RAW = -std::log(1.0f / KPT_SCORE_THRESHOLD - 1.0f);

            std::vector<PoseDetection> detections;

            const int strides[3] = {8, 16, 32};
            const int grid_sizes[3] = {input_h / 8, input_h / 16, input_h / 32};

            // 模型输出布局（9个输出头，每个尺度3个）：
            //   尺度1 (80x80):  output[0]=cls(1ch), output[1]=box(64ch), output[2]=kpt(4ch)
            //   尺度2 (40x40):  output[3]=cls(1ch), output[4]=box(64ch), output[5]=kpt(4ch)
            //   尺度3 (20x20):  output[6]=cls(1ch), output[7]=box(64ch), output[8]=kpt(4ch)
            //
            // 关键点输出通道数为4，对应2个关键点 (head, tail) 的 (x, y) 坐标
            // 布局: [head_x, head_y, tail_x, tail_y]
            // 关键点置信度从检测框的分类置信度继承

            for (int scale = 0; scale < 3; scale++) {
                int cls_idx = scale * 3 + 0;
                int box_idx = scale * 3 + 1;
                int kpt_idx = scale * 3 + 2;

                int grid_h = grid_sizes[scale];
                int grid_w = grid_sizes[scale];
                float stride = strides[scale];

                hbSysFlushMem(&output[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
                hbSysFlushMem(&output[box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
                hbSysFlushMem(&output[kpt_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

#if REMOVE_DEQUANT_NODE == 1
                float cls_scale = 1.0f;
                if (output[cls_idx].properties.quantiType == SCALE)
                    cls_scale = output[cls_idx].properties.scale.scaleData[0];
                float bbox_scale = 1.0f;
                if (output[box_idx].properties.quantiType == SCALE)
                    bbox_scale = output[box_idx].properties.scale.scaleData[0];
                float kpt_scale = 1.0f;
                if (output[kpt_idx].properties.quantiType == SCALE)
                    kpt_scale = output[kpt_idx].properties.scale.scaleData[0];

                int cls_aligned_w = output[cls_idx].properties.alignedShape.dimensionSize[2];
                int cls_aligned_c = output[cls_idx].properties.alignedShape.dimensionSize[3];
                int cls_type = output[cls_idx].properties.tensorType;
                void* cls_base = output[cls_idx].sysMem[0].virAddr;

                int bbox_aligned_w = output[box_idx].properties.alignedShape.dimensionSize[2];
                int bbox_aligned_c = output[box_idx].properties.alignedShape.dimensionSize[3];
                int bbox_type = output[box_idx].properties.tensorType;
                void* bbox_base = output[box_idx].sysMem[0].virAddr;

                int kpt_aligned_w = output[kpt_idx].properties.alignedShape.dimensionSize[2];
                int kpt_aligned_c = output[kpt_idx].properties.alignedShape.dimensionSize[3];
                int kpt_type = output[kpt_idx].properties.tensorType;
                void* kpt_base = output[kpt_idx].sysMem[0].virAddr;

                for (int h = 0; h < grid_h; h++) {
                    for (int w = 0; w < grid_w; w++) {
                        int cls_offset = h * cls_aligned_w * cls_aligned_c + w * cls_aligned_c;
                        int bbox_offset = h * bbox_aligned_w * bbox_aligned_c + w * bbox_aligned_c;
                        int kpt_offset = h * kpt_aligned_w * kpt_aligned_c + w * kpt_aligned_c;

                        float cls_val = 0.0f;
                        if (cls_type == HB_DNN_TENSOR_TYPE_S32)
                            cls_val = ((int32_t*)cls_base)[cls_offset] * cls_scale;
                        else if (cls_type == HB_DNN_TENSOR_TYPE_S8)
                            cls_val = ((int8_t*)cls_base)[cls_offset] * cls_scale;
                        else
                            cls_val = ((float*)cls_base)[cls_offset];

                        if (cls_val < CONF_THRES_RAW) continue;

                        float score = 1.0f / (1.0f + std::exp(-cls_val));

                        float ltrb[4] = {0.0f};
                        for (int i = 0; i < 4; i++) {
                            float dfl_values[REG], dfl_softmax[REG];
                            for (int j = 0; j < REG; j++) {
                                int idx = REG * i + j;
                                float bbox_val = 0.0f;
                                if (bbox_type == HB_DNN_TENSOR_TYPE_S32)
                                    bbox_val = ((int32_t*)bbox_base)[bbox_offset + idx] * bbox_scale;
                                else if (bbox_type == HB_DNN_TENSOR_TYPE_S8)
                                    bbox_val = ((int8_t*)bbox_base)[bbox_offset + idx] * bbox_scale;
                                else
                                    bbox_val = ((float*)bbox_base)[bbox_offset + idx];
                                dfl_values[j] = bbox_val;
                            }
                            softmax(dfl_values, dfl_softmax, REG);
                            for (int j = 0; j < REG; j++)
                                ltrb[i] += dfl_softmax[j] * j;
                        }

                        float cx = (w + 0.5f) * stride;
                        float cy = (h + 0.5f) * stride;
                        float x1 = cx - ltrb[0] * stride;
                        float y1 = cy - ltrb[1] * stride;
                        float x2 = cx + ltrb[2] * stride;
                        float y2 = cy + ltrb[3] * stride;

                        if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
                            x2 <= input_w && y2 <= input_h) {
                            PoseDetection det;
                            det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
                            det.score = score;
                            det.keypoints.resize(KPT_NUM);
                            det.keypoint_scores.resize(KPT_NUM);

                            // 关键点输出: 4通道 [head_x, head_y, tail_x, tail_y]
                            // YOLOv8-Pose 关键点是直接回归的，不需要 sigmoid！
                            // 解码公式: (raw_value * 2.0 + grid_cell_idx) * stride
                            float kpt_raw[4] = {0.0f};
                            if (kpt_type == HB_DNN_TENSOR_TYPE_S32) {
                                for (int k = 0; k < 4; k++)
                                    kpt_raw[k] = ((int32_t*)kpt_base)[kpt_offset + k] * kpt_scale;
                            } else if (kpt_type == HB_DNN_TENSOR_TYPE_S8) {
                                for (int k = 0; k < 4; k++)
                                    kpt_raw[k] = ((int8_t*)kpt_base)[kpt_offset + k] * kpt_scale;
                            } else {
                                for (int k = 0; k < 4; k++)
                                    kpt_raw[k] = ((float*)kpt_base)[kpt_offset + k];
                            }

                            for (int k = 0; k < KPT_NUM; k++) {
                                float decoded_x = (kpt_raw[k * 2 + 0] * 2.0f + w) * stride;
                                float decoded_y = (kpt_raw[k * 2 + 1] * 2.0f + h) * stride;
                                det.keypoints[k] = cv::Point2f(decoded_x, decoded_y);
                                // 关键点置信度继承检测框的分类置信度（原始 logit 值）
                                det.keypoint_scores[k] = cls_val;
                            }
                            detections.push_back(det);
                        }
                    }
                }
#else
                // REMOVE_DEQUANT_NODE == 0: 模型未去除反量化节点，输出为 float 类型
                float* cls_raw = reinterpret_cast<float*>(output[cls_idx].sysMem[0].virAddr);
                float* box_raw = reinterpret_cast<float*>(output[box_idx].sysMem[0].virAddr);
                float* kpt_raw = reinterpret_cast<float*>(output[kpt_idx].sysMem[0].virAddr);

                // 获取输出通道数
                int cls_channels = output[cls_idx].properties.validShape.dimensionSize[3];
                int box_channels = output[box_idx].properties.validShape.dimensionSize[3];
                int kpt_channels = output[kpt_idx].properties.validShape.dimensionSize[3];

                for (int h = 0; h < grid_h; h++) {
                    for (int w = 0; w < grid_w; w++) {
                        int offset = h * grid_w + w;

                        // 分类输出: 每个 grid cell 有 cls_channels 个值
                        float* cur_cls = cls_raw + offset * cls_channels;
                        // 边界框输出: 每个 grid cell 有 box_channels 个值 (4*REG=64)
                        float* cur_box = box_raw + offset * box_channels;
                        // 关键点输出: 每个 grid cell 有 kpt_channels 个值 (4)
                        float* cur_kpt = kpt_raw + offset * kpt_channels;

                        if (cur_cls[0] < CONF_THRES_RAW) continue;

                        float score = 1.0f / (1.0f + std::exp(-cur_cls[0]));

                        // DFL 解码边界框
                        float ltrb[4] = {0.0f};
                        for (int i = 0; i < 4; i++) {
                            float dfl_values[REG], dfl_softmax[REG];
                            for (int j = 0; j < REG; j++)
                                dfl_values[j] = cur_box[i * REG + j];
                            softmax(dfl_values, dfl_softmax, REG);
                            for (int j = 0; j < REG; j++)
                                ltrb[i] += dfl_softmax[j] * j;
                        }

                        float cx = (w + 0.5f) * stride;
                        float cy = (h + 0.5f) * stride;
                        float x1 = cx - ltrb[0] * stride;
                        float y1 = cy - ltrb[1] * stride;
                        float x2 = cx + ltrb[2] * stride;
                        float y2 = cy + ltrb[3] * stride;

                        if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
                            x2 <= input_w && y2 <= input_h) {
                            PoseDetection det;
                            det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
                            det.score = score;
                            det.keypoints.resize(KPT_NUM);
                            det.keypoint_scores.resize(KPT_NUM);

                            // 关键点输出: 4通道 [head_x, head_y, tail_x, tail_y]
                            // YOLOv8-Pose 关键点是直接回归的，不需要 sigmoid！
                            // 解码公式: (raw_value * 2.0 + grid_cell_idx) * stride
                            for (int k = 0; k < KPT_NUM; k++) {
                                float decoded_x = (cur_kpt[k * 2 + 0] * 2.0f + w) * stride;
                                float decoded_y = (cur_kpt[k * 2 + 1] * 2.0f + h) * stride;
                                det.keypoints[k] = cv::Point2f(decoded_x, decoded_y);
                                // 关键点置信度继承检测框的分类置信度（原始 logit 值）
                                det.keypoint_scores[k] = cur_cls[0];
                            }
                            detections.push_back(det);
                        }
                    }
                }
#endif
            }

            LOG_INFO("  Detections before NMS: " << detections.size());

            // ========================================================================
            // 6.5 坐标反映射：将检测结果从模型输入空间映射回原始图像空间
            // ========================================================================
            // 由于使用 letterbox 前处理，模型输入空间中的坐标需要逆变换回原图坐标
            // 原图坐标 = (模型坐标 - 偏移量) / 缩放比例
            for (auto& det : detections) {
                // 反映射 bbox
                float orig_x1 = (det.bbox.x - x_shift) / x_scale;
                float orig_y1 = (det.bbox.y - y_shift) / y_scale;
                float orig_x2 = (det.bbox.x + det.bbox.width - x_shift) / x_scale;
                float orig_y2 = (det.bbox.y + det.bbox.height - y_shift) / y_scale;
                det.bbox.x = orig_x1;
                det.bbox.y = orig_y1;
                det.bbox.width = orig_x2 - orig_x1;
                det.bbox.height = orig_y2 - orig_y1;

                // 反映射关键点坐标
                for (int k = 0; k < KPT_NUM; k++) {
                    det.keypoints[k].x = (det.keypoints[k].x - x_shift) / x_scale;
                    det.keypoints[k].y = (det.keypoints[k].y - y_shift) / y_scale;
                }
            }

            // ========================================================================
            // 7. NMS
            // ========================================================================
            std::vector<cv::Rect2d> nms_boxes;
            std::vector<float> nms_scores;
            std::vector<int> nms_indices;
            for (const auto& det : detections) {
                nms_boxes.push_back(det.bbox);
                nms_scores.push_back(det.score);
            }
            cv::dnn::NMSBoxes(nms_boxes, nms_scores, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);

            LOG_INFO("  Detections after NMS: " << nms_indices.size());

            // ========================================================================
            // 8. Draw results and save
            // ========================================================================
            cv::Mat result_img = img.clone();
            for (int idx : nms_indices) {
                draw_pose(result_img, detections[idx], KPT_THRES_RAW);
            }

            cv::imwrite(outputFileName, result_img);
            LOG_INFO("  Saved result to: " << outputFileName);
        }
    }

    closedir(dir);

    // ========================================================================
    // 9. Cleanup
    // ========================================================================
    for (int i = 0; i < output_count; i++) {
        hbSysFreeMem(&output[i].sysMem[0]);
    }
    delete[] output;
    hbSysFreeMem(&input.sysMem[0]);
    hbDNNRelease(packed_dnn_handle);

    LOG_INFO("=========================================================");
    LOG_INFO("Batch processing complete. Processed " << image_idx << " images.");
    LOG_INFO("=========================================================");

    return 0;
}
