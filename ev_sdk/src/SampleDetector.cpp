//
// Created by hrh on 2019-09-02.
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <cJSON.h>
#include <sys/stat.h>
#include "SampleDetector.hpp"

#include <string>
#include <memory>
#include <vector>
#include <map>

using namespace InferenceEngine;


/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

static bool verifyXMLPath(const char* modelXMLPath){
    // 判断模型文件是否存在，并加载模型
    if (modelXMLPath == nullptr) {
        LOG(ERROR) << "Invalid init args!";
        return false;
    }
    struct stat st;
    if (stat(modelXMLPath, &st) != 0) {
        LOG(ERROR) << modelXMLPath << " not found!";
        return false;
    }
    std::string binFileName = fileNameNoExt(modelXMLPath) + ".bin";
    if (stat(binFileName.c_str(), &st) != 0) {
        LOG(ERROR) << binFileName << " not found!";
        return false;
    }
    return true;
}

SampleDetector::SampleDetector(double thresh) : dThresh(thresh) {
    LOG(INFO) << "Current config: thresh:" << dThresh;
}


int SampleDetector::init(const char *detectModelXMLPath) {
    // 检测模型
    if(!verifyXMLPath(detectModelXMLPath))
        return ERROR_INVALID_INIT_ARGS;
    Core ie;
    dNetwork = ie.ReadNetwork(detectModelXMLPath);
    dNetwork.setBatchSize(1);

    // 获取输入输出信息
    LOG(INFO) << "Preparing detect input blobs";
    dInputInfo = dNetwork.getInputsInfo().begin()->second;
    dInputName = dNetwork.getInputsInfo().begin()->first;
    // dInputInfo->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    dInputInfo->setLayout(Layout::NCHW);
    dInputInfo->setPrecision(Precision::FP32);


    LOG(INFO) << "Loading model to the device";
    // std::map<std::string, std::string> config = {
    //    };
    std::map<std::string, std::string> config = {
    //     {InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, "1"},
    //    {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES},
    //     {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "1"}
    };
    dExecutableNetwork = ie.LoadNetwork(dNetwork, "CPU", config);
    detect_request = dExecutableNetwork.CreateInferRequest();
    dInputC = dInputInfo->getTensorDesc().getDims()[1];
    dInputH = dInputInfo->getTensorDesc().getDims()[2];
    dInputW = dInputInfo->getTensorDesc().getDims()[3];

    input_ptr = detect_request.GetBlob(dInputName);

    // 获取输出信息
    OutputsDataMap dOutputsInfo(dNetwork.getOutputsInfo());
    DataPtr dClsInfo{nullptr};
    DataPtr dLocInfo{nullptr};
    // mobilenetv2_optim FPN ASFF SSH 811 813
    for (const auto& out : dOutputsInfo) {
        LOG(INFO)<< out.first;
        if(out.first=="811"){
            dClsName = out.first;
            dClsInfo = out.second;
        }
        if(out.first=="813"){
            dLocName = out.first;
            dLocInfo = out.second;
        }
    }
    // SizeVector dClsDims= dClsInfo->getTensorDesc().getDims();
    // SizeVector dLocDims= dLocInfo->getTensorDesc().getDims();
    // dClsObjects = dClsDims[2];
    // dLocObjects = dLocDims[2];

    dInputW_mul_H = dInputH*dInputW;
    dInputW_mul_H_mul_2 = 2*dInputW_mul_H;
    feat_w=int(dInputW/down_ratio);
    feat_h=int(dInputH/down_ratio);
    spacial_size = feat_h*feat_w;

    for(int i=0; i<256; i++)
        norm_data[i] = (i-127.5)/128.0;


    for(int i=0; i<4; i++)
        warm();

    LOG(INFO) << "Done.";
    return SampleDetector::INIT_OK;
}

void SampleDetector::unInit() {
}



void SampleDetector::decode(const float* heatmap, const float* ltrb, float thresh,
        float nms_thresh, std::vector<bbox> &faces){
    int ids_num[2] = {0};
    int pos = 0;
    int label = 0;
    float x=0,y=0;
    for(int i = 0; i < feat_h; i++){
        int i_mul_feat_w = i*feat_w;
        for (int j = 0; j < feat_w; j++) {
            x = heatmap[i_mul_feat_w + j];
            y = heatmap[spacial_size+i_mul_feat_w + j];
            if(std::max(x, y)>thresh){
                label = (x>y)?0:1;
                pos = ids_num[label]*2;
               
                ids[label][pos] =i;
                ids[label][pos+1]=j;
                ids_num[label]++;
            }
        }
    }

    int id_h, id_w, index;
    float c_x, c_y, l, t, r, b;
    bbox det;
    for(int k=0; k<2; k++){
        int k_mul_spacial_size = k*spacial_size;
        for(int i=0; i<ids_num[k]; i++){
            pos = i*2;
            id_h = ids[k][pos];
            id_w = ids[k][pos+1];
            index = id_h*feat_w + id_w;
            c_y = id_h*down_ratio;
            c_x = id_w*down_ratio;
            l = ltrb[index];
            t = ltrb[spacial_size+index];
            r = ltrb[2*spacial_size+index];
            b = ltrb[3*spacial_size+index];
            
            det.x1 = c_x-l;
            det.y1 = c_y-t;
            det.x2 = c_x+r;
            det.y2 = c_y+b;
            det.label = k;
            det.s = heatmap[k_mul_spacial_size +index];
            faces.emplace_back(det);
        }
    }
    // nms(faces, nms_thresh);
    softnms(faces, nms_thresh, 1, 0.5, dThresh);
}



void SampleDetector::warm(){
    auto input_data = input_ptr->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
    cv::Mat img = cv::Mat::zeros(dInputH, dInputW, CV_8UC3);
    int resize_rows = img.rows;
    int resize_cols = img.cols;
    int resize_step = img.step;
    uchar* resize_data = img.data;
    for(int h=0; h< dInputH; h++){
        uchar* uc_pixel = resize_data + h*resize_step;
        int dInputW_mul_h = h * dInputW;
        for (int w = 0; w < dInputW; w++) {
            input_data[dInputW_mul_h + w] = norm_data[uc_pixel[2]];
            input_data[dInputW_mul_H + dInputW_mul_h + w] = norm_data[uc_pixel[1]];
            input_data[dInputW_mul_H_mul_2 + dInputW_mul_h + w] = norm_data[uc_pixel[0]];
            uc_pixel+=3;
        }
    }
    detect_request.Infer();
}


STATUS SampleDetector::processImage(const cv::Mat &cv_image, std::vector<Object> &result) {
    // double start = static_cast<double>(cv::getTickCount());
    // 检测
    if (cv_image.empty()) {
        LOG(ERROR) << "Invalid input!";
        return ERROR_INVALID_INPUT;
    }
    int oriW = cv_image.cols;
    int oriH = cv_image.rows;

    input_img = cv_image.clone();
    if (input_img.channels() == 4)
        cv::cvtColor(input_img, input_img, cv::COLOR_BGRA2BGR);
    float sw = 1.0 * oriW / dInputW;
    float sh = 1.0 * oriH / dInputH;
    float scale = sw > sh ? sw : sh;
    float vscale = 1/scale;
    
    if(vscale != 1) {
        cv::resize(input_img, resize_img, cv::Size(), vscale, vscale, cv::INTER_NEAREST);
    }
    else {
        resize_img = input_img.clone();
    }

    // double end = static_cast<double>(cv::getTickCount());
    // std::cout<<"Resize: "<<(end - start) / cv::getTickFrequency()<<std::endl;

    auto input_data = input_ptr->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();  


    int resize_rows = resize_img.rows;
    int resize_cols = resize_img.cols;
    int resize_step = resize_img.step;
    uchar* resize_data = resize_img.data;
    for(int h=0; h< dInputH; h++){
        uchar* uc_pixel = resize_data + h*resize_step;
        int dInputW_mul_h = h * dInputW;
        for (int w = 0; w < dInputW; w++) {
            if(h<resize_rows && w<resize_cols){
                input_data[dInputW_mul_h + w] = norm_data[uc_pixel[2]];
                input_data[dInputW_mul_H + dInputW_mul_h + w] = norm_data[uc_pixel[1]];
                input_data[dInputW_mul_H_mul_2 + dInputW_mul_h + w] = norm_data[uc_pixel[0]];
                uc_pixel+=3;
            }else{
                input_data[dInputW_mul_h + w] = norm_data[0];
                input_data[dInputW_mul_H + dInputW_mul_h + w] = norm_data[0];
                input_data[dInputW_mul_H_mul_2 + dInputW_mul_h + w] = norm_data[0];
            }
        }
    }



    // start = static_cast<double>(cv::getTickCount());
    // std::cout<<"Input: "<<(start - end) / cv::getTickFrequency()<<std::endl;

    detect_request.Infer();

    const Blob::Ptr cls_blob = detect_request.GetBlob(dClsName);
    const float *cls_ptr = cls_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>(); 

    const Blob::Ptr loc_blob = detect_request.GetBlob(dLocName);
    const float *loc_ptr = loc_blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>(); 

    // end = static_cast<double>(cv::getTickCount());
    // std::cout<<"Infer: "<<(end-start) / cv::getTickFrequency()<<std::endl;

    std::vector<bbox> faces;
    decode(cls_ptr, loc_ptr, dThresh, 0.4, faces);

    // start = static_cast<double>(cv::getTickCount());
    // std::cout<<"Decode: "<<(start - end) / cv::getTickFrequency()<<std::endl;

    int x1, y1, x2, y2;
    bbox det;
    for(int i=0;i<faces.size(); i++){
        det = faces.at(i);
        x1 = std::round(det.x1*scale);
        y1 = std::round(det.y1*scale);
        x2 = std::round(det.x2*scale);
        y2 = std::round(det.y2*scale);
        x1 = (x1<0) ? 0 : x1;
        y1 = (y1<0) ? 0 : y1;
        x2 = (x2>oriW) ? oriW : x2;
        y2 = (y2>oriH) ? oriH : y2;

        result.emplace_back(SampleDetector::Object({
            det.s,
            labels[det.label],
            cv::Rect(x1, y1, x2 - x1, y2 - y1)
        })); 
    }
    return SampleDetector::PROCESS_OK;
}


bool SampleDetector::setThresh(double thresh) {
    dThresh = thresh;
    return true;
}

inline bool SampleDetector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

void SampleDetector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(),
		[](const bbox& a, const bbox& b)
	{
		return a.s > b.s;
	});
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


void SampleDetector::softnms(std::vector<bbox> &input_boxes, float NMS_THRESH,
        int method, float sigma, float epsilon){
    std::sort(input_boxes.begin(), input_boxes.end(),
		[](const bbox& a, const bbox& b)
	{
		return a.s > b.s;
	});
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);

            float weight=1.0;
            if(method==1) // linear
                if (ovr>NMS_THRESH) weight-=ovr;
            else if(method==2) // gaussian
                weight = std::exp((ovr*ovr)/sigma*-1);
            else if(method==3){ // diou nms
                float xx3 = std::min(input_boxes[i].x1, input_boxes[j].x1);
                float yy3 = std::min(input_boxes[i].y1, input_boxes[j].y1);
                float xx4 = std::max(input_boxes[i].x2, input_boxes[j].x2);
                float yy4 = std::max(input_boxes[i].y2, input_boxes[j].y2);

                float c1_x = (input_boxes[i].x1+input_boxes[i].x2)/2;
                float c1_y = (input_boxes[i].y1+input_boxes[i].y2)/2;
                float c2_x = (input_boxes[j].x1+input_boxes[j].x2)/2;
                float c2_y = (input_boxes[j].y1+input_boxes[j].y2)/2;
                float c_diag = (c1_x-c2_x)*(c1_x-c2_x)+(c1_y-c2_y)*(c1_y-c2_y);
                float inter_diag = (xx3-xx4)*(xx3-xx4)+(yy3-yy4)*(yy3-yy4);
                if((ovr-inter_diag/c_diag)>NMS_THRESH) weight=0;
            }
            else
                if(ovr>NMS_THRESH) weight=0; // original nms
            
            input_boxes[j].s = input_boxes[j].s * weight;
            if(input_boxes[j].s<epsilon){
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void SampleDetector::set_view(bool view){
    viewImg=view;
}