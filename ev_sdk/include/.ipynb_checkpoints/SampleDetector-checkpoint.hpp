//
// Created by hrh on 2019-09-02.
//

#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <inference_engine.hpp>
#include <map>

#define STATUS int

using namespace InferenceEngine;

/**
 * 使用OpenVINO转换的行人检测模型，模型基于ssd inception v2 coco训练得到，模型转换请参考：
 * https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#ssd_single_shot_multibox_detector_topologies
 */

class SampleDetector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    SampleDetector(double thresh);

    /**
     * 初始化模型
     * @param[in] modelXMLPath OpenVINO IR模型的XML文件路径
     * @return 如果初始化正常，INIT_OK
     */
    STATUS init(const char *detectModelXMLPath);

    /**
     * 反初始化函数
     */
    void unInit();
    

    /**
     * 对cv::Mat格式的图片进行分类，并输出预测分数前top排名的目标名称到mProcessResult
     * @param[in] image 输入图片
     * @param[out] detectResults 检测到的结果
     * @return 如果处理正常，则返回PROCESS_OK，否则返回`ERROR_*`
     */
    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults);

    bool setThresh(double thresh);

    struct bbox{
        float x1;
        float y1;
        float x2;
        float y2;
        float s;
        int label;
    };
    void decode(const float* heatmap, const float* ltrb, float thresh, float nms_thresh, std::vector<bbox> &faces);
    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);
    void softnms(std::vector<bbox> &input_boxes, float NMS_THRESH, int method, float sigma, float epsilon);
    static inline bool cmp(bbox a, bbox b);
    void set_view(bool view);
    void warm();

public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;

    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

private:
 // detect face
    ExecutableNetwork dExecutableNetwork;
    CNNNetwork dNetwork;
    InferRequest detect_request;
    InputInfo::Ptr dInputInfo{nullptr};
    Blob::Ptr input_ptr;
    std::string dInputName;
    double dThresh = 0.35;
    int dInputC = 3;
    int dInputW = 800;
    int dInputH = 640;
    const int down_ratio=4;
    int dInputW_mul_H;
    int dInputW_mul_H_mul_2;
    int feat_w;
    int feat_h;
    int spacial_size;
    std::string dClsName;
    std::string dLocName;
    // int dClsObjects;
    // int dLocObjects;
    bool viewImg=false;
    std::string labels[2]={"mask", "nomask"};
    float norm_data[256]={0};

    // Fuction decode
    int ids[2][200*200*2];

    // ProcessImg
    cv::Mat input_img;
    cv::Mat resize_img;
};

#endif //JI_SAMPLEDETECTOR_HPP
