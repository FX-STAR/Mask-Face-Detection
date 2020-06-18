#include "SampleDetector.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>


int main(int argc, char** argv){
    auto *detector = new SampleDetector(atof(argv[2]));
    int iRet = detector->init("~/Face/Openvino/model/vino/model.xml");
    detector->set_view(true);
    cv::Mat img = cv::imread(argv[1]);
    std::vector<SampleDetector::Object> result;
    detector->processImage(img, result);
    return 0;
}