cmake_minimum_required(VERSION 3.5)
project(ji)
MESSAGE("Coding")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 设置RPATH，使编译好的libji.so在运行时能够找到OpenVINO的依赖库
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/intel/openvino_2020.1.023/deployment_tools/ngraph/lib:\
/opt/intel/opencl:\
/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/hddl/lib:\
/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/gna/lib:\
/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/mkltiny_lnx/lib:\
/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/tbb/lib:\
/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/lib/intel64:\
${CMAKE_INSTALL_RPATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
find_package(PkgConfig REQUIRED)

# 设置OpenVINO InferenceEngine的cmake查找路径
set(InferenceEngine_DIR "/opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/share")
find_package(InferenceEngine 2.1)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rd/cJSON)

# 编译libji.so
list(APPEND libji_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ji.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SampleDetector.cpp)
add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        cjson
        # WKTParser
        ${InferenceEngine_LIBRARIES})

# 安装SampleDetector所依赖的libdarknet.so, 请在此处将自己依赖的库安装到/usr/local/ev_sdk/lib下
##install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/darknet/lib/libdarknet.so DESTINATION lib)



# 以下是EV_SDK所依赖和需要测试接口的程序&库，请勿修改！！！
# --- BEGIN EV_SDK DEPENDENCIES ---
# 编译 JI API 测试程序
add_executable(test-ji-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)

target_link_libraries(test-ji-api
        ${OpenCV_LIBS}
        cjson
        glog
        ${PROJECT_NAME}
        )
# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
# Install test-ji-api
install(TARGETS test-ji-api DESTINATION bin)
# ---  END EV_SDK DEPENDENCIES  ---