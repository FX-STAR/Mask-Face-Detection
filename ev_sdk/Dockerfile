FROM uhub.service.ucloud.cn/eagle_nest/cuda10.1-cudnn7.5.1-dev-ubuntu16.04-opencv4.1.1-torch1.4.0-openvino2020r1

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk

RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*