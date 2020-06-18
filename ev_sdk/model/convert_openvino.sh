#!/bin/bash

# 在这里编写OpenVINO模型转换
# 系统会将所选择的原始模型放在目录/usr/local/ev_sdk/model下，转换后的OpenVINO模型请放在model/openvino目录下，
# 建议所有路径都使用绝对路径

# 获取当前脚本的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
