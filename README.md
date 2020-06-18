## 极市口罩检测 No.1
<!-- ![](res/rank.jpg) -->
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="res/rank.jpg">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;"> </div>
</center>

问题描述：检测图片中戴口罩和没戴口罩的人脸，F1-score 和 FPS 需要 trade-off

<!-- ![](res/score.jpg) -->
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="res/score.jpg">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">分数计算</div>
</center>

模型：
```
Centernet + ttfnet
Backone：mobilenetv2(0.5，relu替换relu6)
neck：   4层FPN + ASFF融合 + SSH
后处理：  softnms
大小大概：2 Mb
(ps: 由于平台C++测速bug，导致我以为这个模型速度不快，训练一次后放弃优化，故精度是没有优化过的, 800x640分辨率F1-score大概0.76)
```


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="res/ttfnet.jpg">
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">原始ttfnet结构</div>
</center>


代码层面优化：
```
1、查表法减均值除方差
2、图像W/H原比例缩放和填充input_ptr合并
3、尽量减少多余乘法，数组替换vector
(resize也有点耗时，听说Opencv4.2加速了)
```