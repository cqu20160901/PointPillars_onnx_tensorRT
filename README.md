# PointPillars_onnx_tensorRT_Cplusplus

本示例中对 pillar 的计算流程用numpy进行了重新，对后处理的mns用一个2D的nms近似代替（只为验证模型结果是对的，不可实际使用）。

运行onnx 测试效果依赖环境：numpy、open3d、onnxruntime


# 导出onnx 

[【参考链接】](https://github.com/zhulf0804/PointPillars/tree/feature/deployment)

# onnx 效果

## pytorch 效果

![image](https://github.com/user-attachments/assets/1e4887d6-c7da-421a-81ea-70c8403401cf)


## onnx 效果

![image](https://github.com/user-attachments/assets/0213f7f8-8459-4eed-a737-bb6c995058f6)


## tensorrt 效果

![image](https://github.com/user-attachments/assets/f2932675-a0a1-4d5c-82c0-68cfd1f207ce)


## onnx 和 tensorrt 输出结果

### onnx 推理输出

![image](https://github.com/user-attachments/assets/c41c1441-121c-4582-993f-96f9501c22f6)

### tensorrt 推理输出

![image](https://github.com/user-attachments/assets/548597a3-19a3-416d-86c2-0cf3ab97debb)



