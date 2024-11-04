# PointPillars_tensorRT_Cplusplus
PointPillars 部署tensorrt

本示例中对 pillar 的计算流程用numpy进行了重新，对后处理的3D_mns用一个近似代替（只为验证模型结果是对的，没有大量测试实际使用可能会有问题）。

运行onnx 测试效果依赖环境：numpy、open3d、onnxruntime


# 导出onnx 

[【参考链接】](https://github.com/zhulf0804/PointPillars/tree/feature/deployment)

# onnx 效果

pytorch 效果

![image](https://github.com/user-attachments/assets/f177a1bd-7862-43a7-bb6a-593975fe7b88)

onnx 效果

![image](https://github.com/user-attachments/assets/cf5baee9-04cd-4b53-8805-1dcf5b0849e1)
