# PointPillars_onnx_tensorRT_Cplusplus

本示例中对 pillar 的计算流程用numpy进行了重实现，对后处理的mns用一个2D的nms近似代替（只为验证模型结果是对的，不可实际使用）。

本示例中，包含完整的：测试脚本代码、模型、测试数据、测试结果。

运行onnx 测试依赖环境：numpy、open3d、onnxruntime

TensorRT版本：TensorRT-8.6.1.6

***特别说明：本示例中没有使用 3d_nms, 只用 2D 的 nms 进行了简单处理，不能实际使用。***


# 导出onnx 

[【参考链接】](https://github.com/zhulf0804/PointPillars/tree/feature/deployment)

模型输入输出维度：

![image](https://github.com/user-attachments/assets/ab78addf-38de-4c16-8f5c-a9378b1fd06d)


## pytorch 效果

![image](https://github.com/user-attachments/assets/1e4887d6-c7da-421a-81ea-70c8403401cf)


对应图像

![image](https://github.com/cqu20160901/PointPillars_onnx_tensorRT/blob/main/onnx_demo/test.png)

## onnx 效果

![image](https://github.com/user-attachments/assets/0213f7f8-8459-4eed-a737-bb6c995058f6)


## tensorrt 效果(由于tensorrt在服务器上运行的，不能可视化，结果写入txt再拉下进行可视化,[【可视化脚本参考】](https://github.com/zhulf0804/PointPillars/blob/feature/deployment/deployment/vis_infer_result.py))

![image](https://github.com/user-attachments/assets/f2932675-a0a1-4d5c-82c0-68cfd1f207ce)


## onnx 和 tensorrt 输出结果

### onnx 推理输出

![image](https://github.com/user-attachments/assets/c41c1441-121c-4582-993f-96f9501c22f6)

### tensorrt 推理输出

![image](https://github.com/user-attachments/assets/548597a3-19a3-416d-86c2-0cf3ab97debb)



