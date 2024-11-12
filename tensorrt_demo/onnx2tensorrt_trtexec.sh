#!/bin/bash
ONNX_MODEL_PATH=/root/autodl-tmp/pointpillars_tensorrt/models/onnx2trtexec/pointpillars.onnx
OUTPUT_PATH=/root/autodl-tmp/pointpillars_tensorrt/models/onnx2trtexec/pointpillars.trt
TRT_SRC_BIN_PATH=/root/autodl-tmp/TensorRT-8.6.1.6/bin


$TRT_SRC_BIN_PATH/trtexec --onnx=$ONNX_MODEL_PATH \--verbose \
    --fp16 \
    --saveEngine=$OUTPUT_PATH \
    --minShapes=input_pillars:200x32x4,input_coors_batch:200x4,input_npoints_per_pillar:200 \
    --optShapes=input_pillars:10000x32x4,input_coors_batch:10000x4,input_npoints_per_pillar:10000 \
    --maxShapes=input_pillars:20000x32x4,input_coors_batch:20000x4,input_npoints_per_pillar:20000 \
    --workspace=4096 \
    --exportProfile="./time.json" 
    >export_trt.log 2>&1 &
