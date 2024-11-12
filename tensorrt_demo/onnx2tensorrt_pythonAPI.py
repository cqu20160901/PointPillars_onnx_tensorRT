import tensorrt as trt
import onnx

# 定义 ONNX 模型路径和输出的 TensorRT 引擎路径
onnx_model_path = "pointpillars.onnx"
trt_engine_path = "pointpillars.trt"

# 创建一个 TensorRT Logger，用于记录日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 创建 TensorRT 构建器
builder = trt.Builder(TRT_LOGGER)

# 使用 kEXPLICIT_BATCH 标志来创建网络
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags=network_flags)  # 使用 kEXPLICIT_BATCH

# 使用 ONNX 解析器来解析 ONNX 模型
onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

# 读取并解析 ONNX 模型
with open(onnx_model_path, 'rb') as model_file:
    if not onnx_parser.parse(model_file.read()):
        print("ERROR: Failed to parse the ONNX model.")
        for error in range(onnx_parser.num_errors):
            print(onnx_parser.get_error(error))
        exit(1)

# 创建一个构建配置对象
config = builder.create_builder_config()

# 设置最大工作空间大小 (1GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 设置最大批次大小
builder.max_batch_size = 20000  # 设置最大批量大小

# 配置精度模式，例如支持 FP16（如果硬件支持）
# if builder.platform_has_fast_fp16:
#     config.set_flag(trt.BuilderFlag.FP16)

# 创建优化配置文件（用于动态形状输入）
profile = builder.create_optimization_profile()

print(network.get_input(0).shape)
print(network.get_input(1).shape)
print(network.get_input(2).shape)


input_tensor = network.get_input(0)
min_shape = (1, input_tensor.shape[1], 4)
opt_shape = (10000, input_tensor.shape[1], 4)
max_shape = (20000, input_tensor.shape[1], 4)
profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)


input_tensor = network.get_input(1)
min_shape = (1, 4)
opt_shape = (10000, 4)
max_shape = (20000, 4)
profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

input_tensor = network.get_input(2)
min_shape = trt.Dims([1])
opt_shape = trt.Dims([10000])
max_shape = trt.Dims([20000])
profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

# 将优化配置文件应用到配置对象
config.add_optimization_profile(profile)

# 构建 TensorRT 引擎
engine = builder.build_engine(network, config)

# 检查引擎是否成功创建
if engine is None:
    print("ERROR: Failed to build the TensorRT engine.")
    exit(1)

# 保存 TensorRT 引擎到文件
with open(trt_engine_path, 'wb') as f:
    f.write(engine.serialize())

print(f"TensorRT engine saved to {trt_engine_path}")
