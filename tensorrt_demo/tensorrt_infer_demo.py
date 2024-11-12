import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
max_num_points = 32
max_voxels = 20000
coors_range = [0, -39.68, -3, 69.12, 39.68, 1]


def box_iou(src_box1, src_box2):
    box1 = src_box1.copy()
    box2 = src_box2.copy()

    box1[3:5] += box1[0:2]
    box2[3:5] += box2[0:2]

    inter_min = np.maximum(box1[0:2], box2[0:2])
    inter_max = np.minimum(box1[3:5], box2[3:5])
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_volume = np.prod(inter_dims)

    volume1 = np.prod(box1[3:5] - box1[0:2])
    volume2 = np.prod(box2[3:5] - box2[0:2])
    iou = inter_volume / (volume1 + volume2 - inter_volume + 1e-6)

    return iou


def nms_3d_fake(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(indices) > 0:
        current = indices[0]
        selected_indices.append(current)

        remaining_boxes = boxes[indices[1:]]
        ious = np.array([box_iou(boxes[current], box) for box in remaining_boxes])

        indices = indices[1:][ious < iou_threshold]

    return selected_indices


def read_points(file_path, dim=4):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)


def convert_to_pointpillars(points, voxel_size, point_cloud_range, max_num_points, max_voxels):
    # 确定体素网格范围
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    # 计算网格尺寸
    grid_size = np.round((np.array(point_cloud_range[3:]) - np.array(point_cloud_range[:3])) / voxel_size).astype(int)

    # 生成体素网格
    x_bins = np.arange(x_min, x_max, voxel_size[0])
    y_bins = np.arange(y_min, y_max, voxel_size[1])

    # 计算每个pillar的点
    pillars = {}
    for point in points:
        if (x_min <= point[0] < x_max) and (y_min <= point[1] < y_max) and (z_min <= point[2] < z_max):
            x_idx = np.digitize(point[0], x_bins) - 1
            y_idx = np.digitize(point[1], y_bins) - 1

            # 检查索引是否在有效范围内
            if (0 <= x_idx < grid_size[0]) and (0 <= y_idx < grid_size[1]):
                key = (x_idx, y_idx)
                if key not in pillars:
                    pillars[key] = []
                pillars[key].append(point)

    # 将pillar转换为数组形式，并补充全0点
    pillar_array = []
    coors = []
    npoints_per_pillar = []
    pillars_num = 0

    zero_point = np.zeros((1, points.shape[1]))  # 创建一个全0点的数组
    for key, point_list in pillars.items():
        # 限制每个pillar最多包含max_num_points个点
        if len(point_list) > max_num_points:
            point_list = point_list[:max_num_points]

        npoints_per_pillar.append(len(point_list))
        pillars_num += 1

        # 补充全0点
        while len(point_list) < max_num_points:
            point_list.append(zero_point[0])

        pillar_array.append(np.array(point_list))
        coors.append(np.array([key[0], key[1], 0]))  # 使用Pillar的坐标（x_idx, y_idx）

        # 限制pillar数量
        if len(pillar_array) >= max_voxels:
            break

    return pillar_array, coors, npoints_per_pillar, pillars_num



def get_input(point_patch):
    raw_point = read_points(point_patch)
    pointpillars, coors, npoints_per_pillar, pillar_num = convert_to_pointpillars(raw_point, voxel_size, point_cloud_range, max_num_points, max_voxels)

    input_pillars = np.array(pointpillars).reshape(pillar_num, max_num_points, -1).astype(np.float32)
    coors = np.array(coors).reshape(pillar_num, -1)
    input_npoints_per_pillar = np.array(npoints_per_pillar).reshape((-1)).astype(np.int32)

    input_coors_batch = []
    for i, cur_coors in enumerate(coors):
        input_coors_batch.append(np.pad(cur_coors, (1, 0), mode='constant', constant_values=i))
    input_coors_batch = np.array(input_coors_batch).reshape(pillar_num, -1)

    return input_pillars, input_npoints_per_pillar, input_coors_batch


def get_engine(trt_engine_path):
    trt_logger = trt.Logger(trt.Logger.WARNING)

    with open(trt_engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    builder = trt.Builder(trt_logger)

    if engine.num_optimization_profiles > 0:
        context.active_optimization_profile = 0
    return engine, context, builder


def postprocess(h_output):
    # x, y, z, length, width, height, yaw, conf0, conf1, conf2, cls
    bbox_pred, bbox_cls_pred, bbox_dir_cls_pred = h_output[:, :7], h_output[:, 7:10], h_output[:, 10]
    bbox_cls_pred = np.max(bbox_cls_pred, axis=1)

    keep_index = nms_3d_fake(bbox_pred, bbox_cls_pred, iou_threshold=0.1)

    bbox_pred = bbox_pred[keep_index]
    bbox_cls_pred = bbox_cls_pred[keep_index]
    bbox_dir_cls_pred = bbox_dir_cls_pred[keep_index]

    return bbox_pred, bbox_cls_pred, bbox_dir_cls_pred


def trt_infer(trt_engine_path, point_patch):
    engine, context, builder = get_engine(trt_engine_path)
    input_pillars, input_npoints_per_pillar, input_coors_batch = get_input(point_patch)

    input_shape_1 = input_pillars.shape
    input_shape_2 = input_coors_batch.shape
    input_shape_3 = input_npoints_per_pillar.shape

    context.set_binding_shape(0, input_shape_1)
    context.set_binding_shape(1, input_shape_2)
    context.set_binding_shape(2, input_shape_3)

    input_size_1 = trt.volume(input_shape_1) * np.dtype(np.float32).itemsize
    input_size_2 = trt.volume(input_shape_2) * np.dtype(np.int32).itemsize
    input_size_3 = trt.volume(input_shape_3) * np.dtype(np.int32).itemsize
    output_shape = engine.get_binding_shape(engine.get_binding_index("output_x"))
    output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

    d_input_1 = cuda.mem_alloc(input_size_1)
    d_input_2 = cuda.mem_alloc(input_size_2)
    d_input_3 = cuda.mem_alloc(input_size_3)
    d_output = cuda.mem_alloc(output_size)

    h_input_1 = input_pillars.astype(np.float32)
    h_input_2 = input_coors_batch.astype(np.int32)
    h_input_3 = input_npoints_per_pillar.astype(np.int32)
    h_output = np.zeros(output_shape, dtype=np.float32)

    cuda.memcpy_htod(d_input_1, h_input_1)
    cuda.memcpy_htod(d_input_2, h_input_2)
    cuda.memcpy_htod(d_input_3, h_input_3)

    context.execute_v2([d_input_1, d_input_2, d_input_3, d_output])

    cuda.memcpy_dtoh(h_output, d_output)

    print("Inference output:", h_output.shape)

    bbox_pred, bbox_cls_pred, bbox_dir_cls_pred = postprocess(h_output)

    print('obj num is:', bbox_pred.shape[0])
    

if __name__ == '__main__':
    print('This is main ....')
    trt_engine_path = './pointpillars.trt'
    point_patch = './test.bin'
    trt_infer(trt_engine_path, point_patch)