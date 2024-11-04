import os
import numpy as np
import onnxruntime as ort
import open3d as o3d

voxel_size = [0.16, 0.16, 4]
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
max_num_points = 32
max_voxels = 20000
coors_range = [0, -39.68, -3, 69.12, 39.68, 1]

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

LINES = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [2, 6],
    [7, 3],
    [1, 5],
    [4, 0]]


def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    density = npy[:, 3]
    colors = [[item, item, item] for item in density]
    ply.colors = o3d.utility.Vector3dVector(colors)
    return ply


def vis_core(plys):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('viewpoint.json')
    for ply in plys:
        vis.add_geometry(ply)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                              dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                       dtype=np.float32)  # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for i in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_pc(pc, bboxes=None, labels=None):
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame])
        return

    if len(bboxes.shape) == 2:
        bboxes = bbox3d2corners(bboxes)

    vis_objs = [pc, mesh_frame]
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        if labels is None:
            color = [1, 0, 0]
        else:
            if labels[i] >= 0 and labels[i] < 3:
                color = COLORS[int(labels[i])]
            else:
                color = COLORS[-1]
        vis_objs.append(bbox_obj(bbox, color=color))
    vis_core(vis_objs)


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts


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


def detect(onnx_path, point_cloud_path):
    raw_point = read_points(point_cloud_path)
    raw_point = point_range_filter(raw_point)
    pointpillars, coors, npoints_per_pillar, pillar_num = convert_to_pointpillars(raw_point, voxel_size,
                                                                                  point_cloud_range, max_num_points,
                                                                                  max_voxels)

    input_pillars = np.array(pointpillars).reshape(pillar_num, max_num_points, -1).astype(np.float32)
    coors = np.array(coors).reshape(pillar_num, -1)
    input_npoints_per_pillar = np.array(npoints_per_pillar).reshape((-1)).astype(np.int32)

    input_coors_batch = []
    for i, cur_coors in enumerate(coors):
        input_coors_batch.append(np.pad(cur_coors, (1, 0), mode='constant', constant_values=i))
    input_coors_batch = np.array(input_coors_batch).reshape(pillar_num, -1)

    ort_session = ort.InferenceSession(onnx_path)
    ort_out = ort_session.run(None, {'input_pillars': input_pillars, 'input_coors_batch': input_coors_batch,
                                     'input_npoints_per_pillar': input_npoints_per_pillar})

    output = ort_out[0]
    # x, y, z, length, width, height, yaw, conf0, conf1, conf2, cls
    bbox_pred, bbox_cls_pred, bbox_dir_cls_pred = output[:, :7], output[:, 7:10], output[:, 10]
    bbox_cls_pred = np.max(bbox_cls_pred, axis=1)

    keep_index = nms_3d_fake(bbox_pred, bbox_cls_pred, iou_threshold=0.1)

    bbox_pred = bbox_pred[keep_index]
    bbox_cls_pred = bbox_cls_pred[keep_index]
    bbox_dir_cls_pred = bbox_dir_cls_pred[keep_index]
    
    print('obj num is:', len(bbox_pred))

    vis_pc(raw_point, bboxes=bbox_pred)


if __name__ == '__main__':
    print('This is main ....')
    onnx_path = './pointpillars.onnx'
    point_cloud_path = './test.bin'
    detect(onnx_path, point_cloud_path)
