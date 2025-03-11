import cv2
import imgviz
import numpy as np
import torch
from functorch import combine_state_for_ensemble
import open3d
from scipy.spatial.transform import Rotation
import trimesh
import scipy
from itertools import permutations
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.ndimage import binary_fill_holes

class BoundingBox():
    def __init__(self):
        super(BoundingBox, self).__init__()
        self.extent = None
        self.R = None
        self.center = None
        self.points3d = None    # (8,3)

def update_vmap(models, optimiser):
    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]
    optimiser.add_param_group({"params": params})  # imap b l
    return (fmodel, params, buffers)

def enlarge_bbox(bbox, scale, w, h):
    assert scale >= 0
    # print(bbox)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_y == 0 or margin_x == 0:
        return None
    # assert margin_x != 0
    # assert margin_y != 0
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = np.clip(min_x, 0, w-1)
    min_y = np.clip(min_y, 0, h-1)
    max_x = np.clip(max_x, 0, w-1)
    max_y = np.clip(max_y, 0, h-1)

    bbox_enlarged = [int(min_x), int(min_y), int(max_x), int(max_y)]
    return bbox_enlarged

def get_bbox2d(obj_mask, bbox_scale=1.0):
    contours, hierarchy = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                          -2:]
    # # Find the index of the largest contour
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # Concatenate all contours
    if len(contours) == 0:
        return None
    cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(cnt)  # todo if multiple contours, choose the outmost one?
    # x, y, w, h = cv2.boundingRect(contours)
    bbox_enlarged = enlarge_bbox([x, y, x + w, y + h], scale=bbox_scale, w=obj_mask.shape[1], h=obj_mask.shape[0])
    return bbox_enlarged

def get_bbox2d_batch(img):
    b,h,w = img.shape[:3]
    rows = torch.any(img, axis=2)
    cols = torch.any(img, axis=1)
    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1)
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1)

    return rmins, rmaxs, cmins, cmaxs

# for association/tracking
class InstData:
    def __init__(self):
        super(InstData, self).__init__()
        self.bbox3D = None
        self.inst_id = None     # instance
        self.class_id = None    # semantic
        self.pc_sample = None
        self.merge_cnt = 0  # merge times counting
        self.cmp_cnt = 0
        
def box_filter(masks, classes, depth, inst_dict, intrinsic_open3d, T_CW, min_pixels=500, voxel_size=0.01):
    bbox3d_scale = 1.0  # 1.05
    inst_data = np.zeros_like(depth, dtype=np.int)
    for i in range(len(masks)):
        diff_mask = None
        inst_mask = masks[i]
        inst_id = classes[i]
        if inst_id == 0:
            continue
        inst_depth = np.copy(depth)
        inst_depth[~inst_mask] = 0.  # inst_mask
        # proj_time = time.time()
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        # print("proj time ", time.time()-proj_time)
        if len(inst_pc.points) <= 10:  # too small
            inst_data[inst_mask] = 0  # set to background
            continue
        if inst_id in inst_dict.keys():
            candidate_inst = inst_dict[inst_id]
            # iou_time = time.time()
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            # print("iou time ", time.time()-iou_time)
            # if indices empty
            candidate_inst.cmp_cnt += 1
            if len(indices) >= 1:
                candidate_inst.pc += inst_pc.select_by_index(indices)  # only merge pcs inside scale*bbox
                # todo check indices follow valid depth
                valid_depth_mask = np.zeros_like(inst_depth, dtype=np.bool)
                valid_pc_mask = valid_depth_mask[inst_depth!=0]
                valid_pc_mask[indices] = True
                valid_depth_mask[inst_depth != 0] = valid_pc_mask
                valid_mask = valid_depth_mask
                diff_mask = np.zeros_like(inst_mask)
                # uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                #                                  T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3, :3], None)
                # uv = np.round(uv_opencv).squeeze().astype(int)
                # u = uv[:, 0].reshape(-1, 1)
                # v = uv[:, 1].reshape(-1, 1)
                # vu = np.concatenate([v, u], axis=-1)
                # valid_mask = np.zeros_like(inst_mask)
                # valid_mask[tuple(vu.T)] = True
                # # cv2.imshow("valid", (inst_depth!=0).astype(np.uint8)*255)
                # # cv2.waitKey(1)
                diff_mask[(inst_depth != 0) & (~valid_mask)] = True
                # cv2.imshow("diff_mask", diff_mask.astype(np.uint8) * 255)
                # cv2.waitKey(1)
            else:   # merge all for scannet
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # downsample_time = time.time()
            # adapt_voxel_size = np.maximum(np.max(candidate_inst.bbox3D.extent)/100, 0.1)
            candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size) # adapt_voxel_size
            # candidate_inst.pc = candidate_inst.pc.farthest_point_down_sample(500)
            # candidate_inst.pc = candidate_inst.pc.random_down_sample(np.minimum(len(candidate_inst.pc.points)/500.,1))
            # print("downsample time ", time.time() - downsample_time)  # 0.03s even
            # bbox_time = time.time()
            try:
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
            except RuntimeError:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # enlarge
            candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
        else:   # new inst
            # init new inst and new sem
            new_inst = InstData()
            new_inst.inst_id = inst_id
            smaller_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
            if np.sum(smaller_mask) < min_pixels:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            inst_depth_small = depth.copy()
            inst_depth_small[~smaller_mask] = 0
            inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
            new_inst.pc = inst_pc_small
            new_inst.pc = new_inst.pc.voxel_down_sample(voxel_size)
            try:
                inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            except RuntimeError:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            # scale up
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            new_inst.bbox3D = inst_bbox3D
            # update inst_dict
            inst_dict.update({inst_id: new_inst})  # init new sem

        # update inst_data
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1  # unsure area

    return inst_data

def accumulate_pointcloud(inst_id, inst_info_list, frame_samples, intrinsic_open3d, voxel_size=0.01):
    inst_pcs = None
    for idx in range(len(inst_info_list)):
        inst_info = inst_info_list[idx]
        frame = inst_info['frame']
        
        sample = frame_samples[frame]
        assert frame == sample['frame_id']
        obj_mask = sample['obj_mask']
        obj_mask = obj_mask == inst_id
        T_CW = np.linalg.inv(sample['T'])
        
        inst_depth = np.copy(sample['depth'])
        inst_depth[~obj_mask] = 0.  # obj_mask
        inst_pc = unproject_colored_pointcloud(sample['image'], inst_depth, intrinsic_open3d, T_CW)
        if inst_pcs is None:
            inst_pcs = inst_pc
        else:
            inst_pcs += inst_pc

    inst_pcs = inst_pcs.voxel_down_sample(voxel_size)
    return inst_pcs

def accumulate_pointcloud_tsdf(inst_id, inst_info_list, frame_samples, intrinsic_open3d, voxel_size=0.01, depth_scale=0.001, max_depth=6.0):
    volume = open3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4*voxel_size,
            color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    for idx in range(len(inst_info_list)):
        inst_info = inst_info_list[idx]
        frame = inst_info['frame']
        
        sample = frame_samples[frame]
        assert frame == sample['frame_id']
        obj_mask = sample['obj_mask']
        obj_mask = obj_mask == inst_id
        T_CW = np.linalg.inv(sample['T'])
        
        inst_depth = 1/depth_scale*np.copy(sample['depth'])
        inst_depth[~obj_mask] = 0.  # obj_mask
        inst_depth = inst_depth.transpose(1,0).astype(np.uint16)
        inst_color = sample['image'].transpose(1,0,2).astype(np.uint8)
        
        color = open3d.geometry.Image(inst_color)
        depth = open3d.geometry.Image(inst_depth)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic_open3d, T_CW)   

    inst_pcs = volume.extract_point_cloud()
    inst_pcs = inst_pcs.voxel_down_sample(voxel_size)
    cl, _ = inst_pcs.remove_radius_outlier(nb_points=100, radius=0.05)
    if np.asarray(cl.points).shape[0] < 100:
        print("too few points left after outlier rejection")
    else:
        inst_pcs = cl
        
    return inst_pcs

def get_bound(inst_pcs):
    try:
        transform, extents = trimesh.bounds.oriented_bounds(np.array(inst_pcs.points))  # pc
        transform = np.linalg.inv(transform)
        
    except scipy.spatial._qhull.QhullError:
        print("fail to get initial pose from instance point cloud")
        return None
    
    for i in range(extents.shape[0]):
        extents[i] = np.maximum(extents[i], 0.10)  # at least rendering 10cm

    bbox3D = BoundingBox()
    bbox3D.center = transform[:3, 3]
    bbox3D.R = transform[:3, :3]
    bbox3D.extent = extents
    
    min_extent = 0.05
    bbox3D.extent = np.maximum(min_extent, bbox3D.extent)
    return bbox3D

def get_obb(inst_info):
    bbox3D = BoundingBox()
    Two = np.copy(inst_info['T_obj'])
    scale_before = np.linalg.det(Two[:3, :3])**(1/3)
    Two[:3, :3] = Two[:3, :3]/scale_before
    center = Two[:3, 3]
    bbox3D.R = Two[:3, :3]
    bbox3D.center = center
    points_w = np.asarray(inst_info['pcs'].points)
    points_o = transform_pointcloud(points_w, np.linalg.inv(Two))
    extent = 2 * np.max(np.stack([np.max(points_o, axis=0), -np.min(points_o, axis=0)], axis=-1), axis=-1)
    extent = np.maximum(extent, 0.10)  # scale at least 10cm
    bbox3D.extent = extent
    inst_info['T_obj'][:3, :3] = Two[:3,:3] * np.max(extent/2)
    inst_info['bbox3D'] = bbox3D

def get_pose_from_pointcloud(inst_pcs):
    bbox3D = get_bound(inst_pcs)

    scale = np.max(bbox3D.extent)/2
    T_obj = np.eye(4)
    T_obj[:3, 3] = np.copy(bbox3D.center)
    T_obj[:3, :3] = np.copy(bbox3D.R)
    
    T_obj[:3, :3] *= scale
        
    return T_obj, bbox3D

def get_possible_transform_from_bbox():
    transform_list = []
    axes = np.eye(3)
    index = [0, 1, 2]
    axis_indices = list(permutations(index, 2)) 
    transform = np.eye(4)
    for axis_index in axis_indices: # 6
        for i in range(4): # *4
            x_axis = axes[np.asarray(axis_index[0])]
            y_axis = axes[np.asarray(axis_index[1])]
            if i == 1:
                x_axis *= -1
            elif i == 2:
                y_axis *= -1
            elif i == 3:
                x_axis *= -1
                y_axis *= -1
            z_axis = np.cross(x_axis, y_axis)
            rotation = np.vstack([x_axis, y_axis, z_axis]).T
            transform[:3, :3] = rotation
            transform_list.append(np.copy(transform))
        
    return transform_list
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def unproject_pointcloud(depth, intrinsic_open3d, T_CW):
    # depth, mask, intrinsic, extrinsic -> point clouds
    try:
        pc_sample = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(np.ascontiguousarray(depth)),
                                                                    intrinsic=intrinsic_open3d,
                                                                    extrinsic=T_CW,
                                                                    depth_scale=1.0,
                                                                    project_valid_depth_only=True)
    except RuntimeError:
        print('fail to unproject pointcloud')
    return pc_sample

def unproject_colored_pointcloud(rgb, depth, intrinsic_open3d, T_CW):
    # depth, mask, intrinsic, extrinsic -> point clouds
    try:
        rgb = open3d.geometry.Image(np.ascontiguousarray(rgb.transpose(1,0,2)))
        depth = open3d.geometry.Image(np.ascontiguousarray(1000*depth.transpose(1,0)))
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_trunc=8.0, convert_rgb_to_intensity=False)
        pc_sample = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic=intrinsic_open3d, extrinsic=T_CW)
    except RuntimeError:
        print('fail to unproject pointcloud')
    return pc_sample

def check_inside_ratio(pc, bbox3D):
    #  pc, bbox3d -> inside ratio
    indices = bbox3D.get_point_indices_within_bounding_box(pc.points)
    assert len(pc.points) > 0
    ratio = len(indices) / len(pc.points)
    # print("ratio ", ratio)
    return ratio, indices

def transform_pointcloud(cloud, T_rel):
    n = cloud.shape[0]
    cloud_hom = np.hstack((cloud, np.ones((n,1))))
    cloud_transformed = (T_rel.dot(cloud_hom.T)).T
    
    return cloud_transformed[:,:3]

def get_tensor_from_transform(RT, Tquad=False, use_so3=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Rotation.from_matrix(R)
    if use_so3:
        so3 = rot.as_rotvec()
        tensor = np.concatenate([so3, T], 0)
    else:
        quad_xyzw = rot.as_quat()
        quad = np.zeros_like(quad_xyzw)
        quad[0] = quad_xyzw[-1]
        quad[1:] = quad_xyzw[:-1]
        if Tquad:
            tensor = np.concatenate([T, quad], 0)
        else:
            tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def get_tensor_from_transform_sim3(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    
    scale = torch.Tensor([np.linalg.det(RT[:3, :3]) ** (1 / 3)])
    RT[:3,:3] = RT[:3,:3]/scale
    tensor = get_tensor_from_transform(RT, Tquad=Tquad)
    tensor = torch.cat([scale, tensor], 0)
     
    return tensor

def get_transform_from_tensor(inputs, use_so3=False):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    if use_so3:
        so3, T = inputs[:, :3], inputs[:, 3:]
        R = so3torotation(so3)
        RT = torch.eye(4, device=inputs.device)[None, ...].repeat(so3.shape[0],1,1)
    else:
        quad, T = inputs[:, :4], inputs[:, 4:]
        R = quad2rotation(quad)
        RT = torch.eye(4, device=inputs.device)[None, ...].repeat(quad.shape[0],1,1)
    RT[:,:3,:] = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT

def get_transform_from_tensor_sim3(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    scale = inputs[:,0]
    quadT = inputs[:,1:]
    RT = get_transform_from_tensor(quadT)
    RT[:,:3,:3] *= scale[:,None,None]
    if N == 1:
        RT = RT[0]
    
    return RT

def so3torotation(so3):
    bs = so3.shape[0]
    rotations_matrix = torch.eye(3).unsqueeze(0).repeat(bs,1,1).to(so3.device)
    angle = torch.norm(so3, dim=-1)
    axis = so3 / angle.unsqueeze(-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    rotation_matrix = torch.stack([
        torch.stack([t*x**2+c, t*x*y-s*z, t*x*z+s*y], dim=1),
        torch.stack([t*x*y+s*z, t*y**2+c, t*y*z-s*x], dim=1),
        torch.stack([t*x*z-s*y, t*y*z+s*x, t*z**2+c], dim=1)
    ], dim=-1)

    rotations_matrix = torch.matmul(rotations_matrix, rotation_matrix)
    
    return rotations_matrix

def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

def importance_sampling_coords(weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, side='right')
    
    inds = torch.max(torch.zeros_like(inds), inds)
    inds = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)

    return inds, u, cdf
    
def plot_reliability(reliability, x, y, z, mesh_dir=None, obj_id=None, center_np=None, r=None):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    fig.update_scenes(camera=dict(up=dict(x=0, y=-1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=-1, z=1)))
    
    if mesh_dir is not None:
        # center mesh
        pred_mesh_path = os.path.join(mesh_dir, f'it_10000_obj{obj_id}.obj')
        pred_mesh = trimesh.load(pred_mesh_path)
        points = pred_mesh.vertices - center_np
        scale = np.abs(points).max()
        points /= scale
        # points = (pred_mesh.vertices - center_np)/r
        gt_pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
        gt_pcd = gt_pcd.voxel_down_sample(0.01)
        points = np.asarray(gt_pcd.points)
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=dict(color=pred_mesh.visual.vertex_colors, size=1)))
    
    reliability = reliability.reshape(100,100)
    fig.add_trace(go.Surface(x=x/scale, y=y/scale, z=z/scale, colorscale='plasma', surfacecolor=reliability,
                             cmin=0, cmax=1, colorbar=dict(len=0.5, x=0.8),
                             showscale=True, opacity=0.75))
    fig.update_layout(title_text="reliability")
    fig.show()
    
def calculate_reliability(metric, eta=0.9, m1=0.1, m2=0.15, M1=0.57, M2=0.65):
    alpha_m = 2*np.log(eta/(1-eta))/(m2-m1)
    beta_m = (m1+m2)/2
    alpha_M = 2*np.log(eta/(1-eta))/(M2-M1)
    beta_M = (M1+M2)/2
    reliability = 1/(1+np.exp(alpha_m*(metric-beta_m))) + 1/(1+np.exp(-alpha_M*(metric-beta_M)))
    return reliability

def geometry_segmentation(rgb, depth, intrinsic_open3d):
    valid_mask = depth>0
    pcd = unproject_pointcloud(depth, intrinsic_open3d, np.eye(4))
    pc = np.asarray(pcd.points)
    depth_map = np.tile(np.zeros_like(depth)[...,None], (1,1,3))
    depth_map[valid_mask] = pc

    normal_image = np.zeros_like(depth_map)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    normals = np.asarray(pcd.normals)
    normals = np.where(normals[:,2:]>0, -normals, normals)
    normal_image[valid_mask] = normals
    H, W = depth.shape
    
    # calculate depth discontinuities
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_depth = cv2.erode(depth, element)
    erosion = depth - eroded_depth
    dilatated_depth = cv2.dilate(depth, element)
    dilatation = dilatated_depth - depth
    ratio = np.zeros_like(depth)
    ratio[valid_mask] = np.maximum(erosion, dilatation)[valid_mask]/depth[valid_mask]
    _, discontinuity_image = cv2.threshold(ratio, 0.01, 1, cv2.THRESH_BINARY)
    
    # calculate convexity map
    min_convexity_map = 10 * np.ones_like(depth)
    for i in range(25):
        if i == 12:
            continue
        kernel = np.zeros((5,5))
        kernel[2,2] = -1
        kernel[i//5,i%5] = 1
        difference_map = cv2.filter2D(depth_map, -1, kernel)
        dot = np.sum(difference_map*(-normal_image), axis=-1)
        _, convexity_mask = cv2.threshold(dot, -0.0005, 1, cv2.THRESH_BINARY)
        _, concavity_mask = cv2.threshold(dot, -0.0005, 1, cv2.THRESH_BINARY_INV)
        
        normal_kernel = np.zeros((5,5))
        normal_kernel[i//5,i%5] = 1
        filtered_normal_image = cv2.filter2D(normal_image, -1, normal_kernel)
        normal_vector_projection = np.sum(normal_image*filtered_normal_image, axis=-1)
        normal_vector_projection = normal_vector_projection * concavity_mask
        convexity_map = convexity_mask + normal_vector_projection
        min_convexity_map = np.minimum(min_convexity_map, convexity_map)
    
    th_convex = 0.9    
    _, convex_map = cv2.threshold(min_convexity_map, th_convex, 1, cv2.THRESH_BINARY) # NOTE: threshold in original code was 0.97
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), anchor=(1,1))
    convex_map = cv2.morphologyEx(convex_map, cv2.MORPH_OPEN, element2)
    convex_map[depth==0] = 0
    
    # compute edge map
    discontinuity_image_closed = cv2.morphologyEx(discontinuity_image, cv2.MORPH_CLOSE, element2)
    edge_map = convex_map - discontinuity_image_closed
    edge_map[edge_map<0] = 0
    edge_map[depth==0] = 0
    edge_map_uint8 = edge_map.astype(np.uint8)    
    
    # compute label map
    contours, hierarchy = cv2.findContours(edge_map_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    labels = np.arange(len(contours))
    
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    small_contours = np.where(areas < 500)[0]
    parent_contours = hierarchy[0, small_contours, 3]
    no_parent_mask = parent_contours == -1
    labels[small_contours[no_parent_mask]] = -1
    no_sibling_mask = np.logical_and(hierarchy[0, small_contours, 0] == -1, hierarchy[0, small_contours, 1] == -1)
    labels[small_contours[(~no_parent_mask) & no_sibling_mask]] = labels[parent_contours[(~no_parent_mask) & no_sibling_mask]]
    labels[small_contours[~(no_sibling_mask | no_parent_mask)]] = -1
    
    output = np.zeros(depth_map.shape, dtype=np.uint8)
    output_labels = np.zeros(depth.shape, dtype=np.int32)  
    for i in range(len(contours)):
        if labels[i] != -1: # NOTE: difference w/ original depth_segmentation
            color = imgviz.label_colormap()[labels[i] % 256]
            cv2.drawContours(output, contours, i, (int(color[0]),int(color[1]),int(color[2])), thickness=2, lineType=cv2.FILLED, hierarchy=hierarchy)
            cv2.drawContours(output_labels, contours, i, int(labels[i]), thickness=2, lineType=cv2.FILLED, hierarchy=hierarchy)

    output[edge_map_uint8 == 0] = np.zeros(3)
    output_labels[edge_map_uint8 == 0] = -1

    min_dists = 0.05 * np.ones_like(depth)
    
    x_range = np.arange(W)
    y_range = np.arange(H)
    y_range, x_range = np.meshgrid(y_range, x_range, indexing='ij')
    valid_edge_points = np.logical_and(edge_map_uint8 == 0, depth > 0)
    for i in range(-4, 5):
        for j in range(-4, 5):
            if i == 0 and j == 0:
                continue
            x_offsets = x_range + i
            y_offsets = y_range + j
            valid_offsets = np.logical_and(x_offsets >= 0, np.logical_and(x_offsets < W, np.logical_and(y_offsets >= 0, y_offsets < H)))
            changed_candidate = valid_edge_points & valid_offsets
            filter_is_edge = valid_edge_points[y_offsets[changed_candidate], x_offsets[changed_candidate]]
            
            filter_points = depth_map[y_offsets[changed_candidate], x_offsets[changed_candidate]]
            edge_points = depth_map[y_range[changed_candidate], x_range[changed_candidate]]
            dists = np.linalg.norm(edge_points - filter_points, axis=-1)
            min_dist_points = min_dists[y_range[changed_candidate], x_range[changed_candidate]].copy()
            valid_dists = dists < min_dist_points
            
            filter_labels = output_labels[y_offsets[changed_candidate], x_offsets[changed_candidate]].copy()
            valid_label = filter_labels >= 0
            
            valid = (~filter_is_edge) & valid_dists & valid_label
            output_labels[y_range[changed_candidate][valid], x_range[changed_candidate][valid]] = filter_labels[valid]
            min_dist_points = np.minimum(min_dist_points, dists)
            min_dists[y_range[changed_candidate][valid], x_range[changed_candidate][valid]] = min_dist_points[valid]
    
    valid_labels = output_labels >= 0
    output[valid_labels] = imgviz.label_colormap()[output_labels[valid_labels] % 256]
    
    labels_unique = np.unique(output_labels)
    n_labels = labels_unique.shape[0]
    segments = []
    segment_masks = []
    for i in range(n_labels):
        label = labels_unique[i]
        if label >= 0:
            label_mask = output_labels == label
            if np.sum(label_mask) < 500: # remove small segments
                output[label_mask] = np.zeros(3, dtype=np.uint8)
                continue
            segment = Segment()
            segment.points = depth_map[label_mask]
            segment.normals = normal_image[label_mask]
            segment.rgbs = rgb[label_mask]
            segments.append(segment)
            segment_masks.append(label_mask)
    
    return normal_image, output, segment_masks, segments

def refine_inst_data(inst_data, segment_masks, threshold=0.7):
    inst_data_refined = np.zeros_like(inst_data)
    obj_ids = list(np.unique(inst_data))
    if 0 in obj_ids:
        obj_ids.remove(0)
    if -1 in obj_ids:
        obj_ids.remove(-1)
    if len(obj_ids) == 0:
        print("this frame has no foreground objects")
        return inst_data_refined
    obj_ids_array = np.array(obj_ids)
    for segment_mask in segment_masks:
        segment_mask = binary_fill_holes(segment_mask) # NOTE: fill holes for each geo seg, not merged
        rates = []
        for obj_id in obj_ids:
            inst_mask = inst_data == obj_id
            intersection = segment_mask & inst_mask
            rate = np.sum(intersection)/np.sum(segment_mask)
            rates.append(rate)
        rates = np.array(rates)
        if np.max(rates) > threshold:
            idx_sel = np.argmax(rates)
            obj_id_sel = obj_ids_array[idx_sel]
            inst_data_refined[segment_mask] = obj_id_sel
     
    return inst_data_refined

class Segment():
    def __init__(self):
        self.points = None
        self.normals = None
        self.rgbs = None