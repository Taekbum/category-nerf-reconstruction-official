import cv2
import imgviz
import numpy as np
import torch
from functorch import combine_state_for_ensemble
import open3d
import math
from scipy.spatial.transform import Rotation
import trimesh
import scipy
import render_rays
import imageio
from itertools import permutations
import matplotlib.pyplot as plt
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
        # inst_pc = unproject_pointcloud(inst_depth.transpose(1,0), intrinsic_open3d, T_CW)
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

def align_to_gravity(box3d_raw, gravity_dir):
    box3d = box3d_raw
    
    Two_raw = np.eye(4)
    Two_raw[:3, 3] = box3d_raw.center
    Two_raw[:3, :3] = box3d_raw.R
    z_o_raw = np.array([0,0,1])
    z_o = (Two_raw[:3, :3].T).dot(-gravity_dir.T)
    theta = math.acos(z_o_raw.dot(z_o))
    # if np.abs(theta) < math.pi/3:
    cross = np.cross(z_o_raw, z_o)
    so3 = theta * cross / np.linalg.norm(cross)
    R = Rotation.from_rotvec(so3).as_matrix()
    Rwo = Two_raw[:3, :3].dot(R)
    
    box3d.R = Rwo
    
    return box3d

def transform_pointcloud(cloud, T_rel):
    n = cloud.shape[0]
    cloud_hom = np.hstack((cloud, np.ones((n,1))))
    cloud_transformed = (T_rel.dot(cloud_hom.T)).T
    
    return cloud_transformed[:,:3]

def trimesh_to_open3d(src):
    dst = open3d.geometry.TriangleMesh()
    dst.vertices = open3d.utility.Vector3dVector(src.vertices)
    dst.triangles = open3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst


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


def load_scene_bound(bbox3D, cfg, use_gt_bound=False):
    if use_gt_bound and hasattr(cfg, 'scene_bound'):
        bound = np.array(cfg.scene_bound)
    else:
        bound = bbox3D#np.stack([bbox3D.center - bbox3D.extent/2, bbox3D.center + bbox3D.extent/2], axis=-1)
    return bound

def world2object(pts, dirs, T, inverse=False):
    if not inverse:
        T = torch.linalg.inv(T)
    if pts is not None:
        pts = (T[:, :3, :3] @ pts[..., None]).squeeze() + T[:, :3, -1]
    if dirs is not None:
        dirs = (T[:, :3, :3] @ dirs[..., None]).squeeze()

    return [pts, dirs]

def intersect_aabb(rays_o, rays_d, bound=None):
    if bound is None:
        bound = torch.from_numpy(np.array([[-1, 1], 
                                           [-1, 1], 
                                           [-1, 1]]).astype(np.float32)).to(rays_o.device)
    
    t = (bound[None, None, None, ...] - rays_o.unsqueeze(-1))/rays_d.unsqueeze(-1)  # (N_obj, H, W, 3, 2)
    near_bb, _ = torch.max(torch.min(t, dim=-1)[0], dim=-1)
    far_bb, _ = torch.min(torch.max(t, dim=-1)[0], dim=-1)
    
    # ## DEBUG ##
    # ax = plt.axes(projection='3d')
    # pts_near = (rays_o + near_bb[...,None] * rays_d)[19, ::100, ::100].reshape((-1,3)).cpu().numpy()
    # pts_far = (rays_o + far_bb[...,None] * rays_d)[19, ::100, ::100].reshape((-1,3)).cpu().numpy()
    # rays_o_ = rays_o[19, ::100, ::100].reshape((-1,3)).cpu().numpy()
    # for i in range(rays_o_.shape[0]):
    #     ax.plot([pts_near[i,0], pts_far[i,0]], [pts_near[i,1], pts_far[i,1]], [pts_near[i,2], pts_far[i,2]], color='r')
    #     ax.plot([rays_o_[i,0], pts_near[i,0]], [rays_o_[i,1], pts_near[i,1]], [rays_o_[i,2], pts_near[i,2]], color='b')
    # # ax.plot([pts_near[:,0], pts_far[:,0]], [pts_near[:,1], pts_far[:,1]], zs=[pts_near[:,2], pts_far[:,2]], color='r')
    # # ax.plot([rays_o[:,0], pts_near[:,0]], [rays_o[:,1], pts_near[:,1]], zs=[rays_o[:,2], pts_near[:,2]], color='b')
    # r = [-1, 1]
    # from itertools import product, combinations
    # for s, e in combinations(np.array(list(product(r, r, r))), 2):
    #     if np.sum(np.abs(s-e)) == r[1]-r[0]:
    #         ax.plot(*zip(s, e), color='g')
    # plt.savefig('debug_near_and_far.png')
    # ## DEBUG ##
    
    intersection_map = torch.stack(torch.where(far_bb > near_bb), dim=-1) # (N_obj, H, W) -> (N_true, 3)
    positive_far = torch.cat(torch.where(far_bb[intersection_map[:,0],intersection_map[:,1],intersection_map[:,2]]>0)) # (n_intersect)
    intersection_map = intersection_map[positive_far] # (n_intersect, 3)
    
    if intersection_map.shape[0] != 0:
        z_ray_in = near_bb[intersection_map[:,0],intersection_map[:,1],intersection_map[:,2]] # (n_intersect)
        z_ray_out = far_bb[intersection_map[:,0],intersection_map[:,1],intersection_map[:,2]]
    else:
        return None, None, None
    
    return z_ray_in, z_ray_out, intersection_map

def box_pts(origins_O, dirs_O, T_objs):
    z_ray_in_o, z_ray_out_o, intersection_map = intersect_aabb(origins_O, dirs_O)
    if z_ray_in_o is not None:
        rays_o_o = origins_O[intersection_map[:,0],intersection_map[:,1],intersection_map[:,2], :] # (n_intersect, 3)
        dirs_o = dirs_O[intersection_map[:,0],intersection_map[:,1],intersection_map[:,2], :]
        Two = T_objs[intersection_map[:,0], ...]
        rays_o, rays_d = world2object(rays_o_o, dirs_o, Two, inverse=True)
        
        pts_box_in_o = rays_o_o + z_ray_in_o[..., None] * dirs_o # (n_intersect, 3) 
        pts_box_in_w, _ = world2object(pts_box_in_o, None, Two, inverse=True)
        z_vals_in_w = ((pts_box_in_w - rays_o)/rays_d)[:, 2] #torch.norm(pts_box_in_w - rays_o, dim=-1) / torch.norm(rays_d, dim=-1)
        
        pts_box_out_o = rays_o_o + z_ray_out_o[..., None] * dirs_o # (n_intersect, 3) 
        pts_box_out_w, _ = world2object(pts_box_out_o, None, Two, inverse=True)
        z_vals_out_w = ((pts_box_out_w - rays_o)/rays_d)[:, 2]#torch.norm(pts_box_out_w - rays_o, dim=-1) / torch.norm(rays_d, dim=-1)
    else:
        pts_box_in_w, rays_o, rays_d, z_vals_in_w, z_vals_out_w, pts_box_in_o, rays_o_o, dirs_o, z_ray_in_o, z_ray_out_o \
            = [], [], [], [], [], [], [], [], [], []
        
    return pts_box_in_w, rays_o, rays_d, z_vals_in_w, z_vals_out_w, \
        pts_box_in_o, rays_o_o, dirs_o, z_ray_in_o, z_ray_out_o, \
        intersection_map
    # return pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, intersection_map


def render_image(c2w, vis_dict, cached_rays_dir, cfg, filename, chunk_size=500000):
    scene_bg = vis_dict[0]
    H, W = cached_rays_dir.shape[:2]
    n_bins = cfg.n_bins_cam2surface + cfg.n_bins
    n_bins_bg = cfg.n_bins_cam2surface_bg + cfg.n_bins
    
    obj_tensors = []
    for cls_id, cls_k in vis_dict.items():
        if cls_id == 0:
            continue
        for obj_id in cls_k.obj_ids:
            obj_tensors.append(cls_k.object_tensor_dict[obj_id])
    obj_tensors = torch.stack(obj_tensors, dim=0)
    T_objs = get_transform_from_tensor_sim3(obj_tensors)
    Toc = torch.linalg.inv(T_objs) @ c2w[None, :, :]  #[N_obj, 4, 4]
    N_obj = len(obj_tensors)
    
    # sample points from background
    rays_o = c2w[None, None, :3, -1] #[1,1,3]
    rays_d = (c2w[None, None, :3, :3] @ cached_rays_dir[..., None].to(cfg.training_device)).squeeze() #[H,W,3]
    
    scene_bound = scene_bg.trainer.bound
    near = 0.01
    far = get_far(scene_bound, rays_o.cpu(), rays_d.cpu())[..., None] # (H, W, 1)
    
    z_vals, perturb = sample_along_ray(near, far, n_bins_bg) #[H,W,n_bins_bg]
    z_vals = z_vals.to(rays_o.device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None] #[H,W,n_bins_bg,3]
    sh_bg = pts.shape
    
    # sample points from objects  
    dirs_O = (Toc[:, None, None, :3, :3] @ cached_rays_dir[None, :, :, :, None].to(cfg.training_device)).squeeze() #[N_obj,H,W,3]
    origins_O = Toc[:, None, None, :3, -1].repeat(1, H, W, 1) #[N_obj, H, W, 3]
    
    pts_box_w, origins_box_w, viewdirs_box_w, z_vals_in_w, z_vals_out_w,\
    pts_box_o, origins_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o, \
        intersection_map = box_pts(origins_O, dirs_O, T_objs)
    Two = T_objs[intersection_map[:,0], ...]
    
    if z_vals_in_o is None or len(z_vals_in_o) == 0:
        z_vals_obj = torch.zeros([1])
        intersection_map = torch.from_numpy(np.zeros([1, 3]))
    else:
        n_intersect = z_vals_in_o.shape[0]
        z_vals_box_o = torch.linspace(0., 1., n_bins)[None, :].repeat(n_intersect, 1).to(z_vals_in_o.device) * \
                                   (z_vals_out_o - z_vals_in_o)[:, None]
        pts_box_samples_o = pts_box_o[:, None, :] + viewdirs_box_o[:, None, :] * z_vals_box_o[..., None]
        # pts_box_samples_w, _ = world2object(pts_box_samples_o.reshape((-1, 3)), None, \
        #     Two.repeat(pts_box_samples_o.shape[1],1,1), inverse=True)
        # pts_box_samples_w = pts_box_samples_w.reshape((n_intersect, n_bins, 3)) #[n_intersect, n_bins, 3]
        
        # z_vals_obj_w = ((pts_box_samples_w - origins_box_w[:, None, :])/viewdirs_box_w[:, None, :])[...,2]#torch.norm(pts_box_samples_w - origins_box_w[:, None, :], dim=-1) #[n_intersect, n_bins]
        z_vals_obj = ((pts_box_samples_o - origins_box_o[:, None, :])/viewdirs_box_o[:, None, :])[...,2]
        
    z_vals, id_z_vals_bg, id_z_vals_obj = combine_z(z_vals,
                                                    z_vals_obj if z_vals_in_o is not None else None,
                                                    intersection_map,
                                                    H,
                                                    W,
                                                    n_bins_bg,
                                                    N_obj,
                                                    n_bins)
        # (H, W, n_bins_bg+N_obj*n_bins), (H, W, n_bins_bg), (H, W, N_obj*n_bins)
    raw = torch.zeros((H, W, n_bins_bg+N_obj*n_bins, 4))
    
    pts_flat = pts.reshape((-1, 3))
    n_chunks = int(np.ceil(pts_flat.shape[0] / chunk_size))
    raw_bg = []
    for i in range(n_chunks): # 2s/it 1000000 pts
        chunk_idx = slice(i * chunk_size, (i + 1) * chunk_size)
        bg_embedding_i = scene_bg.trainer.pe(pts_flat[chunk_idx, ...])
        bg_alpha_i, bg_color_i = scene_bg.trainer.fc_occ_map(bg_embedding_i)
        bg_occupancy_i = render_rays.occupancy_activation(bg_alpha_i)
        raw_bg_i = torch.cat([bg_occupancy_i, bg_color_i], dim=-1)
        raw_bg.append(raw_bg_i)
    raw_bg = torch.cat(raw_bg)
    raw_bg = raw_bg.view(list(sh_bg[:-1])+list(raw_bg.shape[-1:])).cpu()
    raw.scatter_(2, id_z_vals_bg[..., None].repeat(1,1,1,4), raw_bg)
    
    if z_vals_in_o is not None and len(z_vals_in_o) != 0:
        obj_idx = 0
        intersection_map_ = []
        raw_ = []
        for cls_id, cls_k in vis_dict.items():
            if cls_id == 0:
                continue
            shape_codes = cls_k.trainer.shape_codes
            texture_codes = cls_k.trainer.texture_codes
            pts_k = []
            batch_indices = []
            intersection_map_cls = []
            for obj_id in cls_k.obj_ids:
                pts_obj = pts_box_samples_o[intersection_map[:, 0] == obj_idx] #[n_intersect_(obj_id), n_bins, 3]
                pts_obj = pts_obj.view((-1, 3)).cpu()
                if pts_obj.shape[0] > 0:
                    intersection_map_obj = intersection_map[intersection_map[:, 0] == obj_idx].repeat(n_bins, 1).cpu()
                    
                    index = cls_k.trainer.inst_id_to_index[obj_id]
                    indices = torch.from_numpy(np.array([index])).repeat(pts_obj.shape[0])
                    batch_indices.append(indices)
                    pts_k.append(pts_obj)
                    intersection_map_cls.append(intersection_map_obj)
                    
                obj_idx += 1
            
            if len(pts_k) > 0:
                batch_indices = torch.cat(batch_indices).to(cfg.training_device) #[n_intersect_(cls_id)]
                pts_k = torch.cat(pts_k) #[n_intersect_(cls_id), 3]
                shape_code_k = shape_codes(batch_indices).cpu()
                texture_code_k = texture_codes(batch_indices).cpu()
                intersection_map_cls = torch.cat(intersection_map_cls) #[n_intersect_(cls_id), 3]
                
                raw_k = []
                n_chunks = int(np.ceil(pts_k.shape[0] / chunk_size))
                for i in range(n_chunks):
                    chunk_idx = slice(i * chunk_size, (i + 1) * chunk_size)
                    embedding_k_i = cls_k.trainer.pe(pts_k[chunk_idx, ...].to(cfg.training_device))
                    alpha_k_i, color_k_i = cls_k.trainer.fc_occ_map(embedding_k_i, \
                        shape_code_k[chunk_idx, ...].to(cfg.training_device), texture_code_k[chunk_idx, ...].to(cfg.training_device))
                    occupancy_k_i = render_rays.occupancy_activation(alpha_k_i)
                    raw_k_i = torch.cat([occupancy_k_i, color_k_i], dim=-1)
                    raw_k.append(raw_k_i)
                raw_k = torch.cat(raw_k)
                
                intersection_map_.append(intersection_map_cls)
                raw_.append(raw_k)
        
        raw_ = torch.cat(raw_).cpu() #[n_intersect, 4]
        intersection_map_ = torch.cat(intersection_map_) #[n_intersect, 3]
        
        raw_sparse = torch.zeros((H, W, N_obj*n_bins, 4))
        raw_sparse[intersection_map_[:, 1], intersection_map_[:, 2], intersection_map_[:, 0], :] = raw_
        raw.scatter_(2, id_z_vals_obj[..., None].repeat(1,1,1,4), raw_sparse)
        # raw_k_sparse = torch.zeros((H, W, N_obj*n_bins, 4))
        # raw_k_sparse[intersection_map_cls[:, 1], intersection_map_cls[:, 2], intersection_map_cls[:, 0], :] = raw_k
        # raw.scatter_(2, id_z_vals_obj[..., None].repeat(1,1,1,4), raw_k_sparse)   
        
    occupancy, color = raw[..., 0], raw[...,1:] # alpha, color = raw[..., 0], raw[...,1:]
    # occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)
    render_color = render_rays.render(termination[..., None], color, dim=-2) # (H, W, 3)
    
    rgb8 = to8b(render_color.numpy()).transpose(1,0,2)
    imageio.imwrite(filename, rgb8)
    
def sample_along_ray(near, far, N_samples, sampling_method='linear', perturb=1.):
    # Sample along each ray given one of the sampling methods. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0., 1., N_samples)[None, None, :]
    if sampling_method == 'squareddist':
        z_vals = near * (1. - np.square(t_vals)) + far * (np.square(t_vals))
    elif sampling_method == 'lindisp':
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    else:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
        if sampling_method == 'discrete':
            perturb = 0

    # Perturb sampling time along each ray. (vanilla NeRF option)
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    return z_vals, perturb

def get_far(scene_bound, rays_o, rays_d):
    x_scene, y_scene, z_scene = scene_bound.extent/2
    corner_points = np.array([[x_scene,-x_scene,x_scene,-x_scene,x_scene,-x_scene,x_scene,-x_scene],
                            [y_scene,y_scene,-y_scene,-y_scene,y_scene,y_scene,-y_scene,-y_scene],
                            [z_scene,z_scene,z_scene,z_scene,-z_scene,-z_scene,-z_scene,-z_scene]])
    corner_points_w = scene_bound.R @ corner_points + scene_bound.center[:, None]
    corner_points_w = torch.from_numpy(corner_points_w.astype(np.float32))
    min_points, _ = torch.min(corner_points_w, dim=-1)
    max_points, _ = torch.max(corner_points_w, dim=-1)
    bound = torch.stack([min_points, max_points], dim=-1)
    
    t = (bound[None, None, ...] - rays_o.unsqueeze(-1))/rays_d.unsqueeze(-1)  # (H, W, 3, 2)
    far, _ = torch.min(torch.max(t, dim=-1)[0], dim=-1) # (H, W)
    far += 0.01
    
    return far

def combine_z(z_vals_bg, z_vals_obj, intersection_map, H, W, N_samples_bg, N_obj, N_samples_obj):
    if z_vals_obj is None:
        z_vals_obj_sparse = torch.zeros([H, W, N_obj * N_samples_obj])
    else:
        z_vals_obj_sparse = torch.zeros([H, W, N_obj, N_samples_obj]).to(z_vals_obj.device)
        z_vals_obj_sparse[intersection_map[:,1], intersection_map[:,2], intersection_map[:,0], :] = \
            z_vals_obj
        z_vals_obj_sparse = z_vals_obj_sparse.view((H, W, N_obj * N_samples_obj))
    
    if z_vals_bg is None:
        z_vals, _ = torch.sort(z_vals_obj_sparse, dim=-1)
        id_z_vals_bg = None
    else:
        z_vals, _ = torch.sort(torch.cat([z_vals_obj_sparse, z_vals_bg], dim=-1), dim=-1) # (H, W, N_samples_bg+N_obj*N_samples_obj)
        id_z_vals_bg = torch.searchsorted(z_vals, z_vals_bg) # (H, W, N_samples_bg)
    
    id_z_vals_obj = torch.searchsorted(z_vals, z_vals_obj_sparse)#.view((H, W, N_obj*N_samples_obj))
    
    return z_vals, id_z_vals_bg.cpu(), id_z_vals_obj.cpu()

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

def plot_uncertainty_field(metrics_plot, viewing_points, points_all, selected_inds, selected_view=False, obj_id=None, mesh_dir=None, transform_np=None):
    import matplotlib.cm as cm
    import matplotlib.pylab as plab
    from matplotlib.colors import ListedColormap
    
    cmap = plab.cm.plasma
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = 0.75#np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    if mesh_dir is not None:
        # center mesh
        pred_mesh_path = os.path.join(mesh_dir, f'it_10000_obj{obj_id}.obj')
        pred_mesh = trimesh.load(pred_mesh_path)
        points = transform_pointcloud(pred_mesh.vertices, transform_np)
        gt_pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
        gt_pcd = gt_pcd.voxel_down_sample(0.01)
        points = np.asarray(gt_pcd.points)
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        ax.scatter(xs, ys, zs, c='black')
    
    # surface entropies
    xyz = points_all.numpy()
    x_plot = xyz[...,0]
    y_plot = xyz[...,1]
    z_plot = xyz[...,2]
    fmin, fmax = metrics_plot.max(), metrics_plot.min()
    metrics_plot = (metrics_plot - fmin) / (fmax - fmin)
    if not selected_view:
        metrics_plot[selected_inds] = 0
        metrics_plot = metrics_plot.reshape(100,100)
    ax.plot_surface(x_plot, y_plot, z_plot, rstride=1, cstride=1,
        facecolors=my_cmap(metrics_plot), linewidth=0)
    
    m = cm.ScalarMappable(cmap=my_cmap)
    m.set_array(metrics_plot)
    plt.colorbar(m)
    
    # viewing points
    viewing_points = viewing_points/np.linalg.norm(viewing_points, axis=-1, keepdims=True)
    xs_view, ys_view, zs_view = viewing_points[:,0], viewing_points[:,1], viewing_points[:,2]
    ax.scatter(xs_view, ys_view, zs_view, c='green')
    
    # selected points
    if selected_view:
        inds_unique = selected_inds.unique()
        for ind in inds_unique:
            x_sel = viewing_points[ind,0]
            y_sel = viewing_points[ind,1]
            z_sel = viewing_points[ind,2]
            count = (selected_inds==ind).count_nonzero().item()
            ax.scatter(x_sel, y_sel, z_sel, c='red')
            ax.text(x_sel, y_sel, z_sel, str(count), color='red')
    # else:
    #     xyz_sel = xyz.reshape(-1,3)[selected_inds]
    #     x_sel = xyz_sel[:,0]
    #     y_sel = xyz_sel[:,1]
    #     z_sel = xyz_sel[:,2]
    #     ax.scatter(x_sel, y_sel, z_sel, c='red')
    
    # To visualize well
    poles = np.array([[0,0,1], [0,0,-1]])
    xs_pole, ys_pole, zs_pole = poles[:,0], poles[:,1], poles[:,2]
    ax.scatter(xs_pole, ys_pole, zs_pole, c='black')
    xs_eq = x_plot[z_plot==0]
    ys_eq = y_plot[z_plot==0]
    zs_eq = np.zeros_like(xs_eq)
    ax.plot(xs_eq, ys_eq, zs_eq, '-k')
    
    # Turn off the axis planes
    ax.view_init()
    ax.azim = 0
    ax.elev = 0
    ax.set_axis_off()
    plt.show()
    # save_file = os.path.join(save_dir_cls, f'{obj_id}.png')
    # plt.savefig(save_file)

def plot_selected(metrics_plot, viewing_points, points_all, selected_inds, selected_view=False, obj_id=None, mesh_dir=None, transform_np=None):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    
    if mesh_dir is not None:
        # center mesh
        pred_mesh_path = os.path.join(mesh_dir, f'it_10000_obj{obj_id}.obj')
        pred_mesh = trimesh.load(pred_mesh_path)
        points = transform_pointcloud(pred_mesh.vertices, transform_np)
        gt_pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
        gt_pcd = gt_pcd.voxel_down_sample(0.01)
        points = np.asarray(gt_pcd.points)
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs))
    
    # surface entropies
    xyz = points_all.numpy()
    x_plot = xyz[...,0]
    y_plot = xyz[...,1]
    z_plot = xyz[...,2]
    if not selected_view:
        # metrics_plot[selected_inds] = 0
        metrics_plot = metrics_plot.reshape(100,100)
    fig.add_trace(go.Surface(x=x_plot, y=y_plot, z=z_plot, colorscale='plasma', surfacecolor=metrics_plot,
                             cmin=metrics_plot.min(), cmax=metrics_plot.max(), colorbar=dict(len=0.75, x=0.85),
                             showscale=True, opacity=0.75))
    
    # viewing points
    viewing_points = viewing_points/np.linalg.norm(viewing_points, axis=-1, keepdims=True)
    xs_view, ys_view, zs_view = viewing_points[:,0], viewing_points[:,1], viewing_points[:,2]
    fig.add_trace(go.Scatter3d(x=xs_view, y=ys_view, z=zs_view, mode='markers', 
                               marker=dict(size=1, color='green')))
    
    # selected points
    if selected_view:
        inds_unique = selected_inds.unique()
        x_sel, y_sel, z_sel, count = [], [], [] ,[]
        for ind in inds_unique:
            x_sel.append(viewing_points[ind,0])
            y_sel.append(viewing_points[ind,1])
            z_sel.append(viewing_points[ind,2])
            count.append((selected_inds==ind).count_nonzero().item())
        fig.add_trace(go.Scatter3d(x=x_sel, y=y_sel, z=z_sel, mode='markers+text', marker=dict(size=1, color='red'), 
                                    text=count, textposition='top center'))
            
    fig.update_layout(title_text="selected w/ reliability")
    fig.show()
    
def plot_reliability(reliability, x, y, z, mesh_dir=None, obj_id=None, center_np=None, r=None):
    '''
    reliability plot using plotly
    '''
    
    # fig_ = plt.figure()
    # ax = fig_.add_subplot(1,1,1)
    # colors = ['green', 'red', 'blue']
    # reliability_types = [reliability[reliability<beta_m], reliability[np.bitwise_and(reliability>beta_m,reliability<beta_M)], reliability[reliability>beta_M]]
    # for idx in range(len(colors)):
    #     reliability_type = reliability_types[idx]
    #     y = idx * np.ones_like(reliability_type)
    #     # save term_prob plots
    #     ax.scatter(reliability_type, y, c=colors[idx])
    # plt.show()
    # # plt.close()
    
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

def geometry_segmentation(rgb, depth, intrinsic_open3d, debug_dir=None, frame=0):
    # t1 = time.time()
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
    
    valid_mask = depth>0
    pcd = unproject_pointcloud(depth, intrinsic_open3d, np.eye(4))
    pc = np.asarray(pcd.points)
    depth_map = np.tile(np.zeros_like(depth)[...,None], (1,1,3))
    depth_map[valid_mask] = pc
    
    # calculate surface normal
    # normal_image = compute_normals(depth_map)
    # depth_map = depth_map[6:-6, 6:-6]
    # depth = depth[6:-6, 6:-6]
    normal_image = np.zeros_like(depth_map)
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    normals = np.asarray(pcd.normals)
    normals = np.where(normals[:,2:]>0, -normals, normals)
    normal_image[valid_mask] = normals
    H, W = depth.shape
    if debug_dir is not None:
        normal_vis = (255*(normal_image+1)/2).astype(np.uint8)
        normal_dir = os.path.join(debug_dir, "normal")
        os.makedirs(normal_dir, exist_ok=True)
        normal_file = os.path.join(normal_dir, "%03d.png" % frame)
        cv2.imwrite(normal_file, normal_vis)
    
    # calculate depth discontinuities
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_depth = cv2.erode(depth, element)
    erosion = depth - eroded_depth
    dilatated_depth = cv2.dilate(depth, element)
    dilatation = dilatated_depth - depth
    ratio = np.zeros_like(depth)
    ratio[valid_mask] = np.maximum(erosion, dilatation)[valid_mask]/depth[valid_mask]
    _, discontinuity_image = cv2.threshold(ratio, 0.01, 1, cv2.THRESH_BINARY)
    if debug_dir is not None:
        discontinuity_dir = os.path.join(debug_dir, "discontinuity")
        os.makedirs(discontinuity_dir, exist_ok=True)
        discontinuity_file = os.path.join(discontinuity_dir, "%03d.png" % frame)
        cv2.imwrite(discontinuity_file, (255*discontinuity_image).astype(np.uint8))
    
    # # calculate maximum distance map
    # theta = np.pi/6
    # maximum_distance_image = np.zeros_like(depth)
    # sigma_axial_noise = np.zeros_like(depth)
    # sigma_axial_noise[valid_mask] = 0.0012 + 0.0019*(depth[valid_mask]-0.02)**2 + 0.0001/np.sqrt(depth[valid_mask])*theta**2/(np.pi/2-theta)**2
    
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
    if debug_dir is not None:
        convexity_dir = os.path.join(debug_dir, f"convexity_{th_convex}")
        os.makedirs(convexity_dir, exist_ok=True)
        convexity_file = os.path.join(convexity_dir, "%03d.png" % frame)
        cv2.imwrite(convexity_file, (255*convex_map).astype(np.uint8))
    
    # compute edge map
    discontinuity_image_closed = cv2.morphologyEx(discontinuity_image, cv2.MORPH_CLOSE, element2)
    # distance_map_closed = cv2.morphologyEx(distance_map, cv2.MORPH_CLOSE, element2)
    # _, distance_discontinuity_map = cv2.threshold(distance_map_closed + discontinuity_image, 1, 1, cv2.THRESH_TRUNC)
    edge_map = convex_map - discontinuity_image_closed
    edge_map[edge_map<0] = 0
    edge_map[depth==0] = 0
    edge_map_uint8 = edge_map.astype(np.uint8)
    if debug_dir is not None:
        edge_dir = os.path.join(debug_dir, "edge")
        os.makedirs(edge_dir, exist_ok=True)
        edge_file = os.path.join(edge_dir, "edge_%03d.png" % frame)
        cv2.imwrite(edge_file, 255*edge_map_uint8)
    
    
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
    
    # contours_small = tuple(np.array(contours, dtype=object)[small_contours[~(no_sibling_mask) | no_parent_mask]])
    # hierarchy_small = hierarchy[:,small_contours[~(no_sibling_mask) | no_parent_mask],:]
    # cv2.drawContours(edge_map_uint8, contours_small, -1, 0, thickness=2, lineType=cv2.FILLED, hierarchy=hierarchy_small)
    
    output = np.zeros(depth_map.shape, dtype=np.uint8)
    output_labels = np.zeros(depth.shape, dtype=np.int32)  
    for i in range(len(contours)):
        if labels[i] != -1: # NOTE: difference w/ original depth_segmentation
            color = imgviz.label_colormap()[labels[i] % 256]
            cv2.drawContours(output, contours, i, (int(color[0]),int(color[1]),int(color[2])), thickness=2, lineType=cv2.FILLED, hierarchy=hierarchy)
            cv2.drawContours(output_labels, contours, i, int(labels[i]), thickness=2, lineType=cv2.FILLED, hierarchy=hierarchy)
            # cv2.drawContours(edge_map_uint8, contours, i, 0, thickness=2, hierarchy=hierarchy, maxLevel=1)

    output[edge_map_uint8 == 0] = np.zeros(3)
    output_labels[edge_map_uint8 == 0] = -1
    if debug_dir is not None:
        edge_contour_dir = os.path.join(debug_dir, "edge_contour")
        os.makedirs(edge_contour_dir, exist_ok=True)
        edge_contour_file = os.path.join(edge_contour_dir, "edge_contour_%03d.png" % frame)
        cv2.imwrite(edge_contour_file, 255*edge_map_uint8)
        
        raw_output_dir = os.path.join(debug_dir, "raw_output")
        os.makedirs(raw_output_dir, exist_ok=True)
        raw_output_file = os.path.join(raw_output_dir, "raw_output_%03d.png" % frame)
        cv2.imwrite(raw_output_file, output)

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
    if debug_dir is not None:
        output_dir = os.path.join(debug_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "%03d.png" % frame)
        cv2.imwrite(output_file, output) 
    
    labels_unique = np.unique(output_labels)
    n_labels = labels_unique.shape[0]
    segments = []#{}
    segment_masks = []#{}
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
            # segments[label] = Segment()
            # segments[label].points.append(depth_map[label_mask])
            # segments[label].normals.append(normal_image[label_mask])
            # segments[label].rgbs.append(rgb[label_mask])
            # segment_masks[label] = 255*(label_mask.astype(np.uint8))

    # t2 = time.time()
    # print(f"geometry_segmentation takes {t2-t1} seconds")
    
    return normal_image, output, segment_masks, segments

def refine_inst_data(inst_data, segment_masks, segments, threshold=0.7, debug_dir=None, frame=0): # NOTE: threshold=0.1 from voxblox++
    # t1 = time.time()
    inst_data_refined = np.zeros_like(inst_data)
    if debug_dir is not None:
        output = np.tile(np.zeros_like(inst_data)[...,None], (1,1,3))
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
            if debug_dir is not None:
                output[segment_mask] = imgviz.label_colormap()[obj_id_sel % 256]
    
    if debug_dir is not None:
        output_folder = os.path.join(debug_dir, f"output_refined_{threshold}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, "%03d.png" % frame)
        cv2.imwrite(output_file, output) 
    # t2 = time.time()
    # print(f"geometry_segmentation takes {t2-t1} seconds")        
    return inst_data_refined

class Segment():
    def __init__(self):
        self.points = None
        self.normals = None
        self.rgbs = None