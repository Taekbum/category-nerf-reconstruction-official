'''
https://github.com/MIT-SPARK/pose-baselines/blob/main/fpfh_teaser/teaser_fpfh_icp.py
'''

import numpy as np
import torch
import copy

from .helpers import *
import open3d as o3d

def teaser_fpfh_icp(source_points, target_points, voxel_size=0.05, spc=False, visualize=False):
    """
    source_points   : torch.tensor of shape (3, n)
    target_points   : torch.tensor of shape (3, m)

    returns :
    rotation    : torch.tensor of shape (3, 3)
    translation : torch.tensor of shape (3, 1)

    """

    # torch to o3d point clouds
    src_ = pos_tensor_to_o3d(source_points, estimate_normals=False)
    tar_ = pos_tensor_to_o3d(target_points, estimate_normals=False)

    # point cloud downsampling
    src_down = src_.voxel_down_sample(voxel_size=voxel_size)
    tar_down = tar_.voxel_down_sample(voxel_size=voxel_size)

    # o3d point cloud to numpy array
    src_np = pcd2xyz(src_down)  # np array of size 3 by N
    tar_np = pcd2xyz(tar_down)  # np array of size 3 by M
    if spc:
        src_corr = np.tile(src_np, (1, tar_np.shape[1]))
        tar_corr = np.repeat(tar_np, src_np.shape[1], axis=1)
    else:
        # extract FPFH features
        src_feats = extract_fpfh(src_down, voxel_size)
        tar_feats = extract_fpfh(tar_down, voxel_size)

        # establish correspondences by nearest neighbour search in feature space
        corrs_src, corrs_tar = find_correspondences(
            src_feats, tar_feats, mutual_filter=True)
        src_corr = src_np[:, corrs_src]  # np array of size 3 by num_corrs
        tar_corr = tar_np[:, corrs_tar]  # np array of size 3 by num_corrs

    if visualize:
        num_corrs = src_corr.shape[1]
        print(f'FPFH generates {num_corrs} putative correspondences.')

    # visualize the point clouds together with feature correspondences
    if visualize:
        points = np.concatenate((src_corr.T, tar_corr.T), axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i, i + num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        src_down.paint_uniform_color([0.0, 0.0, 1.0])  # show src_down in blue
        tar_down.paint_uniform_color([1.0, 0.0, 0.0])  # show tar_down in red
        o3d.visualization.draw_geometries([src_down, tar_down, line_set])

    # robust global registration using TEASER++
    if spc:
        noise_bound = 0.01#0.1 * voxel_size
        if src_corr.shape[1] > 10000: # 30000 has better performance but much slow
            indices = np.random.choice(src_corr.shape[1], size=10000, replace=False)
            src_corr = src_corr[:, indices]
            tar_corr = tar_corr[:, indices]
    else:
        noise_bound = voxel_size
    teaser_solver = get_teaser_solver(noise_bound)
    teaser_solver.solve(src_corr, tar_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    # local refinement using ICP
    icp_sol = o3d.pipelines.registration.registration_icp(
        src_down, tar_down, noise_bound, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation
    # T_icp = T_teaser

    # visualize the registration after ICP refinement
    if visualize:
        src_down_T_icp = copy.deepcopy(src_down).transform(T_icp)
        o3d.visualization.draw_geometries([src_down_T_icp, tar_down])

    T_ = copy.deepcopy(T_icp)
    T = torch.from_numpy(T_)

    return T[:3, :3], T[:3, 3:4]


def pos_tensor_to_o3d(pos, estimate_normals=True):
    """
    inputs:
    pos: torch.tensor of shape (3, N)

    output:
    open3d PointCloud
    """
    # breakpoint()
    pos_o3d = o3d.utility.Vector3dVector(pos.transpose(0, 1).to('cpu').numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    if estimate_normals:
        object.estimate_normals()

    return object


class TEASER_FPFH_ICP():
    """
    This code implements batch TEASER++ (with correspondences using FPFH) + ICP
    """
    def __init__(self, source_points, voxel_size=0.05, spc=False, visualize=False):
        """
        source_points   : torch.tensor of shape (1, 3, m)
        
        """

        self.source_points = source_points
        self.viz = visualize
        self.device_ = source_points.device
        self.voxel_size = voxel_size
        self.spc = spc

    def forward(self, target_points):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]
        src = self.source_points.squeeze(0)
        viz = self.viz

        R = torch.zeros(batch_size, 3, 3).to(device=self.device_)
        t = torch.zeros(batch_size, 3, 1).to(device=self.device_)

        for b in range(batch_size):
            tar = target_points[b, ...]

            # pruning the tar of all zero points
            idx = torch.any(tar == 0, 0)
            tar_new = tar[:, torch.logical_not(idx)]

            # teaser + fpfh + icp
            R_batch, t_batch = teaser_fpfh_icp(source_points=src,
                                               target_points=tar_new,
                                               spc=self.spc,
                                               visualize=viz,
                                               voxel_size=self.voxel_size)
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


