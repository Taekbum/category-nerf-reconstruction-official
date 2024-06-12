import imgviz
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from utils import enlarge_bbox, get_bbox2d, get_bbox2d_batch, accumulate_pointcloud, accumulate_pointcloud_tsdf, get_pose_from_pointcloud, trimesh_to_open3d, transform_pointcloud, BoundingBox, get_possible_transform_from_bbox, get_obb, get_bound, calculate_reliability, plot_reliability, box_filter, geometry_segmentation, refine_inst_data, unproject_pointcloud
import glob
from torchvision import transforms
import image_transforms
import open3d
import time
import copy
import trimesh
import argparse
from cfg import Config
import sys
from scipy.spatial.transform import Rotation
import model
import embedding
import render_rays
import liegroups
from scene_cateogries import cameraInfo, origin_dirs_O, stratified_bins, normal_bins_sampling
import matplotlib.pyplot as plt

import torch.nn.functional as F

def get_dataset(cfg, debug_dir=None, train_mode=True):
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg, train_mode=train_mode)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg, debug_dir=debug_dir)
    elif cfg.dataset_format == "toy":
        dataset = ToyDataset(cfg)    
    else:
        print("Dataset format {} not found".format(cfg.dataset_format))
        exit(-1)
    return dataset
   
class BaseDataSet(Dataset):
    def __init__(self):
        pass
    
    def get_all_poses(self, align_poses=True):
        print('get_all_poses')
        t1 = time.time()
        cls_id_list = [54]
        for cls_id in self.inst_dict.keys():
            # if not cls_id in cls_id_list:
            #     continue
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if cls_id == 0:
                background_list = inst_dict_cls['frame_info']
                if self.name == "replica":
                    background_pcs = accumulate_pointcloud(0, background_list, self.sample_dict, self.intrinsic_open3d)
                else:
                    background_pcs = accumulate_pointcloud_tsdf(0, background_list, self.sample_dict, self.intrinsic_open3d, depth_scale=self.depth_scale, max_depth=self.max_depth)
                transform, extents = trimesh.bounds.oriented_bounds(np.asarray(background_pcs.points))  # pc
                transform = np.linalg.inv(transform)
                bbox3D = BoundingBox()
                bbox3D.center = transform[:3, 3]
                bbox3D.R = transform[:3, :3]
                bbox3D.extent = extents
                inst_dict_cls['bbox3D'] = bbox3D
                inst_dict_cls['pcs'] = background_pcs
                continue
            for idx in range(n_obj):
                inst_id = obj_ids[idx]
                inst_list = inst_dict_cls[inst_id]['frame_info']
                if self.name == "replica":
                    inst_pcs = accumulate_pointcloud(inst_id, inst_list, self.sample_dict, self.intrinsic_open3d)
                else:
                    if 'pcs' not in inst_dict_cls[inst_id].keys():
                        print(f"{inst_id} is not detected from semantically refined geometry segmentations")
                        inst_pcs = None
                    else:
                        inst_pcs = inst_dict_cls[inst_id]['pcs']
                        inst_pcs = inst_pcs.voxel_down_sample(0.01)
                if inst_pcs is None:
                    inst_dict_cls[inst_id]['T_obj'] = np.eye(4)
                else:
                    inst_dict_cls[inst_id]['pcs'] = inst_pcs
                    if not align_poses:
                        T_obj, bbox3D = get_pose_from_pointcloud(inst_pcs)
                        inst_dict_cls[inst_id]['T_obj'] = T_obj
                        if bbox3D is not None:
                            inst_dict_cls[inst_id]['bbox3D'] = bbox3D
                    
        t2 = time.time()
        print('get_all_poses takes {} seconds'.format(t2-t1))
    
    # def visualize_representative(self, cls_id, obj_ids):
    #     scene_mesh = trimesh.load(self.scene_mesh_file)
    #     scene_mesh_o3d = trimesh_to_open3d(scene_mesh)
    #     geometries_vis = [scene_mesh_o3d]

    #     counts = np.array([self.count_dict[cls_id][obj_id] for obj_id in obj_ids])
    #     idx_representative = np.argmax(counts) #0 # option 2 - argmax / option 1 - argmin
    #     for idx in range(len(obj_ids)):
    #         obj_id = obj_ids[idx]
    #         inst_pc = self.inst_dict[cls_id][obj_id]['pcs']
    #         if idx == idx_representative:
    #             inst_pc.paint_uniform_color([0.0, 0.0, 1.0])
    #         else:
    #             inst_pc.paint_uniform_color([1.0, 0.0, 0.0])
    #         geometries_vis.append(inst_pc)
    #     open3d.visualization.draw_geometries(geometries_vis)
    
    def visualize_representative(self, cls_id, obj_ids):
        geometries_vis = []

        counts = np.array([self.count_dict[cls_id][obj_id] for obj_id in obj_ids])
        max_count = counts.max()
        max_indices = np.where(counts==max_count)[0]
        for idx in range(len(obj_ids)):
            obj_id = obj_ids[idx]
            inst_pc = self.inst_dict[cls_id][obj_id]['pcs']
            geometries_vis.append(inst_pc)
            if idx in max_indices:
                open3d.visualization.draw_geometries([inst_pc])
    
    def get_uncertainty_fields(self, cfg, load_pretrained=False, view_uncertainty=False, suppress_unseen=False, use_reliability=True, iteration=None, use_gt_bound=False):
        
        emb_size1 = 21*(3+1)+3
        emb_size2 = 21*(5+1)+3 - emb_size1
        
        if not view_uncertainty:
            if not hasattr(self, 'count_dict'):
                self.count_dict = {}
        if not hasattr(self, 'bbox3d_dict'):
            self.bbox3d_dict = {}
        cls_id_list = [54] # 47: lamp, 20: chair, 15: box, 91: vase. 64: plate, 70: sculpture, 14: bottle, 44: indoor-plant
        if load_pretrained:
            if not hasattr(self, 'fc_occ_map_dict'):
                self.fc_occ_map_dict = {}
            if not hasattr(self, 'pe_dict'):
                self.pe_dict = {}
            for cls_id in self.inst_dict.keys():
                # if not cls_id in cls_id_list:
                #     continue
                if cls_id == 0:
                    continue
                inst_dict_cls = self.inst_dict[cls_id]
                obj_ids = list(inst_dict_cls.keys())
                # if len(obj_ids) == 1:
                #     self.count_dict[cls_id] = {}
                #     self.bbox3d_dict[cls_id] = {}
                #     self.bbox3d_dict[cls_id][obj_ids[0]] = [] # dummy
                #     continue
                # else:
                if cls_id not in self.fc_occ_map_dict.keys():
                    self.fc_occ_map_dict[cls_id] = {}
                if cls_id not in self.pe_dict.keys():
                    self.pe_dict[cls_id] = {}
                if cls_id not in self.bbox3d_dict.keys():
                    self.bbox3d_dict[cls_id] = {}
                for obj_id in obj_ids:
                    # if self.name == 'toy':
                    #     ckpt_dir = os.path.join(cfg.weight_root, "ckpt", str(cls_id))
                    # else:
                    ckpt_dir = os.path.join(cfg.weight_root, "ckpt", str(obj_id))
                    if not os.path.isdir(ckpt_dir):
                        continue
                    ckpt_paths = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))]
                    if iteration is None:
                        ckpt_path = ckpt_paths[-1]
                    else:
                        ckpt_path = os.path.join(ckpt_dir, f'obj_{obj_id}_it_{iteration}.pth')
                    ckpt = torch.load(ckpt_path, map_location = torch.device('cpu'))
                    
                    self.fc_occ_map_dict[cls_id][obj_id] = model.OccupancyMap(
                        emb_size1,
                        emb_size2,
                        hidden_size=cfg.hidden_feature_size
                    )
                    self.fc_occ_map_dict[cls_id][obj_id].apply(model.init_weights).to(cfg.data_device)
                    self.pe_dict[cls_id][obj_id] = embedding.UniDirsEmbed(max_deg=cfg.n_unidir_funcs, scale=ckpt["obj_scale"]).to(cfg.data_device)
                    # if self.name == 'toy':
                    #     self.fc_occ_map_dict[cls_id][obj_id].load_state_dict(ckpt["FC_state_dict"][obj_id])
                    #     self.pe_dict[cls_id][obj_id].load_state_dict(ckpt["PE_state_dict"][obj_id])
                    #     self.bbox3d_dict[cls_id][obj_id] = None # dummy
                    # else:
                    self.fc_occ_map_dict[cls_id][obj_id].load_state_dict(ckpt["FC_state_dict"])
                    self.pe_dict[cls_id][obj_id].load_state_dict(ckpt["PE_state_dict"])
                    self.bbox3d_dict[cls_id][obj_id] = ckpt["bbox"]
        
        # world coord
        phi = torch.linspace(0, np.pi, 100)
        theta = torch.linspace(0, 2*np.pi, 100)
        phi, theta = torch.meshgrid(phi, theta)
        phi = phi.t()
        theta = theta.t()
        if suppress_unseen:
            phi_flat = phi.reshape(-1)
            theta_flat = theta.reshape(-1)
            upper = phi_flat < np.pi/2
            phi_upper = phi_flat[upper]
            theta_upper = theta_flat[upper]
            phi_other = np.pi - phi_upper
            theta_other = torch.where(theta_upper<np.pi, theta_upper+np.pi, theta_upper-np.pi)
            phi = torch.cat([phi_upper, phi_other]).reshape(100,100)
            theta = torch.cat([theta_upper, theta_other]).reshape(100,100)
        x_norm = torch.sin(phi) * torch.cos(theta)
        y_norm = torch.sin(phi) * torch.sin(theta)
        z_norm = torch.cos(phi)
        self.phi = phi.reshape(-1)
        self.theta = theta.reshape(-1)
        if view_uncertainty:
            cam_info = cameraInfo(cfg)
            n_bbox = 0
            for cls_id in self.fc_occ_map_dict.keys():
                for obj_id in self.inst_dict[cls_id].keys():
                    inst_info = self.inst_dict[cls_id][obj_id]
                    n_bbox += len(inst_info['frame_info'])               
            n_samples = len(self.fc_occ_map_dict.keys()) * 10000 // n_bbox
            n_samples_square = int(np.ceil(np.sqrt(n_samples))**2)
        for cls_id in self.fc_occ_map_dict.keys():
            # if not cls_id in cls_id_list:
            #     continue
            if not cls_id in self.count_dict.keys():
                self.count_dict[cls_id] = {}

            bounds = []
            if view_uncertainty:
                Twc_batch = []
                ray_dirs_batch = []
                depth_batch = []
                state_batch = []
            obj_ids = list(self.inst_dict[cls_id].keys())
            for obj_id in obj_ids:
                if view_uncertainty:
                    # bound = self.inst_dict[cls_id][obj_id]['bbox3D'].extent # obb
                    inst_info = self.inst_dict[cls_id][obj_id]
                    frame_info_list = inst_info['frame_info']
                    for frame_info in frame_info_list:
                        bbox_2d = frame_info['bbox']
                        idx_w = torch.rand(n_samples)
                        idx_h = torch.rand(n_samples)
                        # idx_w = (torch.arange(np.ceil(np.sqrt(n_samples))) + 0.5) / np.ceil(np.sqrt(n_samples))
                        # idx_h = (torch.arange(np.ceil(np.sqrt(n_samples))) + 0.5) / np.ceil(np.sqrt(n_samples))
                        # idx_w, idx_h = torch.meshgrid(idx_w, idx_h, indexing='xy')
                        # idx_w = idx_w.reshape(-1)
                        # idx_h = idx_h.reshape(-1)
                        idx_w = idx_w * (bbox_2d[1] - bbox_2d[0]) + bbox_2d[0]
                        idx_h = idx_h * (bbox_2d[3] - bbox_2d[2]) + bbox_2d[2]
                        idx_w = idx_w.long()
                        idx_h = idx_h.long()
                        ray_dirs = cam_info.rays_dir_cache[idx_w, idx_h].to(cfg.data_device)
                        ray_dirs_batch.append(ray_dirs)
                        
                        frame = frame_info['frame']
                        sample = self.sample_dict[frame]
                        Twc = sample['T'].astype(np.float32)
                        Twc_batch.append(np.tile(Twc[None, ...], (n_samples, 1, 1)))
                        
                        depth = sample["depth"]
                        obj = sample["obj_mask"]
                        state = np.zeros_like(obj, dtype=np.uint8)
                        state[obj == obj_id] = 1
                        state[obj == -1] = 2
                        depth_batch.append(depth[idx_w, idx_h])
                        state_batch.append(state[idx_w, idx_h])
                else:
                    # aabb = inst_dict[cls_id][obj_id]['bbox3D']
                    # bound = aabb.max_bound - aabb.min_bound
                    if use_gt_bound:
                        bound = self.inst_dict[cls_id][obj_id]['bbox3D'].extent
                    else:
                        points = np.asarray(self.inst_dict[cls_id][obj_id]['pcs'].points)
                        bound = points.max(axis=0) - points.min(axis=0) # aabb
                    bound = np.maximum(bound, 0.10) # scale at least 10cm
                    bounds.append(torch.from_numpy((bound/2).astype(np.float32)))
            
            if view_uncertainty:
                Twc_batch = torch.from_numpy(np.concatenate(Twc_batch)).to(cfg.data_device)
                depth_batch = torch.from_numpy(np.concatenate(depth_batch)).to(cfg.data_device)
                state_batch = torch.from_numpy(np.concatenate(state_batch)).to(cfg.data_device)
                ray_dirs_batch = torch.cat(ray_dirs_batch).to(cfg.data_device)
                batch_size = Twc_batch.shape[0]
                view_uncertainty_list = []
                for obj_id in obj_ids:
                    inst_info = self.inst_dict[cls_id][obj_id]
                    Two = torch.from_numpy(inst_info['T_obj'].astype(np.float32)).to(cfg.data_device)
                    Tco_batch = torch.linalg.inv(Twc_batch) @ Two[None, ...].repeat(batch_size,1,1)
                    origins, dirs_o = origin_dirs_O(Tco_batch, ray_dirs_batch)
                    
                    # as same as 'sample_3d_points' function
                    sampled_z = torch.zeros(
                        depth_batch.shape[0],
                        cfg.n_bins_cam2surface + cfg.n_bins,
                        dtype=depth_batch.dtype,
                        device=cfg.data_device)
                    invalid_depth_mask = depth_batch <= cfg.min_depth
                    max_bound = torch.max(depth_batch)
                    invalid_depth_count = invalid_depth_mask.count_nonzero()
                    if invalid_depth_count:
                        sampled_z[invalid_depth_mask, :] = stratified_bins(
                            cfg.min_depth, max_bound,
                            cfg.n_bins_cam2surface + cfg.n_bins, invalid_depth_count,
                            device=cfg.data_device)
                        
                    # sampling for valid depth rays
                    valid_depth_mask = ~invalid_depth_mask
                    valid_depth_count = valid_depth_mask.count_nonzero()

                    if valid_depth_count:
                        # Sample between min bound and depth for all pixels with valid depth
                        sampled_z[valid_depth_mask, :cfg.n_bins_cam2surface] = stratified_bins(
                            cfg.min_depth, depth_batch[valid_depth_mask]-cfg.surface_eps,
                            cfg.n_bins_cam2surface, valid_depth_count, device=cfg.data_device)

                        # sampling around depth for this object
                        obj_mask = (state_batch == 1) & valid_depth_mask # todo obj_mask
                        assert sampled_z.shape[0] == obj_mask.shape[0]
                        obj_count = obj_mask.count_nonzero()

                        if obj_count:
                            sampled_z[obj_mask, cfg.n_bins_cam2surface:] = normal_bins_sampling(
                                depth_batch[obj_mask],
                                cfg.n_bins,
                                obj_count,
                                delta=cfg.surface_eps,
                                device=cfg.data_device)
                        
                        # sampling around depth of other objects
                        other_obj_mask = (state_batch != 1) & valid_depth_mask
                        other_objs_count = other_obj_mask.count_nonzero()
                        if other_objs_count:
                            sampled_z[other_obj_mask, cfg.n_bins_cam2surface:] = stratified_bins(
                                depth_batch[other_obj_mask] - cfg.surface_eps,
                                depth_batch[other_obj_mask] + cfg.stop_eps,
                                cfg.n_bins, other_objs_count, device=cfg.data_device)
                            
                    xyz = origins[:, None, :] + (dirs_o[:, None, :] * sampled_z[..., None])
                    embedding_ = self.pe_dict[cls_id][obj_id](xyz.to(cfg.data_device))
                    sigmas, _ = self.fc_occ_map_dict[cls_id][obj_id](embedding_)
                    
                    occupancies = torch.sigmoid(10*sigmas.squeeze(-1)).detach().cpu()
                    term_probs = render_rays.occupancy_to_termination(occupancies).numpy()
                    entropies = np.sum(-term_probs*np.log(term_probs + 1e-10), axis=-1)
                    heuristic = np.sum(term_probs, axis=-1) * np.exp(-0.5*entropies)
                    reliability = calculate_reliability(heuristic, eta=0.9, m1=0.1, m2=0.15, M1=0.57, M2=0.65)
                    reliability = reliability.reshape(-1,n_samples) # [n_box (for cls_id), n_samples]
                    view_uncertainty_list.append(np.mean(1-reliability, axis=-1))
                view_uncertainty_list = np.mean(np.stack(view_uncertainty_list, axis=-1), axis=-1) # for each view, calculate class-level uncertainty by sum over all obj-level models
                box_idx = 0
                for obj_id in obj_ids:
                    inst_info = self.inst_dict[cls_id][obj_id]
                    frame_info_list = inst_info['frame_info']
                    for frame_info in frame_info_list:
                        frame_info['uncertainty'] = view_uncertainty_list[box_idx]
                        box_idx += 1
                print("hi")
            else:
                bounds = torch.stack(bounds, dim=0) 
                rs = 1.2*torch.sqrt(torch.square(bounds).sum(dim=-1))
                
                entropies_max_list = []
                metric_list = []
                obj_ids = list(self.fc_occ_map_dict[cls_id].keys())
                
                if view_uncertainty and len(obj_ids) <= 1:
                    continue
                
                for idx in range(len(obj_ids)):
                    obj_id = obj_ids[idx]
                    r = rs[idx]
                    x = r * x_norm
                    y = r * y_norm
                    z = r * z_norm
                    
                    rays_o_o = torch.stack([x, y, z], dim=-1).reshape(-1,3)
                    viewdir = -rays_o_o/r               
                    
                    # aabb = inst_dict[cls_id][obj_id]['bbox3D']
                    # center_np = ((aabb.max_bound + aabb.min_bound)/2).astype(np.float32)#np.asarray(inst_dict[cls_id][obj_id]['pcs'].points).mean(axis=0).astype(np.float32)
                    points = np.asarray(self.inst_dict[cls_id][obj_id]['pcs'].points)
                    if self.name == "replica":
                        center_np = ((points.max(axis=0) + points.min(axis=0))/2).astype(np.float32)
                    else: # TODO: do we have to?
                        center_np = (points.mean(axis=0)).astype(np.float32)
                    center = torch.from_numpy(center_np)
                    rays_o = center + rays_o_o
                    
                    far = 2*r
                    z_vals = stratified_bins(0, far, 96, rays_o.shape[0], device='cpu', z_fixed=True)
                    xyz = rays_o[..., None, :] + (viewdir[:, None, :] * z_vals[..., None])
                    embedding_ = self.pe_dict[cls_id][obj_id](xyz.to(cfg.data_device))
                    sigmas, rgbs = self.fc_occ_map_dict[cls_id][obj_id](embedding_)
                    occupancies = torch.sigmoid(10*sigmas.squeeze(-1)).detach().cpu()
                    occ_sum = torch.sum(occupancies, dim=-1)
                    
                    mask = occ_sum > 0.01 # 10
                    term_probs_ = render_rays.occupancy_to_termination(occupancies).numpy()                                      
                    term_probs = term_probs_[mask]
                    
                    entropies = np.sum(-term_probs_*np.log(term_probs_ + 1e-10), axis=-1)
                    if view_uncertainty and suppress_unseen:
                        term_probs_half = term_probs_[:term_probs_.shape[0]//2]
                        term_probs_other = term_probs_[term_probs_.shape[0]//2:]
                        peak_half = np.argmax(term_probs_half, axis=-1)
                        peak_other = np.argmax(term_probs_other, axis=-1)
                        suppress_dir = np.abs(peak_half+peak_other-96)< 2
                        
                        entropies_half = entropies[:entropies.shape[0]//2]
                        entropies_other = entropies[entropies.shape[0]//2:]
                        compare = entropies_half > entropies_other
                        suppress_half = suppress_dir & (~compare)
                        suppress_other = suppress_dir & compare
                        
                        entropies_half[suppress_half, peak_half[suppress_half]] = 0
                        entropies_other[suppress_other, peak_other[suppress_other]] = 0
                        entropies = np.concatenate([entropies_half, entropies_other], axis=0)
                    else:
                        entropies_max_list.append(entropies.max())
                        # # measure option 1: mean
                        # measure = entropies.mean()
                        # self.count_dict[cls_id][obj_id] = measure
                        
                    if use_reliability:
                        heuristic = np.sum(term_probs_, axis=-1) * np.exp(-0.5*entropies)
                        reliability = calculate_reliability(heuristic, eta=0.9, m1=0.1, m2=0.15, M1=0.57, M2=0.65)
                        metric_list.append(1-reliability)
                        # plot_reliability(reliability, x.numpy(), y.numpy(), z.numpy(),
                        #                     mesh_dir='/media/satassd_1/tblee-larr/CVPR24/vMAP_offline/logs/0718/toy/scene_mesh',
                        #                     obj_id=obj_id, center_np=center_np, r=r.numpy())
                    else:
                        metric_list.append(entropies)
                
                # measure option 2: rate of below threshold
                if use_reliability:
                    for i in range(len(obj_ids)):
                        obj_id = obj_ids[i]
                        metric = metric_list[i]
                        measure = metric[metric<0.5].shape[0]
                        self.count_dict[cls_id][obj_id] = measure
                else:
                    threshold = 0.8 * min(entropies_max_list)
                    for i in range(len(obj_ids)):
                        obj_id = obj_ids[i]
                        entropies = metric_list[i]
                        measure = entropies[entropies<threshold].shape[0]
                        self.count_dict[cls_id][obj_id] = measure
    
    def align_poses(self, multi_init_pose=True, use_best_relative_transform=False, eta1=0.06, eta2=0.15, eta3=0.3):
        from teaser_fpfh_icp import TEASER_FPFH_ICP
        print('align_poses')
        t1 = time.time()
        
        if self.name == "replica":
            cls_id_add = 100
        else:
            cls_id_add = 10000
        
        self.chamfer_dict = {}
        self.chamfer_opposite_dict = {}
        self.id_representative_dict = {}
        while self.bbox3d_dict:
            for cls_id in self.bbox3d_dict.copy().keys():
                self.chamfer_dict[cls_id] = {}
                self.chamfer_opposite_dict[cls_id] = {}
                obj_ids = list(self.bbox3d_dict[cls_id].keys())
                counts = [self.count_dict[cls_id][obj_id] for obj_id in self.count_dict[cls_id].keys()]
                if len(counts) > 1:
                    counts = np.array(counts)
                    if self.representative_metric == "uncertainty":
                        idx_representative = np.argmax(counts)
                    elif self.representative_metric == "random":
                        idx_representative = np.random.choice(np.arange(len(counts)))
                else:
                    idx_representative = 0

                inst_dict_cls = self.inst_dict[cls_id]
                
                # get pose for representative
                obj_id_representative = obj_ids[idx_representative]
                inst_pcs_template = inst_dict_cls[obj_id_representative]['pcs']
                T_obj, bbox3D = get_pose_from_pointcloud(inst_pcs_template)
                inst_dict_cls[obj_id_representative]['T_obj'] = T_obj
                if bbox3D is not None:
                    inst_dict_cls[obj_id_representative]['bbox3D'] = bbox3D
                
                self.id_representative_dict[cls_id] = obj_id_representative
                
                other_obj_ids = []
                for idx in range(len(obj_ids)):
                    if idx != idx_representative:
                        obj_id = obj_ids[idx]
                        other_obj_ids.append(obj_id)

                if len(other_obj_ids) == 0:
                    # inst_dict_cls[obj_id_representative]['bbox3D'] = get_bound(inst_pcs_template)
                    self.bbox3d_dict.pop(cls_id) 
                    continue
                
                # # # visualize representatives
                # self.visualize_representative(cls_id, obj_ids)
                
                # idx_representative
                # inst_dict_cls[obj_id_representative]['bbox3D'] = get_bound(inst_pcs_template)
                T_obj_template = np.copy(inst_dict_cls[obj_id_representative]['T_obj'])
                scale_template = np.linalg.det(T_obj_template[:3, :3]) ** (1/3)
                T_obj_template[:3, :3] = T_obj_template[:3, :3]/scale_template
                template_np_w = np.array(inst_pcs_template.points)
                    
                template = torch.from_numpy(template_np_w.transpose(1,0)).unsqueeze(0).to(self.device)
                if multi_init_pose:
                    transform_list = get_possible_transform_from_bbox()
                    template_np_w_list = []
                    for transform in transform_list:
                        template_np_w_transformed = transform_pointcloud(template_np_w, transform)
                        template_np_w_list.append(template_np_w_transformed)
                    template = torch.from_numpy(np.stack(template_np_w_list).transpose(0,2,1)).to(self.device)

                for idx in range(len(other_obj_ids)):
                    obj_id = other_obj_ids[idx]
                    inst_pcs = inst_dict_cls[obj_id]['pcs']
                    source_np_w = np.array(inst_pcs.points)
                    
                    scale_source = np.max(source_np_w.max(axis=0)-source_np_w.min(axis=0))/2 # 0720
                    
                    # use TEASER++
                    source = torch.from_numpy(source_np_w.transpose(1,0)).unsqueeze(0).to(self.device)
                    teaser = TEASER_FPFH_ICP(source, voxel_size=0.1, spc=True, device=self.device, visualize=False)
                    R_rel, t_rel = teaser.forward(template)
                    if multi_init_pose:
                        T_rel_multi = np.repeat(np.eye(4)[None, ...], template.shape[0], axis=0)
                        T_rel_multi[:, :3, :3] = R_rel.detach().cpu().numpy()
                        T_rel_multi[:, :3, 3:] = t_rel.detach().cpu().numpy()
                        chamfer_unidir_list = np.zeros(T_rel_multi.shape[0])
                        for idx_cand in range(T_rel_multi.shape[0]):
                            T_rel = np.linalg.inv(transform_list[idx_cand]) @ T_rel_multi[idx_cand]
                            source_transformed = transform_pointcloud(source_np_w, T_rel)
                            inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                            chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source # important: normalize cd metric!!
                            chamfer_unidir_list[idx_cand] = chamfer_unidir

                        idx_sel = np.argmin(chamfer_unidir_list)
                        T_rel = np.linalg.inv(transform_list[idx_sel]) @ T_rel_multi[idx_sel]
                        chamfer_unidir = chamfer_unidir_list[idx_sel]
                        
                    else:
                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R_rel.squeeze(0).detach().cpu().numpy()
                        T_rel[:3, 3:] = t_rel.squeeze(0).detach().cpu().numpy()
                        source_transformed = transform_pointcloud(source_np_w, T_rel)
                        inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                        chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source
                    
                    if use_best_relative_transform:
                        se3 = liegroups.SE3.from_matrix(T_rel, normalize=True).log()
                        se3_sampled = np.random.normal(loc=se3, scale=0.05, size=(99,6))
                        T_rel_samples = np.zeros((100,4,4))
                        T_rel_samples[0] = T_rel
                        chamfer_unidirs = np.zeros(100)
                        chamfer_unidirs[0] = chamfer_unidir
                        for sample_idx in range(99):
                            T_rel_sample = liegroups.SE3.exp(se3_sampled[sample_idx]).as_matrix()
                            T_rel_samples[sample_idx+1] = T_rel_sample
                            source_transformed_ = transform_pointcloud(source_np_w, T_rel_sample)
                            inst_pcs_transformed_ = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed_))
                            chamfer_unidir_ = np.asarray(inst_pcs_transformed_.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source
                            chamfer_unidirs[sample_idx+1] = chamfer_unidir_
                        min_sample_idx = np.argmin(chamfer_unidirs)
                        T_rel = T_rel_samples[min_sample_idx] 
                        chamfer_unidir = chamfer_unidirs[min_sample_idx]                   
                    
                    # consider as other subcategory if observed point cloud is note aligned well
                    self.chamfer_dict[cls_id][obj_id] = chamfer_unidir
                    if chamfer_unidir > eta2: # TODO: do we need different threshold for scannet?
                        subcategorize = True
                    elif chamfer_unidir < eta1:
                        subcategorize = False
                    else:
                        chamfer_opposite = np.asarray(inst_pcs_template.compute_point_cloud_distance(inst_pcs_transformed)).mean()/scale_template
                        self.chamfer_opposite_dict[cls_id][obj_id] = chamfer_opposite
                        if chamfer_opposite > eta3:
                            subcategorize = True
                        else:
                            subcategorize = False
                    
                    if not self.subcategorize:
                        subcategorize = False
                    
                    if subcategorize:
                        cls_id_sub = cls_id + cls_id_add
                        inst_dict = inst_dict_cls[obj_id]
                        metric = self.count_dict[cls_id][obj_id]
                        bbox3d = self.bbox3d_dict[cls_id][obj_id]
                        
                        if not cls_id_sub in self.inst_dict.keys():
                            self.inst_dict[cls_id_sub] = {}
                        self.inst_dict[cls_id_sub].update({obj_id: inst_dict})
                        if not cls_id_sub in self.count_dict.keys():
                            self.count_dict[cls_id_sub] = {}
                        self.count_dict[cls_id_sub].update({obj_id: metric})
                        if not cls_id_sub in self.bbox3d_dict.keys():
                            self.bbox3d_dict[cls_id_sub] = {}
                        self.bbox3d_dict[cls_id_sub].update({obj_id: bbox3d})
                        if not cls_id_sub in self.pe_dict.keys():
                            self.pe_dict[cls_id_sub] = {}
                        self.pe_dict[cls_id_sub].update({obj_id: self.pe_dict[cls_id][obj_id]})
                        if not cls_id_sub in self.fc_occ_map_dict.keys():
                            self.fc_occ_map_dict[cls_id_sub] = {}
                        self.fc_occ_map_dict[cls_id_sub].update({obj_id: self.fc_occ_map_dict[cls_id][obj_id]})
                        
                        inst_dict_cls.pop(obj_id, None)
                        self.count_dict[cls_id].pop(obj_id, None)
                        self.bbox3d_dict[cls_id].pop(obj_id, None)
                        self.pe_dict[cls_id].pop(obj_id, None)
                        self.fc_occ_map_dict[cls_id].pop(obj_id, None)
                    else: #elif (cls_id == 20 and obj_id in [42,78,144,186,188]) or (cls_id != 20 and chamfer_unidir < 0.1):
                        T_obj = np.linalg.inv(T_rel) @ T_obj_template # if template = T_rel @ source
                        inst_dict_cls[obj_id]['T_obj'] = T_obj # center to aligned position
                        
                        # bound to obb w.r.t aligned pose
                        get_obb(inst_dict_cls[obj_id])

                        # source_transformed = transform_pointcloud(source_np_w, T_rel)# + source_np_w_mean
                        # inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                        # inst_pcs_transformed.paint_uniform_color([1.0, 0.0, 0.0])
                        # inst_pcs_template.paint_uniform_color([0.0, 0.0, 1.0])
                        # open3d.visualization.draw_geometries([self.scene_mesh, inst_pcs_transformed, inst_pcs_template])
                
                self.bbox3d_dict.pop(cls_id)       

        t2 = time.time()
        print('align_poses takes {} seconds'.format(t2-t1))

    def select_poses(self, model_type='iPCRNet', 
                     weight_root='./Thirdparty/learning3d/pretrained'):
        print('select poses')
        t1 = time.time()
        # sys.path.append('/media/satassd_1/tblee-larr/CVPR24/vMap_plus_copy/Thirdparty')
        sys.path.append('./Thirdparty')
        from learning3d.models import PointNet, PointNetLK, iPCRNet
        if model_type == 'PointNetLK':
            registration_model = PointNetLK(feature_model=PointNet(use_bn=True), 
                                delta=1e-02, xtol=1e-07, p0_zero_mean=False, p1_zero_mean=False, pooling='max')
            ckpt_path = os.path.join(weight_root, 'exp_pnlk', 'models', 'best_model.t7')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        elif model_type == 'iPCRNet':
            registration_model = iPCRNet(feature_model=PointNet(), pooling='max')
            ckpt_path = os.path.join(weight_root, 'exp_ipcrnet', 'models', 'best_model.t7')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
        registration_model.load_state_dict(checkpoint)
        registration_model = registration_model.to(self.device)
        
        cls_id_list = [54]
        for cls_id in self.inst_dict.keys():
            if cls_id == 0:
                continue
            # if not cls_id in cls_id_list:
            #     continue
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            idx_representative = np.random.choice(len(obj_ids))
            
            other_obj_ids = []
            for idx in range(len(obj_ids)):
                if idx != idx_representative:
                    obj_id = obj_ids[idx]
                    other_obj_ids.append(obj_id)
            
            if len(other_obj_ids) == 0:
                obj_id = obj_ids[0]
                inst_pcs_template = inst_dict_cls[obj_id]['pcs']
                inst_dict_cls[obj_id]['bbox3D'] = get_bound(inst_pcs_template)
                continue
            
            # idx_representative
            obj_id = obj_ids[idx_representative]
            inst_pcs_template = inst_dict_cls[obj_id]['pcs']
            inst_dict_cls[obj_id]['bbox3D'] = get_bound(inst_pcs_template)
            template_np_w = np.array(inst_pcs_template.points)
            T_obj_template = np.copy(inst_dict_cls[obj_id]['T_obj'])
                
            # change input if learning based method
            template_np = transform_pointcloud(template_np_w, np.linalg.inv(T_obj_template))
            template = torch.from_numpy(template_np[None, ...].astype(np.float32)).to(self.device)
            with torch.no_grad():
                template_feature = registration_model.pooling(registration_model.feature_model(template)).detach().cpu().numpy() # [B, N, 3] -> [B, K]
            
            for idx in range(len(other_obj_ids)):
                obj_id = other_obj_ids[idx]
                inst_pcs = inst_dict_cls[obj_id]['pcs']
                source_np_w = np.array(inst_pcs.points)

                T_obj = np.copy(inst_dict_cls[obj_id]['T_obj'])
                source_nps = []
                T_obj_cands = []
                T_rel_cands = get_possible_transform_from_bbox()
                for i in range(len(T_rel_cands)):
                    T_obj_cand = T_obj @ T_rel_cands[i]
                    source_np = transform_pointcloud(source_np_w, np.linalg.inv(T_obj_cand))
                    source_nps.append(source_np)
                    T_obj_cands.append(T_obj_cand)
                source_nps = np.stack(source_nps, 0)
                source = torch.from_numpy(source_nps.astype(np.float32)).to(self.device)
                with torch.no_grad():
                    source_feature = registration_model.pooling(registration_model.feature_model(source)).detach().cpu().numpy() # [B, N, 3] -> [B, K]
                r = np.linalg.norm(source_feature - template_feature, axis=-1)
                idx_sel = np.argmin(r)
                T_obj_sel = T_obj_cands[idx_sel]
                inst_dict_cls[obj_id]['T_obj'] = T_obj_sel # core process
        
        t2 = time.time()  
        print('select_poses takes {} seconds'.format(t2-t1))     
    
    def visualize_pcs(self): # for debug initial poses
        geometries_visualized = []
        for cls_id in self.inst_dict.keys():
            inst_dict_cls = self.inst_dict[cls_id]
            if cls_id == 0:
                pcs = inst_dict_cls['pcs']
                geometries_visualized.append(pcs)
            else:
                for inst_id in inst_dict_cls.keys():
                    pcs = inst_dict_cls[inst_id]['pcs']
        open3d.visualization.draw_geometries(geometries_visualized)
    
    def visualize_coords(self): # for debug initial poses
        # id_to_rgb = {47: [0,255,0], 147: [0,255,0], 247: [0,255,0],
        #              26: [0,0,255], 7: [255,255,0], 11: [128,0,128], 111: [128,0,128], 54: [255,165,0], 56: [0,255,255]}
        scene_mesh = trimesh.load(self.scene_mesh_file)
        scene_mesh_o3d = trimesh_to_open3d(scene_mesh)
        geometries_visualized = [scene_mesh_o3d]
        cls_id_list = [80,180,280]#[61,161,261,47,147,247,26,7,11,111,54,56]
        for cls_id in self.inst_dict.keys():
            # if not cls_id in cls_id_list:
            #     continue
            label_idx = (cls_id+1) % 256
            label_color = imgviz.label_colormap()[label_idx]/255.0
            inst_dict_cls = self.inst_dict[cls_id]
            # if cls_id == 0:
            #     bbox3D = inst_dict_cls['bbox3D']
            #     bbox3D_o3d = open3d.geometry.OrientedBoundingBox(bbox3D.center, bbox3D.R, bbox3D.extent)
            #     bbox3D_o3d.color = label_color
            #     geometries_visualized += [bbox3D_o3d]
            # else:
            if cls_id != 0 and cls_id != 47:
                for inst_id in inst_dict_cls.keys():
                    # if inst_id in [22, 53, 43, 40]:
                    #     continue
                    if 'bbox3D' in inst_dict_cls[inst_id].keys():
                        bbox3D = inst_dict_cls[inst_id]['bbox3D']
                        bbox3D_o3d = open3d.geometry.OrientedBoundingBox(bbox3D.center, bbox3D.R, bbox3D.extent)
                        bbox3D_o3d.color = label_color
                        # if cls_id % 100 == 61:
                        #     if len(inst_dict_cls.keys()) == 4:
                        #         bbox3D_o3d.color = np.array([255,0,0])/255
                        #     else:
                        #         bbox3D_o3d.color = np.array([255,128,128])/255
                        # else:
                        #     bbox3D_o3d.color = np.array(id_to_rgb[cls_id])/255#label_color
                        T_obj = np.copy(inst_dict_cls[inst_id]['T_obj'])
                        # coord_obj = copy.deepcopy(coord).transform(T_obj)
                        scale = np.linalg.det(T_obj[:3,:3])**(1/3)
                        T_obj[:3,:3] = T_obj[:3,:3]/scale
                        coord_ = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]).scale(1.4*scale, np.zeros(3))
                        coord_obj = coord_.transform(T_obj)
                        # pc = inst_dict_cls[inst_id]['pcs']
                        # line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(bbox3D_o3d)
                        # line_set.lines.line_width = 5.0
                        # line_set.lines.colors = open3d.utility.Vector3dVector([id_to_rgb[cls_id]] * len(line_set.lines.points))
                        geometries_visualized += [bbox3D_o3d]
        open3d.visualization.draw_geometries(geometries_visualized)

    def calculate_mask_rate(self):
        print('calculate_mask_rate')
        
        self.mask_rate_dict = {}
        cls_id_list = [54]
        for cls_id in self.inst_dict.keys():
            # if not cls_id in cls_id_list:
            #     continue
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if cls_id != 0 and n_obj > 1:
                self.mask_rate_dict[cls_id] = {}
                for idx in range(n_obj):
                    inst_id = obj_ids[idx]
                    inst_info = inst_dict_cls[inst_id]
                    inst_pcs_sampled = inst_info['pcs'].voxel_down_sample(0.05)
                    inst_pointcloud_w = np.asarray(inst_pcs_sampled.points)
                    n_pts = inst_pointcloud_w.shape[0]
                    
                    self.mask_rate_dict[cls_id][inst_id] = []
                    frame_info_list = inst_info['frame_info']
                    for idx, frame_info in enumerate(frame_info_list):
                        frame = frame_info['frame']
                        sample = self.sample_dict[frame]
                        Tcw = np.linalg.inv(sample['T'])
                        obj_mask = sample['obj_mask']
                        
                        # Project 3D points to image plane
                        inst_pointcloud_c = transform_pointcloud(inst_pointcloud_w, Tcw)
                        pixels_homo = ((self.K).dot(inst_pointcloud_c.T)).T
                        pixels_uv = (pixels_homo[:, :2] / pixels_homo[:, 2:])
                        
                        in_fov = (pixels_uv[:, 0] > 0) & (pixels_uv[:, 0] < self.W) & \
                            (pixels_uv[:, 1] > 0) & (pixels_uv[:, 1] < self.H)
                        pixels_uv = pixels_uv[in_fov].astype(np.int32)
                        pixels_uv = np.unique(pixels_uv, axis=0)
                        # n_pts = pixels_uv.shape[0]
                        
                        # # DEBUG
                        # debug_image = np.copy(sample["image"])
                        # debug_image[obj_mask==inst_id] = np.array([[255, 0, 0]])
                        # debug_image[pixels_uv[:,0], pixels_uv[:,1]] = np.array([[0, 0, 255]])
                        # debug_image = debug_image.transpose(1,0,2)
                        # save_dir = os.path.join("logs/0809/debug_mask_rate", str(inst_id))
                        # os.makedirs(save_dir, exist_ok=True)
                        # save_path = os.path.join(save_dir, f"{frame}.png")
                        # cv2.imwrite(save_path, debug_image)
                        
                        n_pts_in_mask = np.count_nonzero(obj_mask[pixels_uv[:,0], pixels_uv[:,1]] == inst_id.item())
                        self.mask_rate_dict[cls_id][inst_id].append(1-n_pts_in_mask/n_pts)
                    self.mask_rate_dict[cls_id][inst_id] = torch.from_numpy(np.array(self.mask_rate_dict[cls_id][inst_id]))
    
    def set_template_scale(self):
        self.scale_template_dict = {}
        cls_id_list = [54]
        for cls_id in self.inst_dict.keys():
            if cls_id == 0:
                continue
            obj_ids = list(self.inst_dict[cls_id].keys())
            # if not cls_id in cls_id_list:
            #     continue
            if len(obj_ids) > 1:
                counts = np.array([self.count_dict[cls_id][obj_id] for obj_id in obj_ids])
                idx_representative = np.argmax(counts)
            else:
                idx_representative = 0
            obj_id_representative = obj_ids[idx_representative]
            T_obj_template = np.copy(self.inst_dict[cls_id][obj_id_representative]['T_obj'])
            scale_template = np.linalg.det(T_obj_template[:3, :3]) ** (1/3)
            self.scale_template_dict[cls_id] = scale_template
            for obj_id in obj_ids:
                T_obj = self.inst_dict[cls_id][obj_id]['T_obj']
                T_obj[:3, :3] = T_obj[:3, :3]/scale_template
    
    def get_observation_heuristic(self):
        self.heuristic_dict = {}
        for cls_id in self.inst_dict.keys():
            if cls_id == 0:
                continue
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if n_obj > 1:
                self.heuristic_dict[cls_id] = {}
                for obj_id in obj_ids:
                    inst_info = inst_dict_cls[obj_id]
                    frame_info_list = inst_info['frame_info']
                    pc = np.array(inst_info['pcs'].points)
                    two = pc.mean(axis=0)
                    dirs = []
                    for frame_info in frame_info_list:
                        frame = frame_info['frame']
                        sample = self.sample_dict[frame]
                        twc = sample['T'][:3,3]
                        dir = two - twc
                        dir = dir/np.linalg.norm(dir)
                        dirs.append(dir)
                    dirs = np.array(dirs)
                    self.heuristic_dict[cls_id][obj_id] = np.sqrt(np.abs(np.linalg.det(dirs.T @ dirs)))


class ToyDataset(BaseDataSet):
    def __init__(self, cfg):
        self.object_wise_model = cfg.object_wise_model
        self.codenerf = cfg.codenerf
        self.name = "toy"
        self.device = cfg.data_device
        self.root_dir = cfg.dataset_dir
        self.instance_paths = sorted(glob.glob(os.path.join(self.root_dir, 'instances', '*.txt')))
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])

        self.W = cfg.W
        self.H = cfg.H
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.edge = cfg.mw
        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        self.K = np.eye(3)
        self.K[0,0] = self.fx
        self.K[1,1] = self.fy
        self.K[0,2] = self.cx
        self.K[1,2] = self.cy

        # Not sure: door-37 handrail-43 lamp-47 pipe-62 rack-66 shower-stall-73 stair-77 switch-79 wall-cabinet-94 picture-59
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2
        
        self.n_img = len(os.listdir(os.path.join(self.root_dir, "depth")))
        
        if cfg.get_frames_incremental:
            self.get_frames(cfg.frames[0], init=True)
        else:
            self.get_all_frames()
        self.get_all_poses_toy()
        if cfg.subcategorize:
            self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained, use_gt_bound=True) # iteration=cfg.it_add_obs[0]
            self.subcategorize_objects()
        else:
            self.id_representative_dict = {1: 0}
        # self.chamfer_dict = None
        # self.chamfer_opposite_dict = None

    def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        self.sample_dict = {}
        self.rgb_list = []
        self.depth_list = []
        self.obj_list = []
        for idx in range(self.n_img):
            rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
            depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
            inst_file = os.path.join(self.root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")
            obj_file = os.path.join(self.root_dir, "semantic_class", "semantic_class_" + str(idx) + ".png")
            
            depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1,0)            
            image = cv2.imread(rgb_file).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1,0,2)
            obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
            inst = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32
            
            bbox_scale = self.bbox_scale
            
            obj_ = np.zeros_like(obj)
            cls_list = []
            inst_list = []
            batch_masks = []

            for inst_id in np.unique(inst):
                inst_mask = inst == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                sem_cls = np.unique(obj[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] == 1
                sem_cls = sem_cls[0]
                obj_mask = inst == inst_id
                batch_masks.append(obj_mask)
                if sem_cls == 0 and inst_id != 0: # undefined class: 0 in data, -1 in our system
                    cls_list.append(inst_id+1000)
                else:
                    cls_list.append(sem_cls)
                inst_list.append(inst_id)
                            
            if len(batch_masks) > 0:
                batch_masks = torch.from_numpy(np.stack(batch_masks))
                cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)

                for i in range(batch_masks.shape[0]):
                    w = rmaxs[i] - rmins[i]
                    h = cmaxs[i] - cmins[i]
                    if w <= 10 or h <= 10:  # too small   todo
                        continue
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                 w=obj.shape[1], h=obj.shape[0])
                    sem_cls = cls_list[i]
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    
                    if not sem_cls in self.inst_dict.keys():
                        self.inst_dict[sem_cls] = {}
                    bbox = torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))
                    if not inst_id in self.inst_dict[sem_cls].keys():
                        self.inst_dict[sem_cls][inst_id] = {'frame_info': [{'frame': idx, 'bbox': bbox}]}
                    else:
                        self.inst_dict[sem_cls][inst_id]['frame_info'].append({'frame': idx, 'bbox': bbox})

            inst[obj_ == 0] = 0  # for background

            if idx == 0:
                self.inst_dict[0] = {'frame_info': []}
            background_mask = inst # or obj_, maybe both OK
            self.inst_dict[0]['frame_info'].append({'frame': idx, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))})

            T = self.Twc[idx]   # could change to ORB-SLAM pose or else    

            if self.depth_transform:
                depth = self.depth_transform(depth) 
            
            sample = {"image": image, "depth": depth, "obj_mask": inst, "T": T, "frame_id": idx} #"sem_mask": obj, 
                             
            if image is None or depth is None:
                print(rgb_file)
                print(depth_file)
                raise ValueError
            
            # self.sample_list.append(sample)
            self.sample_dict[idx] = sample  
            
            self.rgb_list.append(image)
            self.depth_list.append(depth)
            self.obj_list.append(inst)
        
        self.rgb_list = torch.from_numpy(np.stack(self.rgb_list, axis=0))
        self.depth_list = torch.from_numpy(np.stack(self.depth_list, axis=0))
        self.obj_list = torch.from_numpy(np.stack(self.obj_list, axis=0))
        
        t2 = time.time()
        print('get_all_frames takes {} seconds'.format(t2-t1))   
    
    def get_frames(self, frame_ids, init=False):
        print('get_frames')
        t1 = time.time()
        if not hasattr(self, 'inst_dict'):
            self.inst_dict = {}
        if not hasattr(self, 'sample_dict'):
            self.sample_dict = {}
        rgb_list = []
        depth_list = []
        obj_list = []
        
        prev_obj_cls_dict = {}
        prev_cls_obj_dict = {}
        for cls_id in self.inst_dict.keys():
            if cls_id == 0:
                continue
            for obj_id in self.inst_dict[cls_id].keys():
                prev_obj_cls_dict[obj_id] = cls_id
                prev_cls_obj_dict[cls_id] = obj_id
        
        for ii, idx in enumerate(frame_ids):
            rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
            depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
            inst_file = os.path.join(self.root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")
            obj_file = os.path.join(self.root_dir, "semantic_class", "semantic_class_" + str(idx) + ".png")
            
            depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1,0)            
            image = cv2.imread(rgb_file).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1,0,2)
            obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
            inst = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32
            
            bbox_scale = self.bbox_scale
            
            obj_ = np.zeros_like(obj)
            cls_list = []
            inst_list = []
            batch_masks = []

            for inst_id in np.unique(inst):
                inst_mask = inst == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                sem_cls = np.unique(obj[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] == 1
                sem_cls = sem_cls[0]
                obj_mask = inst == inst_id
                batch_masks.append(obj_mask)
                if sem_cls == 0 and inst_id != 0: # undefined class: 0 in data, -1 in our system
                    cls_list.append(inst_id+1000)
                else:
                    cls_list.append(sem_cls)
                inst_list.append(inst_id)
                            
            if len(batch_masks) > 0:
                batch_masks = torch.from_numpy(np.stack(batch_masks))
                cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)

                for i in range(batch_masks.shape[0]):
                    w = rmaxs[i] - rmins[i]
                    h = cmaxs[i] - cmins[i]
                    if w <= 10 or h <= 10:  # too small   todo
                        continue
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                    w=obj.shape[1], h=obj.shape[0])
                    sem_cls = cls_list[i]
                    if not init and sem_cls != 0 and not self.object_wise_model:
                        cls_id_add = max([cls_id_-cls_id_%100 for cls_id_ in prev_cls_obj_dict.keys() if cls_id_ % 100 == sem_cls]) + 100
                        sem_cls += cls_id_add
                    
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    
                    if not sem_cls in self.inst_dict.keys():
                        self.inst_dict[sem_cls] = {}
                    bbox = torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))
                    
                    
                    if inst_id in prev_obj_cls_dict.keys():
                        cls_id = prev_obj_cls_dict[inst_id]
                        self.inst_dict[cls_id][inst_id]['frame_info'].append({'frame': idx, 'bbox': bbox})
                    elif not inst_id in self.inst_dict[sem_cls].keys():
                        self.inst_dict[sem_cls][inst_id] = {'frame_info': [{'frame': idx, 'bbox': bbox}]}
                    else:
                        self.inst_dict[sem_cls][inst_id]['frame_info'].append({'frame': idx, 'bbox': bbox})

            inst[obj_ == 0] = 0  # for background

            if ii == 0:
                self.inst_dict[0] = {'frame_info': []}
            background_mask = inst # or obj_, maybe both OK
            self.inst_dict[0]['frame_info'].append({'frame': idx, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))})

            T = self.Twc[idx]   # could change to ORB-SLAM pose or else    

            if self.depth_transform:
                depth = self.depth_transform(depth) 
            
            sample = {"image": image, "depth": depth, "obj_mask": inst, "T": T, "frame_id": idx} #"sem_mask": obj, 
                                
            if image is None or depth is None:
                print(rgb_file)
                print(depth_file)
                raise ValueError
            
            # self.sample_list.append(sample)
            self.sample_dict[idx] = sample  
            
            rgb_list.append(image)
            depth_list.append(depth)
            obj_list.append(inst)
        
        rgb_list = torch.from_numpy(np.stack(rgb_list, axis=0))
        depth_list = torch.from_numpy(np.stack(depth_list, axis=0))
        obj_list = torch.from_numpy(np.stack(obj_list, axis=0))
        
        self.rgb_list = rgb_list if not hasattr(self, 'rgb_list') else torch.cat([self.rgb_list, rgb_list], dim=0)
        self.depth_list = depth_list if not hasattr(self, 'depth_list') else torch.cat([self.depth_list, depth_list], dim=0)
        self.obj_list = obj_list if not hasattr(self, 'obj_list') else torch.cat([self.obj_list, obj_list], dim=0)
        
        t2 = time.time()
        print('get_frames takes {} seconds'.format(t2-t1))
    
    def get_all_poses_toy(self):
        print('get_all_poses')
        t1 = time.time()
        for cls_id in self.inst_dict.keys():
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if cls_id == 0:
                background_list = inst_dict_cls['frame_info']
                background_pcs = accumulate_pointcloud(0, background_list, self.sample_dict, self.intrinsic_open3d)
                transform, extents = trimesh.bounds.oriented_bounds(np.asarray(background_pcs.points))  # pc
                transform = np.linalg.inv(transform)
                bbox3D = BoundingBox()
                bbox3D.center = transform[:3, 3]
                bbox3D.R = transform[:3, :3]
                bbox3D.extent = extents
                inst_dict_cls['bbox3D'] = bbox3D
                inst_dict_cls['pcs'] = background_pcs
                continue
            for idx in range(n_obj):
                inst_id = obj_ids[idx]
                inst_list = inst_dict_cls[inst_id]['frame_info']
                inst_pcs = accumulate_pointcloud(inst_id, inst_list, self.sample_dict, self.intrinsic_open3d)

                inst_dict_cls[inst_id]['pcs'] = inst_pcs

                T_obj, bbox3D = self.get_pose_from_gt(inst_id)
                inst_dict_cls[inst_id]['T_obj'] = T_obj
                inst_dict_cls[inst_id]['bbox3D'] = bbox3D
                
        t2 = time.time()
        print('get_all_poses takes {} seconds'.format(t2-t1))
    
    def get_poses_toy(self, previous_obj_ids=[]):
        print('get_poses')
        t1 = time.time()
        for cls_id in self.inst_dict.keys():
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if cls_id == 0:
                if len(previous_obj_ids) == 0:
                    background_list = inst_dict_cls['frame_info']
                    background_pcs = accumulate_pointcloud(0, background_list, self.sample_dict, self.intrinsic_open3d)
                    transform, extents = trimesh.bounds.oriented_bounds(np.asarray(background_pcs.points))  # pc
                    transform = np.linalg.inv(transform)
                    bbox3D = BoundingBox()
                    bbox3D.center = transform[:3, 3]
                    bbox3D.R = transform[:3, :3]
                    bbox3D.extent = extents
                    inst_dict_cls['bbox3D'] = bbox3D
                    inst_dict_cls['pcs'] = background_pcs
                continue
            for idx in range(n_obj):
                inst_id = obj_ids[idx]
                if inst_id not in previous_obj_ids:
                    inst_list = inst_dict_cls[inst_id]['frame_info']
                    inst_pcs = accumulate_pointcloud(inst_id, inst_list, self.sample_dict, self.intrinsic_open3d)

                    inst_dict_cls[inst_id]['pcs'] = inst_pcs

                    T_obj, bbox3D = self.get_pose_from_gt(inst_id)
                    inst_dict_cls[inst_id]['T_obj'] = T_obj
                    inst_dict_cls[inst_id]['bbox3D'] = bbox3D
                
        t2 = time.time()
        print('get_poses takes {} seconds'.format(t2-t1))  
    
    def get_pose_from_gt(self, inst_id):
        instance_path = self.instance_paths[inst_id-1]
        lines = [line.split(" ") for line in open(instance_path, 'r')]
        line = lines[0]
        assert len(line) == 9
        line = [float(x) for x in line]
        # get origin, extend, rz in EDN: 
        origin = 0.01*np.array(line[:3])
        dx, dy, dz = line[4], line[5], line[3]
        scale = 0.01*np.array([dx, dy, dz])
        
        rz = np.pi * np.array([line[8]]) / 180
        cos = np.cos(rz)
        sin = np.sin(rz)
        
        T_obj = np.eye(4)
        T_obj[:3,3] = np.array([origin[1], -origin[2], origin[0]])
        T_obj[:3,:3] = np.array([[cos, 0, sin],
                                    [0, 1, 0],
                                    [-sin, 0, cos]])
        
        bbox3D = BoundingBox()
        bbox3D.center = np.copy(T_obj[:3,3])
        bbox3D.R = np.copy(T_obj[:3,:3])
        bbox3D.extent = scale
        T_obj[:3,:3] *= np.max(bbox3D.extent)/2
        
        return T_obj, bbox3D
    
    def subcategorize_objects(self):
        cls_id_add = 100
        self.chamfer_dict = {}
        self.chamfer_opposite_dict = {}
        self.id_representative_dict = {}
        while self.bbox3d_dict:
            for cls_id in self.bbox3d_dict.copy().keys():
                self.chamfer_dict[cls_id] = {}
                self.chamfer_opposite_dict[cls_id] = {}
                obj_ids = list(self.bbox3d_dict[cls_id].keys())
                counts = [self.count_dict[cls_id][obj_id] for obj_id in self.count_dict[cls_id].keys()]
                if len(counts) > 1:
                    counts = np.array(counts)
                    idx_representative = np.argmax(counts)
                else:
                    idx_representative = 0
                
                inst_dict_cls = self.inst_dict[cls_id]   
                obj_id_representative = obj_ids[idx_representative]
                self.id_representative_dict[cls_id] = obj_id_representative
                inst_pcs_template = inst_dict_cls[obj_id_representative]['pcs']
                T_template = inst_dict_cls[obj_id_representative]['T_obj']
                scale_template = np.linalg.det(T_template[:3, :3]) ** (1/3)
                        
                other_obj_ids = []
                for idx in range(len(obj_ids)):
                    if idx != idx_representative:
                        obj_id = obj_ids[idx]
                        other_obj_ids.append(obj_id)

                if len(other_obj_ids) == 0:
                    self.bbox3d_dict.pop(cls_id) 
                    continue
                
                for idx in range(len(other_obj_ids)):
                    obj_id = other_obj_ids[idx]
                    T_source = inst_dict_cls[obj_id]['T_obj']
                    T_rel = T_template @ np.linalg.inv(T_source)
                    
                    inst_pcs = inst_dict_cls[obj_id]['pcs']
                    source_np_w = np.array(inst_pcs.points)
                    scale_source = np.max(source_np_w.max(axis=0)-source_np_w.min(axis=0))/2 # 0720
                    source_transformed = transform_pointcloud(source_np_w, T_rel)
                    inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                    chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source
                    
                    # consider as other subcategory if observed point cloud is note aligned well
                    self.chamfer_dict[cls_id][obj_id] = chamfer_unidir
                    if chamfer_unidir > 0.15: # TODO: do we need different threshold for scannet?
                        subcategorize = True
                    elif chamfer_unidir < 0.06:
                        subcategorize = False
                    else:
                        chamfer_opposite = np.asarray(inst_pcs_template.compute_point_cloud_distance(inst_pcs_transformed)).mean()/scale_template
                        self.chamfer_opposite_dict[cls_id][obj_id] = chamfer_opposite
                        if chamfer_opposite > 0.28:
                            subcategorize = True
                        else:
                            subcategorize = False
                    
                    if subcategorize:
                        cls_id_sub = cls_id + cls_id_add
                        inst_dict = inst_dict_cls[obj_id]
                        metric = self.count_dict[cls_id][obj_id]
                        bbox3d = self.bbox3d_dict[cls_id][obj_id]
                        
                        if not cls_id_sub in self.inst_dict.keys():
                            self.inst_dict[cls_id_sub] = {}
                        self.inst_dict[cls_id_sub].update({obj_id: inst_dict})
                        if not cls_id_sub in self.count_dict.keys():
                            self.count_dict[cls_id_sub] = {}
                        self.count_dict[cls_id_sub].update({obj_id: metric})
                        if not cls_id_sub in self.bbox3d_dict.keys():
                            self.bbox3d_dict[cls_id_sub] = {}
                        self.bbox3d_dict[cls_id_sub].update({obj_id: bbox3d})
                        if not cls_id_sub in self.pe_dict.keys():
                            self.pe_dict[cls_id_sub] = {}
                        self.pe_dict[cls_id_sub].update({obj_id: self.pe_dict[cls_id][obj_id]})
                        if not cls_id_sub in self.fc_occ_map_dict.keys():
                            self.fc_occ_map_dict[cls_id_sub] = {}
                        self.fc_occ_map_dict[cls_id_sub].update({obj_id: self.fc_occ_map_dict[cls_id][obj_id]})
                        
                        inst_dict_cls.pop(obj_id, None)
                        self.count_dict[cls_id].pop(obj_id, None)
                        self.bbox3d_dict[cls_id].pop(obj_id, None)
                        self.pe_dict[cls_id].pop(obj_id, None)
                        self.fc_occ_map_dict[cls_id].pop(obj_id, None)
                
                self.bbox3d_dict.pop(cls_id)
    
    def subcategorize_additional_objects(self, previous_obj_ids=None):
        if not hasattr(self, 'chamfer_dict'):
            self.chamfer_dict = {}
        if not hasattr(self, 'chamfer_opposite_dict'):
            self.chamfer_opposite_dict = {}
        if not hasattr(self, 'id_representative_dict'):
            self.id_representative_dict = {}
            
        new_objects = []
        bbox3d_dict_copy = self.bbox3d_dict.copy()
        for cls_id in bbox3d_dict_copy.keys():
            for obj_id in bbox3d_dict_copy[cls_id].keys():
                if obj_id not in previous_obj_ids:
                    new_objects.append(obj_id)
        
        if len(new_objects) == 0:
            return
        
        for obj_id in new_objects: # obj_id - each added object / inst_dict_before[obj_id] : dict contains the objects info
            # find current subcat which include the added object
            cls_id_before = None
            for cls_id in self.inst_dict.keys():
                if obj_id in self.inst_dict[cls_id].keys():
                    cls_id_before = cls_id
                    break
            
            assert cls_id_before is not None

            inst_dict_before = self.inst_dict[cls_id_before]
            inst_dict = inst_dict_before[obj_id].copy()
            T_source = inst_dict['T_obj']
            inst_pcs = inst_dict['pcs']
            source_np_w = np.array(inst_pcs.points)
            scale_source = np.max(source_np_w.max(axis=0)-source_np_w.min(axis=0))/2 # 0720
            
            for cls_id in bbox3d_dict_copy.keys(): # determine its subcategory using previous subcategories
                if cls_id > 10000: # for newly added 
                    continue
                if cls_id not in self.chamfer_dict.keys():
                    self.chamfer_dict[cls_id] = {}
                if cls_id not in self.chamfer_opposite_dict.keys():
                    self.chamfer_opposite_dict[cls_id] = {}
                
                obj_ids_count = [obj_id_count for obj_id_count in list(bbox3d_dict_copy[cls_id].keys())]# if obj_id_count in previous_obj_ids] # do not include newly added object for representative selection / if you want, change bbox3d_dict_copy -> self.bbox3d_dict
                counts = [self.count_dict[cls_id][obj_id_count] for obj_id_count in obj_ids_count]
                if len(counts) > 1:
                    counts = np.array(counts)
                    idx_representative = np.argmax(counts)
                else:
                    idx_representative = 0
                
                inst_dict_cls = self.inst_dict[cls_id]   
                obj_id_representative = obj_ids_count[idx_representative]
                self.id_representative_dict[cls_id] = obj_id_representative
                inst_pcs_template = inst_dict_cls[obj_id_representative]['pcs']
                T_template = inst_dict_cls[obj_id_representative]['T_obj']
                T_rel = T_template @ np.linalg.inv(T_source)
                scale_template = np.linalg.det(T_template[:3, :3]) ** (1/3)
            
                source_transformed = transform_pointcloud(source_np_w, T_rel)
                inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                if self.count_dict[cls_id_before][obj_id] > counts[idx_representative]:
                    chamfer_unidir = np.asarray(inst_pcs_template.compute_point_cloud_distance(inst_pcs_transformed)).mean()/scale_template
                else:
                    chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source
                
                # consider as other subcategory if observed point cloud is note aligned well
                self.chamfer_dict[cls_id][obj_id] = chamfer_unidir
                if chamfer_unidir > 0.15: # TODO: do we need different threshold for scannet?
                    subcategorize = True
                elif chamfer_unidir < 0.06:
                    subcategorize = False
                else:
                    if self.count_dict[cls_id_before][obj_id] > counts[idx_representative]:
                        chamfer_opposite = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source
                    else:
                        chamfer_opposite = np.asarray(inst_pcs_template.compute_point_cloud_distance(inst_pcs_transformed)).mean()/scale_template
                    self.chamfer_opposite_dict[cls_id][obj_id] = chamfer_opposite
                    if chamfer_opposite > 0.12:
                        subcategorize = True
                    else:
                        subcategorize = False
            
                if not subcategorize: # cls_id is the subcategory of the added object
                    count_info = self.count_dict[cls_id_before][obj_id]
                    pe_info = copy.deepcopy(self.pe_dict[cls_id_before][obj_id])
                    fc_occ_map_info = copy.deepcopy(self.fc_occ_map_dict[cls_id_before][obj_id])
                    
                    inst_dict_before.pop(obj_id, None)
                    self.count_dict[cls_id_before].pop(obj_id, None)
                    self.bbox3d_dict[cls_id_before].pop(obj_id, None)
                    self.pe_dict[cls_id_before].pop(obj_id, None)
                    self.fc_occ_map_dict[cls_id_before].pop(obj_id, None)
                    if len(self.inst_dict[cls_id_before].keys()) == 0:
                        self.inst_dict.pop(cls_id_before)
                    if len(self.count_dict[cls_id_before].keys()) == 0:
                        self.count_dict.pop(cls_id_before)
                    if len(self.bbox3d_dict[cls_id_before].keys()) == 0:
                        self.bbox3d_dict.pop(cls_id_before)
                    if len(self.pe_dict[cls_id_before].keys()) == 0:
                        self.pe_dict.pop(cls_id_before)
                    if len(self.fc_occ_map_dict[cls_id_before].keys()) == 0:
                        self.fc_occ_map_dict.pop(cls_id_before)
                    
                    inst_dict_cls[obj_id] = inst_dict
                    self.count_dict[cls_id][obj_id] = count_info
                    self.pe_dict[cls_id][obj_id] = pe_info
                    self.fc_occ_map_dict[cls_id][obj_id] = fc_occ_map_info
                    break # no need to search other existing subcategories for newly added object once it is categorized into one subcategory

        # pop except for new subcategories
        for cls_id in self.bbox3d_dict.copy().keys():
            if cls_id // 100 == 0:
                subcat_ids = [cls_id_ for cls_id_ in self.bbox3d_dict.keys() if cls_id_ % 100 == cls_id]
                max_subcat_id = max(subcat_ids)
                pop_max_subcat_id = False
                for obj_id in previous_obj_ids:
                    if obj_id in self.bbox3d_dict[max_subcat_id].keys():
                        pop_max_subcat_id = True
                cls_ids_to_pop = subcat_ids if pop_max_subcat_id else subcat_ids[:-1]
                for cls_id_to_pop in cls_ids_to_pop:
                    self.bbox3d_dict.pop(cls_id_to_pop)

    # for revision
    def add_additional_observation(self, cfg, frame_ids, iteration=None):
        previous_obj_ids = []
        for cls_id in self.inst_dict.keys():
            if cls_id != 0:
                previous_obj_ids += list(self.inst_dict[cls_id].keys())
        self.get_frames(frame_ids)
        self.get_poses_toy()
        if cfg.subcategorize:
            self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained, use_gt_bound=True) # iteration=iteration
            self.subcategorize_additional_objects(previous_obj_ids=previous_obj_ids)
            if len(self.bbox3d_dict.keys()) > 0: # some new objects are not categorized into previous subcategories
                self.subcategorize_objects()

class Replica(BaseDataSet):
    def __init__(self, cfg, train_mode=True):
        self.codenerf = cfg.codenerf
        self.name = "replica"
        self.device = cfg.data_device
        self.root_dir = cfg.dataset_dir
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])

        self.W = cfg.W
        self.H = cfg.H
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.edge = cfg.mw
        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        self.K = np.eye(3)
        self.K[0,0] = self.fx
        self.K[1,1] = self.fy
        self.K[0,2] = self.cx
        self.K[1,2] = self.cy
        
        # background semantic classes: undefined--1, undefined-0 beam-5 blinds-12 curtain-30 ceiling-31 door-37 floor-40 picture-59 pillar-60 vent-92 wall-93 wall-plug-95 window-97 rug-98
        self.background_cls_list = [5,12,30,31,40,60,92,93,95,97,98,79]
        # Not sure: door-37 handrail-43 lamp-47 pipe-62 rack-66 shower-stall-73 stair-77 switch-79 wall-cabinet-94 picture-59
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2
        
        self.n_img = len(os.listdir(os.path.join(self.root_dir, "depth")))
        
        scene_mesh = trimesh.load(cfg.scene_mesh_file)
        self.scene_mesh = trimesh_to_open3d(scene_mesh)  
        self.get_all_frames()
        
        if train_mode:
            self.get_all_poses(align_poses=cfg.align_poses)
            
            self.representative_metric = cfg.representative_metric
            self.subcategorize = cfg.subcategorize
            if cfg.align_poses:
                self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained)
                self.align_poses(eta1=cfg.eta1, eta2=cfg.eta2, eta3=cfg.eta3)
            else:
                self.select_poses()
            
            self.metric_dict = {}
            self.phi = None
            self.theta = None
            if cfg.uncertainty_guided_sampling:
                self.get_uncertainty_fields(cfg, view_uncertainty=True, suppress_unseen=cfg.suppress_unseen)
                # self.get_uncertainty_fields(cfg, pose_aligned=True, suppress_unseen=cfg.suppress_unseen)
            if cfg.use_certain_data and not cfg.use_uncertainty:
                self.calculate_mask_rate()
            if cfg.template_scale:
                self.set_template_scale()
            
            render_trag_file = self.root_dir.strip(self.root_dir.split('/')[-1]) + "01/traj_w_c.txt" #"Datasets/Replica/vmap/room_1/imap/01/traj_w_c.txt"
            self.render_poses = torch.from_numpy(np.loadtxt(render_trag_file, delimiter=" ").reshape([-1, 4, 4]).astype(np.float32)).to(cfg.training_device)
            
            # DEBUG
            self.scene_mesh_file = cfg.scene_mesh_file
            # self.visualize_pcs()
            # self.visualize_coords()
        
    def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        # self.sample_list = []
        self.sample_dict = {}
        self.rgb_list = []
        self.depth_list = []
        self.obj_list = []
        for idx in range(self.n_img):
            rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
            depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
            inst_file = os.path.join(self.root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")
            obj_file = os.path.join(self.root_dir, "semantic_class", "semantic_class_" + str(idx) + ".png")
            
            depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1,0)            
            image = cv2.imread(rgb_file).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1,0,2)
            obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
            inst = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32
            
            bbox_scale = self.bbox_scale
            
            obj_ = np.zeros_like(obj)
            cls_list = []
            inst_list = []
            batch_masks = []

            for inst_id in np.unique(inst):
                inst_mask = inst == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                sem_cls = np.unique(obj[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] == 1
                sem_cls = sem_cls[0]
                if sem_cls in self.background_cls_list:
                    continue
                obj_mask = inst == inst_id
                batch_masks.append(obj_mask)
                if sem_cls == 0 and inst_id != 0: # undefined class: 0 in data, -1 in our system
                    cls_list.append(inst_id+1000)
                else:
                    cls_list.append(sem_cls)
                inst_list.append(inst_id)
                            
            if len(batch_masks) > 0:
                batch_masks = torch.from_numpy(np.stack(batch_masks))
                cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)

                for i in range(batch_masks.shape[0]):
                    w = rmaxs[i] - rmins[i]
                    h = cmaxs[i] - cmins[i]
                    if w <= 10 or h <= 10:  # too small   todo
                        continue
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                 w=obj.shape[1], h=obj.shape[0])
                    sem_cls = cls_list[i]
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    # bbox_dict.update({inst_id: torch.from_numpy(np.array(
                    #     [bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})  # bbox order
                    
                    if not sem_cls in self.inst_dict.keys():
                        self.inst_dict[sem_cls] = {}
                    bbox = torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))
                    if not inst_id in self.inst_dict[sem_cls].keys():
                        self.inst_dict[sem_cls][inst_id] = {'frame_info': [{'frame': idx, 'bbox': bbox}]}
                    else:
                        self.inst_dict[sem_cls][inst_id]['frame_info'].append({'frame': idx, 'bbox': bbox})

            inst[obj_ == 0] = 0  # for background
            # for i, inst_id in enumerate(inst_list):
            #     sem_cls = cls_list[i]
            #     if sem_cls in self.inst_dict.keys():
            #         if inst_id in self.inst_dict[sem_cls].keys():
            #             self.inst_dict[sem_cls][inst_id]['frame_info'][-1]['obj_mask'] =  inst
            
            # bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # bbox order
            if idx == 0:
                self.inst_dict[0] = {'frame_info': []}
            background_mask = inst # or obj_, maybe both OK
            self.inst_dict[0]['frame_info'].append({'frame': idx, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))})
            # self.inst_dict[0]['frame_info'].append({'frame': idx, 'background_mask': background_mask, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))})

            T = self.Twc[idx]   # could change to ORB-SLAM pose or else    

            if self.depth_transform:
                depth = self.depth_transform(depth) 
            
            # sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
            #         "obj": obj, "bbox_dict": bbox_dict, "frame_id": idx}
            sample = {"image": image, "depth": depth, "obj_mask": inst, "T": T, "frame_id": idx} #"sem_mask": obj, 
                             
            if image is None or depth is None:
                print(rgb_file)
                print(depth_file)
                raise ValueError
            
            # self.sample_list.append(sample)
            self.sample_dict[idx] = sample  
            
            self.rgb_list.append(image)
            self.depth_list.append(depth)
            self.obj_list.append(inst)
        
        self.rgb_list = torch.from_numpy(np.stack(self.rgb_list, axis=0))
        self.depth_list = torch.from_numpy(np.stack(self.depth_list, axis=0))
        self.obj_list = torch.from_numpy(np.stack(self.obj_list, axis=0))
        
        t2 = time.time()
        print('get_all_frames takes {} seconds'.format(t2-t1))               
                    
    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        # return self.sample_list[idx]
        return self.sample_dict[idx]

class ScanNet(BaseDataSet):
    def __init__(self, cfg, debug_dir=None):
        self.representative_metric = cfg.representative_metric
        self.subcategorize = cfg.subcategorize
        self.name = "scannet"
        self.device = cfg.data_device
        self.root_dir = cfg.dataset_dir
        self.color_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.inst_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'instance-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        self.sem_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))  # label-filt
        self.load_poses(os.path.join(self.root_dir, 'pose'))
        self.n_img = len(self.color_paths)
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])
        self.max_depth = cfg.max_depth
        self.depth_scale = cfg.depth_scale
        self.W = cfg.W
        self.H = cfg.H
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.edge = cfg.mw
        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        self.K = np.eye(3)
        self.K[0,0] = self.fx
        self.K[1,1] = self.fy
        self.K[0,2] = self.cx
        self.K[1,2] = self.cy

        self.min_pixels = 1500
        # from scannetv2-labels.combined.tsv
        #1-wall, 3-floor, 16-window, 41-ceiling, 232-light switch   0-unknown? 21-pillar 161-doorframe, shower walls-128, curtain-21, windowsill-141
        self.background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        self.bbox_scale = 0.2
        self.inst_filter_dict = {}
        self.inst_dict = {}
        
        self.use_refined_mask = cfg.use_refined_mask
        self.debug_dir = debug_dir
        self.get_all_frames()
        
        self.get_all_poses(align_poses=cfg.align_poses)
        
        self.scene_mesh_file = cfg.scene_mesh_file
        if cfg.align_poses:
            self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained)
            self.align_poses()
        else:
            self.select_poses()
            
        self.metric_dict = {}
        self.phi = None
        self.theta = None
        if cfg.uncertainty_guided_sampling:
            self.get_uncertainty_fields(cfg, view_uncertainty=True, suppress_unseen=cfg.suppress_unseen)
            # self.get_uncertainty_fields(cfg, pose_aligned=True, suppress_unseen=cfg.suppress_unseen)
        if cfg.use_certain_data and not cfg.use_uncertainty:
            self.calculate_mask_rate()
        if cfg.template_scale:
            self.set_template_scale()
            
        # # DEBUG
        # self.visualize_coords()

    def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        # self.sample_list = []
        self.sample_dict = {}
        self.rgb_list = []
        self.depth_list = []
        self.obj_list = []
        reduce = 0
        for index in range(self.n_img):
            index_reduced = index - reduce
            bbox_scale = self.bbox_scale
            color_path = self.color_paths[index]
            depth_path = self.depth_paths[index]
            inst_path = self.inst_paths[index]
            sem_path = self.sem_paths[index]
            color_data = cv2.imread(color_path).astype(np.uint8)
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_data = np.nan_to_num(depth_data, nan=0.)
            T = None
            if self.poses is not None:
                T = self.poses[index]
                if np.any(np.isinf(T)):
                    print("pose inf!")
                    reduce += 1
                    continue          

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
            inst_data = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
            inst_data = cv2.resize(inst_data, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
            sem_data = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)#.astype(np.int32)
            sem_data = cv2.resize(sem_data, (W, H), interpolation=cv2.INTER_NEAREST)
            
            if True:
                viz_sem_dir = os.path.join(self.root_dir, 'viz-label-filt')
                os.makedirs(viz_sem_dir, exist_ok=True)
                viz_sem_file = os.path.join(viz_sem_dir, f'{index}.png')
                sem_data_viz = np.zeros_like(color_data)
                for sem_id in np.unique(sem_data):
                    sem_mask = sem_data == sem_id
                    sem_data_viz[sem_mask] = imgviz.label_colormap()[sem_id % 256]
                cv2.imwrite(viz_sem_file, sem_data_viz)
            
            if self.edge:
                edge = self.edge # crop image edge, there are invalid value on the edge of the color image
                color_data = color_data[edge:-edge, edge:-edge]
                depth_data = depth_data[edge:-edge, edge:-edge]
            if self.depth_transform:
                depth_data = self.depth_transform(depth_data)
        
            if self.edge:
                edge = self.edge
                inst_data = inst_data[edge:-edge, edge:-edge]
                sem_data = sem_data[edge:-edge, edge:-edge]            
            inst_data += 1  # shift from 0->1 , 0 is for background

            cls_list = []
            inst_list = []
            batch_masks = []
            
            inst_to_cls = {0:0}
            for inst_id in np.unique(inst_data):
                inst_mask = inst_data == inst_id
                sem_cls = np.unique(sem_data[inst_mask])  # sem label, only interested obj
                assert sem_cls.shape[0] == 1
                sem_cls = sem_cls[0]
                if sem_cls in self.background_cls_list:
                    inst_data[inst_mask] = 0
                    continue
                batch_masks.append(inst_mask)
                cls_list.append(sem_cls)
                inst_list.append(inst_id)
                inst_to_cls[inst_id] = sem_cls
            T_CW = np.linalg.inv(T)
            # inst_data = box_filter(batch_masks, inst_list, depth_data, self.inst_filter_dict, self.intrinsic_open3d, T_CW, min_pixels=self.min_pixels)
            
            # semantically refine depth segmentation
            normal, geometry_label, segment_masks, segments = geometry_segmentation(color_data, depth_data, self.intrinsic_open3d, debug_dir=self.debug_dir, frame=index)
            inst_data_refined = refine_inst_data(inst_data, segment_masks, segments, debug_dir=self.debug_dir, frame=index)
            if self.use_refined_mask:
                inst_data = inst_data_refined
            
            refined_obj_ids = np.unique(inst_data)
            for obj_id in refined_obj_ids:
                mask = inst_data == obj_id
                bbox2d = get_bbox2d(mask, bbox_scale=bbox_scale)
                if bbox2d is None:
                    inst_data[mask] = 0 # set to bg
                else:
                    min_x, min_y, max_x, max_y = bbox2d
                    sem_cls = inst_to_cls[obj_id]
                    if not sem_cls in self.inst_dict.keys():
                        self.inst_dict[sem_cls] = {}
                    bbox = torch.from_numpy(np.array([min_x, max_x, min_y, max_y]))
                    if not obj_id in self.inst_dict[sem_cls].keys():
                        self.inst_dict[sem_cls][obj_id] = {'frame_info': [{'frame': index_reduced, 'bbox': bbox}]}
                    else:
                        self.inst_dict[sem_cls][obj_id]['frame_info'].append({'frame': index_reduced, 'bbox': bbox})

            # accumulate pointcloud for foreground objects
            obj_ids_refined = np.unique(inst_data_refined)
            for obj_id in obj_ids_refined:
                if obj_id == 0:
                    continue
                depth_data_copy = depth_data.copy()
                mask = inst_data_refined == obj_id
                depth_data_copy[~mask] = 0.
                inst_pc = unproject_pointcloud(depth_data_copy, self.intrinsic_open3d, T_CW)
                
                i = inst_list.index(obj_id)
                sem_cls = cls_list[i]
                if 'pcs' not in self.inst_dict[sem_cls][obj_id].keys():
                    self.inst_dict[sem_cls][obj_id]['pcs'] = inst_pc
                else:
                    self.inst_dict[sem_cls][obj_id]['pcs'] += inst_pc
            
            if index_reduced == 0:
                self.inst_dict[0] = {'frame_info': []}
            background_mask = inst_data.transpose(1,0) # or obj_, maybe both OK
            self.inst_dict[0]['frame_info'].append({'frame': index_reduced, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))}) 
            # self.inst_dict[0]['frame_info'].append({'frame': index_reduced, 'background_mask': background_mask, 'bbox': torch.from_numpy(np.array([int(0), int(background_mask.shape[0]), 0, int(background_mask.shape[1])]))}) 

            # sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
            #         "obj": obj, "bbox_dict": bbox_dict, "frame_id": idx}
            sample = {"image": color_data.transpose(1,0,2), "depth": depth_data.transpose(1,0), "obj_mask": inst_data.transpose(1,0), "T": T, "frame_id": index_reduced}#"sem_mask": sem_data, 
                  
            if color_data is None or depth_data is None:
                print(color_path)
                print(depth_path)
                raise ValueError
            
            # self.sample_list.append(sample)
            self.sample_dict[index_reduced] = sample    
            
            self.rgb_list.append(color_data.transpose(1,0,2))
            self.depth_list.append(depth_data.transpose(1,0))
            self.obj_list.append(inst_data.transpose(1,0))
        
        self.rgb_list = torch.from_numpy(np.stack(self.rgb_list, axis=0))
        self.depth_list = torch.from_numpy(np.stack(self.depth_list, axis=0))
        self.obj_list = torch.from_numpy(np.stack(self.obj_list, axis=0))
        self.n_img -= reduce
            
        t2 = time.time()
        print('get_all_frames takes {} seconds'.format(t2-t1))
           
    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            self.poses.append(c2w)

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        # return self.sample_list[idx]
        return self.sample_dict[index]

# # for revision, from ESLAM
# class TUM_RGBD(BaseDataSet):
#     def __init__(self, cfg):
#         # super(TUM_RGBD, self).__init__(cfg, args, scale, device)
#         self.name = 'tumrgbd'
#         self.device = cfg.data_device
#         self.scale = 1
#         self.png_depth_scale = 5000.0

#         self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg.H, cfg.W, cfg.fx, cfg.fy, cfg.cx, cfg.cy
#         self.K = np.eye(3)
#         self.K[0,0] = self.fx
#         self.K[1,1] = self.fy
#         self.K[0,2] = self.cx
#         self.K[1,2] = self.cy
        
#         self.distortion = np.array(cfg.distortion) if cfg.distortion is not None else None
#         self.crop_size = cfg.crop_size if cfg.crop_size is not None else None

#         self.input_folder = cfg.dataset_dir

#         self.crop_edge = cfg.mw
        
#         self.color_paths, self.depth_paths, self.poses = self.loadtum(
#             self.input_folder, frame_rate=32)
#         self.inst_paths = [os.path.join(self.input_folder, "semantic_instance", os.path.basename(x)) for x in self.color_paths]
#         self.sem_paths = [os.path.join(self.input_folder, "semantic_class", os.path.basename(x)) for x in self.color_paths]
#         self.n_img = len(self.color_paths)
        
#         self.depth_transform = transforms.Compose(
#             [image_transforms.DepthScale(1/self.png_depth_scale),
#              image_transforms.DepthFilter(cfg.max_depth)])

#     def parse_list(self, filepath, skiprows=0):
#         """ read list data """
#         data = np.loadtxt(filepath, delimiter=' ',
#                           dtype=np.unicode_, skiprows=skiprows)
#         return data

#     def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
#         """ pair images, depths, and poses """
#         associations = []
#         for i, t in enumerate(tstamp_image):
#             if tstamp_pose is None:
#                 j = np.argmin(np.abs(tstamp_depth - t))
#                 if (np.abs(tstamp_depth[j] - t) < max_dt):
#                     associations.append((i, j))

#             else:
#                 j = np.argmin(np.abs(tstamp_depth - t))
#                 k = np.argmin(np.abs(tstamp_pose - t))

#                 if (np.abs(tstamp_depth[j] - t) < max_dt) and \
#                         (np.abs(tstamp_pose[k] - t) < max_dt):
#                     associations.append((i, j, k))

#         return associations

#     def loadtum(self, datapath, frame_rate=-1):
#         """ read video data in tum-rgbd format """
#         if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
#             pose_list = os.path.join(datapath, 'groundtruth.txt')
#         elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
#             pose_list = os.path.join(datapath, 'pose.txt')

#         image_list = os.path.join(datapath, 'rgb.txt')
#         depth_list = os.path.join(datapath, 'depth.txt')

#         image_data = self.parse_list(image_list)
#         depth_data = self.parse_list(depth_list)
#         pose_data = self.parse_list(pose_list, skiprows=1)
#         pose_vecs = pose_data[:, 1:].astype(np.float64)

#         tstamp_image = image_data[:, 0].astype(np.float64)
#         tstamp_depth = depth_data[:, 0].astype(np.float64)
#         tstamp_pose = pose_data[:, 0].astype(np.float64)
#         associations = self.associate_frames(
#             tstamp_image, tstamp_depth, tstamp_pose)

#         indicies = [0]
#         for i in range(1, len(associations)):
#             t0 = tstamp_image[associations[indicies[-1]][0]]
#             t1 = tstamp_image[associations[i][0]]
#             if t1 - t0 > 1.0 / frame_rate:
#                 indicies += [i]

#         images, poses, depths, intrinsics = [], [], [], []
#         inv_pose = None
#         for ix in indicies:
#             (i, j, k) = associations[ix]
#             images += [os.path.join(datapath, image_data[i, 1])]
#             depths += [os.path.join(datapath, depth_data[j, 1])]
#             c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
#             if inv_pose is None:
#                 inv_pose = np.linalg.inv(c2w)
#                 c2w = np.eye(4)
#             else:
#                 c2w = inv_pose@c2w
#             c2w[:3, 1] *= -1
#             c2w[:3, 2] *= -1
#             c2w = torch.from_numpy(c2w).float()
#             poses += [c2w]

#         return images, depths, poses

#     def pose_matrix_from_quaternion(self, pvec):
#         """ convert 4x4 pose matrix to (t, q) """
#         from scipy.spatial.transform import Rotation

#         pose = np.eye(4)
#         pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
#         pose[:3, 3] = pvec[:3]
#         return pose

#     def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        self.sample_dict = {}
        self.rgb_list = []
        self.depth_list = []
        self.obj_list = []
        for index in range(self.n_img):
            color_path = self.color_paths[index]
            depth_path = self.depth_paths[index]
            pose = self.poses[index]
            color_data = cv2.imread(color_path).astype(np.uint8)
            if self.distortion is not None:
                # undistortion is only applied on color image, not depth!
                color_data = cv2.undistort(color_data, self.K, self.distortion)
            
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_data = np.nan_to_num(depth_data, nan=0.)
            T = pose
            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
            if self.depth_transform:
                depth_data = self.depth_transform(depth_data)
                
            if self.crop_size is not None:
                # follow the pre-processing step in lietorch, actually is resize
                color_data = color_data.permute(2, 0, 1)
                color_data = F.interpolate(
                    color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
                depth_data = F.interpolate(
                    depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
                color_data = color_data.permute(1, 2, 0).contiguous()
            
            inst_data = np.zeros_like(depth_data)
            sem_data = np.zeros_like(depth_data)
            if self.crop_edge:
                edge = self.crop_edge
                color_data = color_data[edge:-edge, edge:-edge]
                depth_data = depth_data[edge:-edge, edge:-edge]
            if self.depth_transform:
                depth_data = self.depth_transform(depth_data)
            if self.crop_edge:
                edge = self.crop_edge
                inst_data = inst_data[edge:-edge, edge:-edge]
                sem_data = sem_data[edge:-edge, edge:-edge]
    
# DEBUG
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    # parser.add_argument('--config', default='./configs/ScanNet/config_scannet0000_vMAP.json', type=str)
    parser.add_argument('--config', default='./configs/Replica/config_replica_room0_vMAP.json', type=str)
    args = parser.parse_args()
    
    config_file = args.config
    cfg = Config(config_file) 
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg)
    print('hi')