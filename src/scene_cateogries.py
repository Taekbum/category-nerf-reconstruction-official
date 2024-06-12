import numpy as np
import torch
from torch.autograd import Variable
from utils import get_tensor_from_transform_sim3, get_transform_from_tensor_sim3, load_scene_bound, importance_sampling_coords, plot_selected
import trainer
import argparse
from cfg import Config
import copy
import os
from time import perf_counter_ns
# from torch.distributions.multivariate_normal import MultivariateNormal

class performance_measure:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, type, value, tb):
        self.end_time = perf_counter_ns()
        self.exec_time = self.end_time - self.start_time

        print(f"{self.name} excution time: {(self.exec_time)/1000000:.2f} ms")

def origin_dirs_O(T_CO, dirs_C):
    
    assert T_CO.shape[0] == dirs_C.shape[0]
    assert T_CO.shape[1:] == (4, 4)
    assert dirs_C.shape[1] == 3

    T_OC = torch.linalg.inv(T_CO)
    dirs_O = (T_OC[:, :3, :3] @ dirs_C[..., None]).squeeze()

    origins = T_OC[:, :3, -1]

    return origins, dirs_O

def origin_dirs_W(T_WC, dirs_C):

    assert T_WC.shape[0] == dirs_C.shape[0]
    assert T_WC.shape[1:] == (4, 4)
    assert dirs_C.shape[1] == 3

    dirs_W = (T_WC[:, :3, :3] @ dirs_C[..., None]).squeeze()

    origins = T_WC[:, :3, -1]

    return origins, dirs_W


# @torch.jit.script
def stratified_bins(min_depth, max_depth, n_bins, n_rays, type=torch.float32, device = "cuda:0", z_fixed = False):
    # type: (Tensor, Tensor, int, int) -> Tensor

    bin_limits_scale = torch.linspace(0, 1, n_bins+1, dtype=type, device=device)

    if not torch.is_tensor(min_depth):
        min_depth = torch.ones(n_rays, dtype=type, device=device) * min_depth
    
    if not torch.is_tensor(max_depth):
        max_depth = torch.ones(n_rays, dtype=type, device=device) * max_depth

    depth_range = max_depth - min_depth
  
    lower_limits_scale = depth_range[..., None] * bin_limits_scale + min_depth[..., None]
    lower_limits_scale = lower_limits_scale[:, :-1]

    assert lower_limits_scale.shape == (n_rays, n_bins)

    bin_length_scale = depth_range / n_bins
    # if z_fixed:
    #     z_vals_scale = lower_limits_scale
    # else:
    increments_scale = torch.rand(
        n_rays, n_bins, device=device,
        dtype=torch.float32) * bin_length_scale[..., None]

    z_vals_scale = lower_limits_scale + increments_scale

    assert z_vals_scale.shape == (n_rays, n_bins)

    return z_vals_scale

# @torch.jit.script
def normal_bins_sampling(depth, n_bins, n_rays, delta, device = "cuda:0"):
    # type: (Tensor, int, int, float) -> Tensor

    # device = "cpu"
    # bins = torch.normal(0.0, delta / 3., size=[n_rays, n_bins], devi
        # self.keyframes_batch = torch.empty(self.n_keyframes,ce=device).sort().values
    bins = torch.empty(n_rays, n_bins, dtype=torch.float32, device=device).normal_(mean=0.,std=delta / 3.).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth[:, None] + bins

    assert z_vals.shape == (n_rays, n_bins)

    return z_vals



class sceneCategory():
    """
    shared MLP + instance-specific code for instance mapping
    single batch contain all instances in this class
    optimize (MLP, code, Tco)
    """
    
    def __init__(self, cfg, cls_id, inst_dict, sample_dict, cached_rays_dir, id_representative=0, train_mode=True, scale_template=None, adaptive_obj_num=False) -> None:
        self.adaptive_obj_num = adaptive_obj_num
        self.object_wise_model = cfg.object_wise_model
        self.uniform_region = cfg.uniform_region
        self.smooth = cfg.smooth
        self.use_certain_data = cfg.use_certain_data
        self.select_minimum = cfg.select_minimum
        
        self.no_shuffle = cfg.no_shuffle
        self.use_nocs = cfg.use_nocs
        self.n_per_optim = cfg.n_per_optim
        self.cls_id = cls_id
        if cls_id != 0:
            self.obj_ids = list(inst_dict.keys())
        else:
            self.obj_ids = [0]
        self.data_device = cfg.data_device
        self.training_device = cfg.training_device
        
        if cls_id == 0: # background
            self.obj_scale = cfg.bg_scale
            self.hidden_feature_size = cfg.hidden_feature_size_bg
            self.n_bins_cam2surface = cfg.n_bins_cam2surface_bg
        else:
            self.obj_scale = cfg.obj_scale
            self.hidden_feature_size = cfg.hidden_feature_size
            self.n_bins_cam2surface = cfg.n_bins_cam2surface
        
        if train_mode:
            init_idx = list(sample_dict.keys())[0]
            rgb_init = sample_dict[init_idx]["image"]
            self.frames_width = rgb_init.shape[0]
            self.frames_height = rgb_init.shape[1]
        
        self.min_bound = cfg.min_depth
        self.max_bound = cfg.max_depth
        self.n_bins = cfg.n_bins
        
        self.surface_eps = cfg.surface_eps
        self.stop_eps = cfg.stop_eps
            
        # RGB + pixel state batch
        self.rgb_idx = slice(0, 3)
        self.state_idx = slice(3, 4)
        
        # Pixel states: (except background)
        self.other_obj = 0  # pixel belong to other obj
        self.this_obj = 1  # pixel belong to obj 
        self.unknown_obj = 2  # pixel state is unknown
        
        self.use_equal_batch = cfg.use_equal_batch
        self.rgbs_batch_dict = {}
        self.depth_batch_dict = {}
        self.t_wc_batch_dict = {}
        self.frame_batch_dict = {}
        self.ray_dirs_batch_dict = {}
        self.i_batch_dict = {} 
        if not self.use_equal_batch:
            self.rgbs_batch_all = []
            self.depth_batch_all = []
            self.t_co_batch_all = []
            self.t_wc_batch_all = []
            self.ray_dirs_batch_all = []
            self.batch_indices_all = []
        if len(self.obj_ids)==1 and cfg.use_zero_code:
            inst_id_to_index = {self.obj_ids[0]: 0}
        else:
            inst_id_to_index = {inst_id: self.obj_ids.index(inst_id) for inst_id in self.obj_ids}
        
        self.use_zero_code = cfg.use_zero_code
        self.use_mean_code = cfg.use_mean_code
        
        if train_mode:        
            if cls_id != 0:
                self.extent_dict = {}
                self.object_tensor_dict = {}
                # for uncertainty-guided sampling
                self.indices_batch = []
                self.ids_batch = []
                self.frames_batch = []
                self.frame_ids_batch = []
                self.cam_origins = []
                self.uncertainty_batch = []
                # self.obj_origins = []
                # self.cam_z = []
                self.bbox_2d_dict = {}
                for inst_id in inst_dict.keys():
                    inst_idx = inst_id_to_index[inst_id]
                    inst_info = inst_dict[inst_id]
                    
                    # extent
                    if 'bbox3D' in inst_info.keys():
                        bbox3D = inst_info['bbox3D']
                        extent = bbox3D.extent
                    else:
                        extent = np.array([2.0, 2.0, 2.0])
                    self.extent_dict[inst_id] = extent

                    T_obj = inst_info['T_obj']
                    object_tensor = get_tensor_from_transform_sim3(np.copy(T_obj))
                    object_tensor = object_tensor.to(self.data_device)
                    if not cfg.fix_pose and not cfg.staged_poseoptim and len(self.obj_ids) > 1 and inst_id != id_representative:
                        object_tensor = Variable(object_tensor.to(self.data_device), 
                                                requires_grad=True)
                    self.object_tensor_dict[inst_id] = object_tensor
                    
                    frame_info_list = inst_info['frame_info']
                    if cfg.fix_codenerf and cls_id == 20:
                        import random
                        while True:
                            frame_info = random.choice(frame_info_list)
                            frame = frame_info['frame']
                            sample = sample_dict[frame]
                            obj_mask = sample['obj_mask'] == inst_id
                            if np.sum(obj_mask) > 10000:
                                break
                            else:
                                print(f'too small mask for {inst_id}')
                            
                    frame_buffer_size = len(frame_info_list)
                    
                    t_wc_batch = torch.empty(
                        frame_buffer_size, 4, 4,
                        dtype=torch.float32,
                        device=self.data_device)  # world to camera transform
                    
                    rgbs_batch = []
                    depth_batch = []
                    frame_batch = []
                    ray_dirs_batch = []
                    self.bbox_2d_dict[inst_id] = []
                    # uncertainty_batch = []
                    for idx, frame_info in enumerate(frame_info_list):
                        frame = frame_info['frame']
                        # for uncertainty-guided sampling
                        self.indices_batch.append(inst_idx)
                        self.ids_batch.append(inst_id)
                        self.frames_batch.append(idx)
                        self.frame_ids_batch.append(frame)
                        
                        sample = sample_dict[frame]
                        # rgb = torch.from_numpy(sample['image'])
                        # depth = torch.from_numpy(sample['depth'])
                        T_cam = torch.from_numpy(sample['T']).to(self.data_device)
                        obj_mask = torch.from_numpy(sample['obj_mask'])
                        
                        state = torch.zeros_like(obj_mask, dtype=torch.uint8)
                        state[obj_mask == inst_id] = 1
                        state[obj_mask == -1] = 2
                        
                        bbox_2d = frame_info['bbox']
                        idx_w = slice(bbox_2d[0], bbox_2d[1])
                        idx_h = slice(bbox_2d[2], bbox_2d[3])
                        self.bbox_2d_dict[inst_id].append(bbox_2d)
                        
                        t_wc_batch[idx] = T_cam
                        rgb_box = torch.from_numpy(sample['image'])[idx_w, idx_h, :].reshape(-1,3)#rgb[idx_w, idx_h, :].reshape(-1,3)
                        state_box = state[idx_w, idx_h].reshape(-1)
                        rgbs_box = torch.cat([rgb_box, state_box[:, None]], dim=-1)#.to(self.data_device)
                        depth_box = torch.from_numpy(sample['depth'])[idx_w, idx_h].reshape(-1)#depth[idx_w, idx_h].reshape(-1)#.to(self.data_device)
                        frame_box = torch.from_numpy(np.array([idx])).repeat((bbox_2d[1]-bbox_2d[0]) * (bbox_2d[3]-bbox_2d[2]))#.to(self.data_device)
                        ray_dirs_box = cached_rays_dir[idx_w, idx_h, :].reshape(-1,3)
                        
                        # if 'uncertainty' in frame_info.keys() and len(self.obj_ids) > 1:
                        #     uncertainty_batch.append(frame_info['uncertainty'])
                        
                        rgbs_batch.append(rgbs_box)
                        depth_batch.append(depth_box)
                        frame_batch.append(frame_box)
                        ray_dirs_batch.append(ray_dirs_box)
                    
                    # for uncertainty-guided sampling
                    self.bbox_2d_dict[inst_id] = torch.stack(self.bbox_2d_dict[inst_id])
                    two = torch.from_numpy(T_obj.astype(np.float32)).to(self.data_device)
                    toc_batch = torch.linalg.inv(two)[None, :, :].repeat(t_wc_batch.shape[0], 1, 1) @ t_wc_batch
                    # tco_batch = torch.linalg.inv(t_wc_batch) @ two[None, :, :].repeat(t_wc_batch.shape[0], 1, 1)
                    # self.obj_origins.append(tco_batch[:,:3,3])
                    self.cam_origins.append(toc_batch[:,:3,3])
                    # scale = torch.linalg.det(toc_batch[:,:3,:3])**(1/3)
                    # self.cam_z.append(toc_batch[:,:3,2]/scale.squuze(-1))
                    # if len(self.obj_ids) > 1:
                    #     self.uncertainty_batch.append(torch.Tensor(uncertainty_batch))
                    
                    rgbs_batch = torch.cat(rgbs_batch, 0)
                    depth_batch = torch.cat(depth_batch, 0)
                    frame_batch = torch.cat(frame_batch, 0) # (frame_buffer_size*w*h, )
                    ray_dirs_batch = torch.cat(ray_dirs_batch, 0)

                    if self.use_equal_batch:
                        # shuffle batched data 
                        shuffled_idx = np.arange(rgbs_batch.shape[0])
                        np.random.shuffle(shuffled_idx)
                        rgbs_batch = rgbs_batch[shuffled_idx]
                        depth_batch = depth_batch[shuffled_idx]
                        frame_batch = frame_batch[shuffled_idx]
                        ray_dirs_batch = ray_dirs_batch[shuffled_idx]
                        
                        self.rgbs_batch_dict[inst_id] = rgbs_batch
                        self.depth_batch_dict[inst_id] = depth_batch
                        self.frame_batch_dict[inst_id] = frame_batch
                        self.ray_dirs_batch_dict[inst_id] = ray_dirs_batch
                        self.t_wc_batch_dict[inst_id] = t_wc_batch
                        
                        self.i_batch_dict[inst_id] = 0
                    else:
                        index = inst_id_to_index[inst_id]
                        indices = torch.from_numpy(np.array([index])).repeat(rgbs_batch.shape[0])
                        self.batch_indices_all.append(indices)
                        self.rgbs_batch_all.append(rgbs_batch)
                        self.depth_batch_all.append(depth_batch)
                        self.ray_dirs_batch_all.append(ray_dirs_batch)
                        
                        T_obj_torch = torch.from_numpy(T_obj.astype(np.float32)).to(self.data_device)
                        tco_batch = torch.linalg.inv(t_wc_batch) @ \
                            T_obj_torch[None, :, :].repeat(t_wc_batch.shape[0], 1, 1) # (frame_buffer_size, 4, 4)
                        sampled_tco = tco_batch[frame_batch, :, :]
                        self.t_co_batch_all.append(sampled_tco.cpu())
                        if not adaptive_obj_num and (not self.use_nocs or len(self.obj_ids) == 1):
                            self.t_wc_batch_all.append(t_wc_batch[frame_batch, :, :].cpu())
                
                # for uncertainty-guided sampling
                self.indices_batch = torch.LongTensor(self.indices_batch)
                self.ids_batch = torch.LongTensor(self.ids_batch)
                self.frames_batch = torch.LongTensor(self.frames_batch)
                self.frame_ids_batch = torch.LongTensor(self.frame_ids_batch)
                self.cam_origins = torch.cat(self.cam_origins)
                # self.obj_origins = torch.cat(self.obj_origins)
                # self.cam_z = torch.cat(self.cam_z)
                self.cam_origins = self.cam_origins/torch.linalg.norm(self.cam_origins, dim=-1, keepdim=True)
                # if len(self.obj_ids) > 1:
                #     self.uncertainty_batch = torch.cat(self.uncertainty_batch)
                #     if cfg.large_concent:
                #         self.uncertainty_batch = (self.uncertainty_batch-self.uncertainty_batch.min())/(self.uncertainty_batch.max()-self.uncertainty_batch.min())
                #     if self.smooth:
                #         self.uncertainty_batch = 1 + self.uncertainty_batch
                
                if not self.use_equal_batch:        
                    self.rgbs_batch_all = torch.cat(self.rgbs_batch_all, 0)
                    self.depth_batch_all = torch.cat(self.depth_batch_all, 0)
                    self.ray_dirs_batch_all = torch.cat(self.ray_dirs_batch_all, 0)
                    self.t_co_batch_all = torch.cat(self.t_co_batch_all, 0)
                    self.batch_indices_all = torch.cat(self.batch_indices_all, 0)
                    if not adaptive_obj_num and (not self.use_nocs or len(self.obj_ids) == 1):
                        self.t_wc_batch_all = torch.cat(self.t_wc_batch_all, 0)
                    
                    if not self.no_shuffle:
                        # shuffle batched data 
                        shuffled_idx = np.arange(self.rgbs_batch_all.shape[0])
                        np.random.shuffle(shuffled_idx)
                        self.batch_indices_all = self.batch_indices_all[shuffled_idx]
                        self.rgbs_batch_all = self.rgbs_batch_all[shuffled_idx]
                        self.depth_batch_all = self.depth_batch_all[shuffled_idx]
                        self.ray_dirs_batch_all = self.ray_dirs_batch_all[shuffled_idx]
                        self.t_co_batch_all = self.t_co_batch_all[shuffled_idx]
                        if not adaptive_obj_num and (not self.use_nocs or len(self.obj_ids) == 1):
                            self.t_wc_batch_all = self.t_wc_batch_all[shuffled_idx]
                    self.i_batch = 0

            else:
                frame_info_list = inst_dict['frame_info']
                frame_buffer_size = len(frame_info_list)
                
                t_wc_batch = torch.empty(
                                frame_buffer_size, 4, 4,
                                dtype=torch.float32,
                                device=self.data_device)  # world to camera transform

                rgbs_batch = []
                depth_batch = []
                frame_batch = []
                ray_dirs_batch = []
                for idx, frame_info in enumerate(frame_info_list):
                    frame = frame_info['frame']
                    
                    sample = sample_dict[frame]
                    rgb = torch.from_numpy(sample['image'])
                    depth = torch.from_numpy(sample['depth'])
                    T_cam = torch.from_numpy(sample['T']).to(self.data_device)
                    obj_mask = torch.from_numpy(sample['obj_mask'])
                    
                    state = torch.zeros_like(obj_mask, dtype=torch.uint8)
                    state[obj_mask == 0] = 1
                    state[obj_mask == -1] = 2
                    
                    bbox_2d = frame_info['bbox']
                    idx_w = slice(bbox_2d[0], bbox_2d[1])
                    idx_h = slice(bbox_2d[2], bbox_2d[3])
                    
                    t_wc_batch[idx] = T_cam
                    rgb_box = rgb[idx_w, idx_h, :].reshape(-1,3)
                    state_box = state[idx_w, idx_h].reshape(-1)
                    rgbs_box = torch.cat([rgb_box, state_box[:, None]], dim=-1)
                    depth_box = depth[idx_w, idx_h].reshape(-1)
                    frame_box = torch.from_numpy(np.array([idx])).repeat((bbox_2d[1]-bbox_2d[0]) * (bbox_2d[3]-bbox_2d[2]))
                    ray_dirs_box = cached_rays_dir[idx_w, idx_h, :].reshape(-1,3)
                    
                    rgbs_batch.append(rgbs_box)
                    depth_batch.append(depth_box)
                    frame_batch.append(frame_box)
                    ray_dirs_batch.append(ray_dirs_box)
                
                rgbs_batch = torch.cat(rgbs_batch, 0)
                depth_batch = torch.cat(depth_batch, 0)
                frame_batch = torch.cat(frame_batch, 0) # (frame_buffer_size*w*h, )
                ray_dirs_batch = torch.cat(ray_dirs_batch, 0)
                
                # shuffle batched data 
                shuffled_idx = np.arange(rgbs_batch.shape[0])
                np.random.shuffle(shuffled_idx)
                rgbs_batch = rgbs_batch[shuffled_idx]
                depth_batch = depth_batch[shuffled_idx]
                frame_batch = frame_batch[shuffled_idx]
                ray_dirs_batch = ray_dirs_batch[shuffled_idx]
                
                self.rgbs_batch_dict[0] = rgbs_batch
                self.depth_batch_dict[0] = depth_batch
                self.frame_batch_dict[0] = frame_batch
                self.ray_dirs_batch_dict[0] = ray_dirs_batch
                self.t_wc_batch_dict[0] = t_wc_batch
            
                self.i_batch_dict[0] = 0
        
        # trainer
        trainer_cfg = copy.deepcopy(cfg)
        trainer_cfg.hidden_feature_size = self.hidden_feature_size
        trainer_cfg.obj_scale = self.obj_scale
        self.trainer = trainer.Trainer(trainer_cfg, cls_id, self.obj_ids, adaptive_obj_num=adaptive_obj_num)
        if cls_id == 0:
            self.trainer.bound = load_scene_bound(inst_dict['bbox3D'], cfg)
        elif not adaptive_obj_num and (not self.use_nocs or len(self.obj_ids) == 1):
            self.trainer.bound_dict = {}
            for obj_id in self.obj_ids:
                self.trainer.bound_dict[obj_id] = inst_dict[obj_id]['bbox3D']
        else:
            extent_dict = {}
            for inst_id in inst_dict.keys():
                inst_idx = inst_id_to_index[inst_id]
                inst_info = inst_dict[inst_id]
                
                # extent
                if 'bbox3D' in inst_info.keys():
                    bbox3D = inst_info['bbox3D']
                    extent = bbox3D.extent
                else:
                    extent = np.array([2.0, 2.0, 2.0])
                extent_dict[inst_id] = extent
            self.trainer.extent_dict = extent_dict
            
        if scale_template is not None:
            self.trainer.scale_template = scale_template
            
    def get_training_samples(self, n_samples):
        if self.cls_id == 0 or self.use_equal_batch:
            num_objects = len(list(self.rgbs_batch_dict.keys()))
            n_samples_per_instance = n_samples // num_objects
        
            batch_gt_rgb = []
            batch_gt_depth = []
            batch_depth_mask = []
            batch_obj_mask = []
            batch_input_pcs = []
            batch_sampled_z = []
            batch_indices = []
            
            for idx, obj_id in enumerate(self.rgbs_batch_dict.keys()):
                if idx == len(self.obj_ids)-1:
                    n_samples_per_instance = n_samples - (num_objects-1)*(n_samples // num_objects)
                i_batch = self.i_batch_dict[obj_id]
                rgbs_batch = self.rgbs_batch_dict[obj_id][i_batch:i_batch+n_samples_per_instance].to(self.data_device)
                depth_batch = self.depth_batch_dict[obj_id][i_batch:i_batch+n_samples_per_instance].to(self.data_device)
                frame_batch = self.frame_batch_dict[obj_id][i_batch:i_batch+n_samples_per_instance].to(self.data_device)
                ray_dirs_batch = self.ray_dirs_batch_dict[obj_id][i_batch:i_batch+n_samples_per_instance].to(self.data_device)
                twc_batch = self.t_wc_batch_dict[obj_id]
                if self.cls_id == 0 or (not self.use_nocs) or len(self.obj_ids)==1:
                    sampled_twc = twc_batch[frame_batch, :, :] # (n_samples_per_instance, 4, 4)
                    origins, dirs_o = origin_dirs_W(sampled_twc, ray_dirs_batch)
                else:
                    object_tensor = self.object_tensor_dict[obj_id]
                    two = get_transform_from_tensor_sim3(object_tensor)
                    tco_batch = torch.linalg.inv(twc_batch) @ \
                        two[None, :, :].repeat(twc_batch.shape[0], 1, 1) # (frame_buffer_size, 4, 4)
                
                    # Get sampled keyframe poses
                    sampled_tco = tco_batch[frame_batch, :, :] # (n_samples_per_instance, 4, 4)
                
                    origins, dirs_o = origin_dirs_O(sampled_tco, ray_dirs_batch)
                    
                index = self.trainer.inst_id_to_index[obj_id]
                indices = torch.from_numpy(np.array([index])).repeat(n_samples_per_instance)
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z = \
                    self.sample_3d_points(rgbs_batch, depth_batch, origins, dirs_o)
                    
                batch_gt_rgb.append(gt_rgb)
                batch_gt_depth.append(gt_depth)
                batch_depth_mask.append(valid_depth_mask)
                batch_obj_mask.append(obj_mask)
                batch_input_pcs.append(input_pcs)
                batch_sampled_z.append(sampled_z)
                batch_indices.append(indices)
                i_batch += n_samples_per_instance
                # shuffle per epoch for each instance
                if i_batch >= self.rgbs_batch_dict[obj_id].shape[0] - n_samples_per_instance:
                    print(f"shuffle data of {obj_id} after each epoch")
                    rand_idx = torch.randperm(self.rgbs_batch_dict[obj_id].shape[0])
                    self.rgbs_batch_dict[obj_id] = self.rgbs_batch_dict[obj_id][rand_idx]
                    self.depth_batch_dict[obj_id] = self.depth_batch_dict[obj_id][rand_idx]
                    self.frame_batch_dict[obj_id] = self.frame_batch_dict[obj_id][rand_idx]
                    self.ray_dirs_batch_dict[obj_id] = self.ray_dirs_batch_dict[obj_id][rand_idx]
                    i_batch = 0
                    
                # update i_batch value
                self.i_batch_dict[obj_id] = i_batch

            batch_gt_rgb = torch.cat(batch_gt_rgb)
            batch_gt_depth = torch.cat(batch_gt_depth)
            batch_depth_mask = torch.cat(batch_depth_mask)
            batch_obj_mask = torch.cat(batch_obj_mask)
            batch_input_pcs = torch.cat(batch_input_pcs)
            batch_sampled_z = torch.cat(batch_sampled_z)
            batch_indices = torch.cat(batch_indices)
            
            # data for explicit guide of mean shape
            if (self.use_zero_code or self.use_mean_code) and num_objects > 1: # choice 2: same amount for instance-free & each instance
                agnostic_indices = torch.from_numpy(np.random.choice(n_samples, size=self.n_per_optim - n_samples, replace=False))
                batch_gt_rgb = torch.cat([batch_gt_rgb, batch_gt_rgb[agnostic_indices]])
                batch_gt_depth = torch.cat([batch_gt_depth, batch_gt_depth[agnostic_indices]])
                batch_depth_mask = torch.cat([batch_depth_mask, batch_depth_mask[agnostic_indices]])
                batch_obj_mask = torch.cat([batch_obj_mask, batch_obj_mask[agnostic_indices]])
                batch_input_pcs = torch.cat([batch_input_pcs, batch_input_pcs[agnostic_indices]])
                batch_sampled_z = torch.cat([batch_sampled_z, batch_sampled_z[agnostic_indices]])
        else:
            batch_indices = self.batch_indices_all[self.i_batch:self.i_batch+n_samples]
            
            rgbs_batch = self.rgbs_batch_all[self.i_batch:self.i_batch+n_samples].to(self.data_device)
            depth_batch = self.depth_batch_all[self.i_batch:self.i_batch+n_samples].to(self.data_device)
            ray_dirs_batch = self.ray_dirs_batch_all[self.i_batch:self.i_batch+n_samples].to(self.data_device)
            if self.adaptive_obj_num or (self.use_nocs and len(self.obj_ids) > 1):
                tco_batch = self.t_co_batch_all[self.i_batch:self.i_batch+n_samples].to(self.data_device)
                origins, dirs_o = origin_dirs_O(tco_batch, ray_dirs_batch)
            else:
                twc_batch = self.t_wc_batch_all[self.i_batch:self.i_batch+n_samples].to(self.data_device)
                origins, dirs_o = origin_dirs_W(twc_batch, ray_dirs_batch)
            
            batch_gt_rgb, batch_gt_depth, batch_depth_mask, batch_obj_mask, batch_input_pcs, batch_sampled_z = \
                self.sample_3d_points(rgbs_batch, depth_batch, origins, dirs_o)
                
            self.i_batch += n_samples
            # shuffle per epoch for each instance
            if self.i_batch >= self.rgbs_batch_all.shape[0] - n_samples:
                if not self.no_shuffle:
                    print(f"shuffle data of {self.cls_id} after each epoch")
                    rand_idx = torch.randperm(self.rgbs_batch_all.shape[0])
                    self.batch_indices_all = self.batch_indices_all[rand_idx]
                    self.rgbs_batch_all = self.rgbs_batch_all[rand_idx]
                    self.depth_batch_all = self.depth_batch_all[rand_idx]
                    self.ray_dirs_batch_all = self.ray_dirs_batch_all[rand_idx]
                    self.t_co_batch_all = self.t_co_batch_all[rand_idx]
                    if not self.adaptive_obj_num and (not self.use_nocs or len(self.obj_ids) == 1):
                        self.t_wc_batch_all = self.t_wc_batch_all[rand_idx]
                self.i_batch = 0
            
        return batch_gt_rgb, batch_gt_depth, batch_depth_mask, batch_obj_mask, batch_input_pcs, batch_sampled_z, batch_indices.to(self.data_device)

    def get_uncertainty_guided_training_samples(self, rgb_list, depth_list, obj_list, cached_rays_dir, n_samples):
        selected_inds, _, _ = importance_sampling_coords(self.uncertainty_batch.unsqueeze(0), n_samples)
        selected_inds = selected_inds.squeeze(0)
        # selected_inds = np.random.randint(self.uncertainty_batch.shape[0], size=n_samples)
        inst_ids_selected = self.ids_batch[selected_inds]
        frames_selected = self.frames_batch[selected_inds]
        frame_ids_selected = self.frame_ids_batch[selected_inds]
        
        bbox_2d_list = torch.stack([self.bbox_2d_dict[inst_ids_selected[i].item()][frames_selected[i]] for i in range(n_samples)])
        idx_w = torch.rand(n_samples)
        idx_h = torch.rand(n_samples)
        # resizing idx_w and idx_h to be in the bbox range
        idx_w = idx_w * (bbox_2d_list[:, 1] - bbox_2d_list[:, 0]) + bbox_2d_list[:, 0]
        idx_h = idx_h * (bbox_2d_list[:, 3] - bbox_2d_list[:, 2]) + bbox_2d_list[:, 2]
        idx_w = idx_w.long()
        idx_h = idx_h.long()
        
        obj_indices = torch.from_numpy(np.stack([self.trainer.inst_id_to_index[inst_ids_selected[i].item()] for i in range(n_samples)])).to(self.data_device)
        twc_batch = torch.stack([self.t_wc_batch_dict[inst_ids_selected[i].item()][frames_selected[i]] for i in range(n_samples)])
        object_tensor_list = torch.stack([self.object_tensor_dict[inst_id] for inst_id in self.obj_ids])
        two_list = get_transform_from_tensor_sim3(object_tensor_list)
        two_batch = two_list[obj_indices]
        sampled_tco = torch.linalg.inv(twc_batch) @ two_batch
        
        rgb_batch = rgb_list[frame_ids_selected, idx_w, idx_h]
        obj_batch = obj_list[frame_ids_selected, idx_w, idx_h]
        state = torch.zeros_like(obj_batch, dtype=torch.uint8)
        state[obj_batch==inst_ids_selected] = 1
        state[obj_batch == -1] = 2
        rgbs_batch = torch.cat([rgb_batch, state[..., None]], dim=-1).to(self.data_device)
        
        depth_batch = depth_list[frame_ids_selected, idx_w, idx_h].to(self.data_device)
        ray_dirs_batch = cached_rays_dir[idx_w, idx_h].to(self.data_device)
            
        origins, dirs_o = origin_dirs_O(sampled_tco, ray_dirs_batch)
        gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z = \
            self.sample_3d_points(rgbs_batch, depth_batch, origins, dirs_o)
            
        return gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, obj_indices
        

    def get_uncertainty_guided_training_samples_(self, rgb_list, depth_list, obj_list, cached_rays_dir, metric_list, phi, theta, n_samples, mask_rate_dict=None):
        n_origins = self.cam_origins.shape[0]
        if self.uniform_region:
            sampled_inds = np.random.randint(phi.shape[0], size=n_samples)
        else:
            metrics = torch.mean(metric_list, dim=-1) #sum
            if self.smooth:
                metrics = 1 + metrics#/torch.linalg.norm(metrics)
            sampled_inds, _, _ = importance_sampling_coords(metrics.unsqueeze(0), n_samples)
            sampled_inds = sampled_inds.squeeze(0)
        
        phi_sampled = phi[sampled_inds].to(self.data_device)
        theta_sampled = theta[sampled_inds].to(self.data_device)
        x = torch.sin(phi_sampled) * torch.cos(theta_sampled)
        y = torch.sin(phi_sampled) * torch.sin(theta_sampled)
        z = torch.cos(phi_sampled)
        points_sampled = torch.stack([x, y, z], dim=-1)
        
        # extents = np.stack([self.extent_dict[inst_id]/2 for inst_id in self.obj_ids])
        # r = 1.2*np.linalg.norm(extents, axis=-1).max()
        # transform_np = get_transform_from_tensor_sim3(self.object_tensor_dict[self.obj_ids[0]]).cpu().numpy()
        # transform_np = np.linalg.inv(transform_np)
        # x_all = torch.sin(phi) * torch.cos(theta)
        # y_all = torch.sin(phi) * torch.sin(theta)
        # z_all = torch.cos(phi)
        # points_all = torch.stack([x_all, y_all, z_all], dim=-1).reshape((100,100,3))
        # plot_selected(np.copy(metrics.numpy()), self.cam_origins.cpu().numpy(), 1.7*points_all, sampled_inds, obj_id=self.obj_ids[0], 
        #                        mesh_dir='/media/satassd_1/tblee-larr/CVPR24/vMAP_offline/logs/0718/room_0/scene_mesh', transform_np=transform_np)

        cos = torch.sum(self.cam_origins[None,...].repeat((n_samples,1,1)) * points_sampled[:,None,:].repeat((1,n_origins,1)), dim=-1)
        # sin_culling = torch.linalg.norm(torch.cross(self.cam_z[None,...].repeat((n_samples,1,1)), \
        #     points_sampled[:,None,:].repeat((1,n_origins,1)), dim=-1), dim=-1)
        distances = torch.acos(cos) # (n_samples, n_origins)
        if self.use_certain_data:
            # nearest, _ = torch.min(distances, dim=-1, keepdim=True)
            # distances = torch.where(distances<2*nearest, distances, np.pi)
            _, selected_inds_topk = torch.topk(distances, 5, dim=-1, largest=False) # (n_samples, 5) # selected_inds_topk w.r.t sorted distances_topk
            if mask_rate_dict is None:
                # metric_list = metric_list/torch.sum(metric_list, dim=0, keepdim=True) # 0920
                inst_index_topk = self.indices_batch[selected_inds_topk.reshape(-1)].reshape((n_samples, 5))
                sampled_metric_list = metric_list[sampled_inds] # (n_samples, len(self.obj_ids))
                sampled_metric_topk = torch.gather(sampled_metric_list, -1, inst_index_topk)
            else:
                inst_ids_selected_topk = self.ids_batch[selected_inds_topk.reshape(-1)]
                frames_selected_topk = self.frames_batch[selected_inds_topk.reshape(-1)]
                mask_rate_batch = torch.stack([mask_rate_dict[inst_ids_selected_topk[i].item()][frames_selected_topk[i]] for i in range(5*n_samples)])
                sampled_metric_topk = mask_rate_batch.reshape((n_samples, 5))
            if self.select_minimum:
                # least uncertain instance for each point in points_sampled
                sel = torch.argmin(sampled_metric_topk, dim=-1, keepdim=True).to(self.data_device)
            else:
                sampled_metric_topk = sampled_metric_topk/torch.sum(sampled_metric_topk, dim=-1, keepdim=True)
                sel, _, _ = importance_sampling_coords(1-sampled_metric_topk, 1)
                sel = sel.to(self.data_device)
            selected_inds = torch.gather(selected_inds_topk, -1, sel).squeeze()
                  
        else:
            selected_inds = torch.argmin(distances, dim=-1)
        inst_ids_selected = self.ids_batch[selected_inds]
        frames_selected = self.frames_batch[selected_inds]
        frame_ids_selected = self.frame_ids_batch[selected_inds]
        
        # plot_selected(metrics.numpy().reshape((100,100)), self.cam_origins.cpu().numpy(), 1.7*points_all, selected_inds, selected_view=True, obj_id=self.obj_ids[0], 
        #                        mesh_dir='/media/satassd_1/tblee-larr/CVPR24/vMAP_offline/logs/0718/room_0/scene_mesh', transform_np=transform_np)
        
        bbox_2d_list = torch.stack([self.bbox_2d_dict[inst_ids_selected[i].item()][frames_selected[i]] for i in range(n_samples)])
        # selected_obj_origins = self.obj_origins[selected_inds]
        # pixels_homo = (torch.from_numpy(intrinsic.astype(np.float32)).to(self.training_device) @ (selected_obj_origins.T)).T
        # pixels_uv = (pixels_homo[:, :2] / pixels_homo[:, 2:])
        
        # mask1 = pixels_uv[:,0] >= bbox_2d_list[:,0]
        # mask2 = pixels_uv[:,0] <= bbox_2d_list[:,1]
        # mask3 = pixels_uv[:,1] >= bbox_2d_list[:,2]
        # mask4 = pixels_uv[:,1] <= bbox_2d_list[:,3]
        # mask = torch.logical_and(torch.logical_and(mask1, mask2),
        #                          torch.logical_and(mask3, mask4))
        
        # minimum_radius = torch.min(torch.abs(torch.stack([pixels_uv[:,:1]-bbox_2d_list[:,:2], pixels_uv[:,1:]-bbox_2d_list[:,2:]], dim=-1)), dim=-2)[0] # TODO: or pre-defined?
        # sigma = minimum_radius/2
        # mvn = MultivariateNormal(pixels_uv, torch.diag_embed(sigma))
        # idx_ = mvn.sample()
        
        # idx_w = idx_[:,0]
        # idx_h = idx_[:,1]
        # # idx_w = torch.zeros(n_samples)
        # # idx_h = torch.zeros(n_samples)
        # # idx_w[mask] = idx_[:,0]
        # # idx_h[mask] = idx_[:,1]
        # # idx_w[~mask1] = bbox_2d_list[~mask1,0]
        # # idx_w[~mask2] = bbox_2d_list[~mask2,1]
        # # idx_h[~mask3] = bbox_2d_list[~mask3,2]
        # # idx_h[~mask4] = bbox_2d_list[~mask4,3]
        # idx_w = torch.where(idx_w > bbox_2d_list[:,1], bbox_2d_list[:,1], idx_w)
        # idx_w = torch.where(idx_w < bbox_2d_list[:,0], bbox_2d_list[:,0], idx_w)
        # idx_h = torch.where(idx_h > bbox_2d_list[:,3], bbox_2d_list[:,3], idx_h)
        # idx_h = torch.where(idx_h < bbox_2d_list[:,2], bbox_2d_list[:,2], idx_h)
        idx_w = torch.rand(n_samples)
        idx_h = torch.rand(n_samples)
        # resizing idx_w and idx_h to be in the bbox range
        idx_w = idx_w * (bbox_2d_list[:, 1] - bbox_2d_list[:, 0]) + bbox_2d_list[:, 0]
        idx_h = idx_h * (bbox_2d_list[:, 3] - bbox_2d_list[:, 2]) + bbox_2d_list[:, 2]
        idx_w = idx_w.long()
        idx_h = idx_h.long()
        
        obj_indices = torch.from_numpy(np.stack([self.trainer.inst_id_to_index[inst_ids_selected[i].item()] for i in range(n_samples)])).to(self.data_device)
        twc_batch = torch.stack([self.t_wc_batch_dict[inst_ids_selected[i].item()][frames_selected[i]] for i in range(n_samples)])
        object_tensor_list = torch.stack([self.object_tensor_dict[inst_id] for inst_id in self.obj_ids])
        two_list = get_transform_from_tensor_sim3(object_tensor_list)
        two_batch = two_list[obj_indices]
        sampled_tco = torch.linalg.inv(twc_batch) @ two_batch
        
        rgb_batch = rgb_list[frame_ids_selected, idx_w, idx_h]
        obj_batch = obj_list[frame_ids_selected, idx_w, idx_h]
        state = torch.zeros_like(obj_batch, dtype=torch.uint8)
        state[obj_batch==inst_ids_selected] = 1
        state[obj_batch == -1] = 2
        rgbs_batch = torch.cat([rgb_batch, state[..., None]], dim=-1).to(self.data_device)
        
        depth_batch = depth_list[frame_ids_selected, idx_w, idx_h].to(self.data_device)
        ray_dirs_batch = cached_rays_dir[idx_w, idx_h].to(self.data_device)
            
        origins, dirs_o = origin_dirs_O(sampled_tco, ray_dirs_batch)
        gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z = \
            self.sample_3d_points(rgbs_batch, depth_batch, origins, dirs_o)
            
        return gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z, obj_indices
        
    def sample_3d_points(self, sampled_rgbs, sampled_depth, origins, dirs_o):
        """
        slightly changed from vMap code (related to origins, dirs)
        
        3D sampling strategy

        * For pixels with invalid depth:
            - N+M from minimum bound to max (stratified)
        
        * For pixels with valid depth:
            # Pixel belongs to this object
                - N from cam to surface (stratified)
                - M around surface (stratified/normal)
            # Pixel belongs that don't belong to this object
                - N from cam to surface (stratified)
                - M around surface (stratified)
            # Pixel with unknown state
                - Do nothing!
        """

        n_bins_cam2surface = self.n_bins_cam2surface
        n_bins = self.n_bins
        eps = self.surface_eps
        other_objs_max_eps = self.stop_eps #0.05   # todo 0.02
        # print("max depth ", torch.max(sampled_depth))
        sampled_z = torch.zeros(
            sampled_rgbs.shape[0],
            n_bins_cam2surface + n_bins,
            dtype=sampled_depth.dtype,
            device=self.data_device)  # shape (N*n_rays, n_bins_cam2surface + n_bins)

        invalid_depth_mask = (sampled_depth <= self.min_bound).view(-1)
        # max_bound = self.max_bound
        max_bound = torch.max(sampled_depth)
        # sampling for points with invalid depth
        invalid_depth_count = invalid_depth_mask.count_nonzero()
        if invalid_depth_count:
            sampled_z[invalid_depth_mask, :] = stratified_bins(
                self.min_bound, max_bound,
                n_bins_cam2surface + n_bins, invalid_depth_count,
                device=self.data_device)

        # sampling for valid depth rays
        valid_depth_mask = ~invalid_depth_mask
        valid_depth_count = valid_depth_mask.count_nonzero()


        if valid_depth_count:
            # Sample between min bound and depth for all pixels with valid depth
            sampled_z[valid_depth_mask, :n_bins_cam2surface] = stratified_bins(
                self.min_bound, sampled_depth.view(-1)[valid_depth_mask]-eps,
                n_bins_cam2surface, valid_depth_count, device=self.data_device)

            # sampling around depth for this object
            obj_mask = (sampled_rgbs[..., -1] == self.this_obj).view(-1) & valid_depth_mask # todo obj_mask
            assert sampled_z.shape[0] == obj_mask.shape[0]
            obj_count = obj_mask.count_nonzero()

            if obj_count:
                sampling_method = "normal"  # stratified or normal
                if sampling_method == "stratified":
                    sampled_z[obj_mask, n_bins_cam2surface:] = stratified_bins(
                        sampled_depth.view(-1)[obj_mask] - eps, sampled_depth.view(-1)[obj_mask] + eps,
                        n_bins, obj_count, device=self.data_device)

                elif sampling_method == "normal":
                    sampled_z[obj_mask, n_bins_cam2surface:] = normal_bins_sampling(
                        sampled_depth.view(-1)[obj_mask],
                        n_bins,
                        obj_count,
                        delta=eps,
                        device=self.data_device)

                else:
                    raise (
                        f"sampling method not implemented {sampling_method}, \
                            stratified and normal sampling only currenty implemented."
                    )

            # sampling around depth of other objects
            other_obj_mask = (sampled_rgbs[..., -1] != self.this_obj).view(-1) & valid_depth_mask
            other_objs_count = other_obj_mask.count_nonzero()
            if other_objs_count:
                sampled_z[other_obj_mask, n_bins_cam2surface:] = stratified_bins(
                    sampled_depth.view(-1)[other_obj_mask] - eps,
                    sampled_depth.view(-1)[other_obj_mask] + other_objs_max_eps,
                    n_bins, other_objs_count, device=self.data_device)
        
        sampled_z = sampled_z.view(sampled_rgbs.shape[0], -1)  # view as (n_rays, n_samples)
        
        input_pcs = origins[..., None, :] + (dirs_o[:, None, :] *
                                                    sampled_z[..., None])
        obj_labels = sampled_rgbs[..., -1].view(-1)
        return sampled_rgbs[..., :3], sampled_depth, valid_depth_mask, obj_labels, input_pcs, sampled_z
    
    def save_checkpoints(self, path, iter, chamfer_dict=None, chamfer_opposite_dict=None):
        checkpoint_load_file = (path + "/cls_" + str(self.cls_id) + "_iteration_{:05d}.pth".format(iter))
        if self.cls_id != 0 and self.object_wise_model:
            save_dict = {
                "global_step": iter,
                "FC_state_dict": {key: self.trainer.fc_occ_map_dict[key].state_dict() for key in self.trainer.fc_occ_map_dict.keys()},
                "cls_id": self.cls_id,
                "instance_id_to_index": self.trainer.inst_id_to_index,
                "obj_scale": self.trainer.obj_scale
            }
        else:
            save_dict = {
                "global_step": iter,
                "FC_state_dict": self.trainer.fc_occ_map.state_dict(),
                "cls_id": self.cls_id,
                "instance_id_to_index": self.trainer.inst_id_to_index,
                "obj_scale": self.trainer.obj_scale
            }
        if self.cls_id == 0:
            save_dict["PE_state_dict"] = self.trainer.pe.state_dict()
            save_dict["bound"] = self.trainer.bound
        else:
            save_dict["obj_tensor_dict"] = self.object_tensor_dict
            if self.adaptive_obj_num or len(self.obj_ids) > 1:
                save_dict["extent_dict"] = self.extent_dict
            if self.object_wise_model:
                save_dict["PE_state_dict"] = {key: self.trainer.pe_dict[key].state_dict() for key in self.trainer.pe_dict.keys()}
            else:
                save_dict["PE_state_dict"] = self.trainer.pe.state_dict()
                save_dict["shape_code_state_dict"] = self.trainer.shape_codes.state_dict()
                save_dict["texture_code_state_dict"] = self.trainer.texture_codes.state_dict()
            save_dict["bound"] = self.trainer.extent_dict
            
        if chamfer_dict is not None:
            save_dict["chamfer_dict"] = chamfer_dict
            
        if chamfer_opposite_dict is not None:
            save_dict["chamfer_opposite_dict"] = chamfer_opposite_dict
        
        torch.save(
            save_dict,
            checkpoint_load_file,
        )
        # optimiser?

    def load_checkpoints(self, ckpt_file, load_codenerf=False):
        checkpoint_load_file = (ckpt_file)
        if not os.path.exists(checkpoint_load_file):
            print("ckpt not exist ", checkpoint_load_file)
            return
        checkpoint = torch.load(checkpoint_load_file)
        if load_codenerf:
            self.trainer.fc_occ_map.load_state_dict(checkpoint["model_params"])
            self.trainer.fc_occ_map.to(self.training_device)
            self.trainer.pe.load_state_dict(checkpoint["pe"])
            self.trainer.pe.to(self.training_device)
            if self.use_mean_code:
                self.trainer.shape_codes.load_state_dict(checkpoint["shape_code_params"])
                self.trainer.texture_codes.load_state_dict(checkpoint["texture_code_params"])
                self.trainer.shape_codes.to(self.training_device)
                self.trainer.texture_codes.to(self.training_device)
        else:
            self.cls_id = checkpoint["cls_id"]
            if self.cls_id != 0 and self.object_wise_model:
                for obj_id in self.trainer.fc_occ_map_dict.keys():
                    self.trainer.fc_occ_map_dict[obj_id].load_state_dict(checkpoint["FC_state_dict"][obj_id])
                    self.trainer.pe_dict[obj_id].load_state_dict(checkpoint["PE_state_dict"][obj_id])
                    self.trainer.fc_occ_map_dict[obj_id].to(self.training_device)
                    self.trainer.pe_dict[obj_id].to(self.training_device)
            else:
                self.trainer.fc_occ_map.load_state_dict(checkpoint["FC_state_dict"])
                self.trainer.pe.load_state_dict(checkpoint["PE_state_dict"])
                self.trainer.fc_occ_map.to(self.training_device)
                self.trainer.pe.to(self.training_device)
            if self.cls_id != 0:
                self.object_tensor_dict = checkpoint["obj_tensor_dict"]
                if not self.object_wise_model:
                    self.trainer.shape_codes.load_state_dict(checkpoint["shape_code_state_dict"])
                    self.trainer.texture_codes.load_state_dict(checkpoint["texture_code_state_dict"])
                    self.trainer.shape_codes.to(self.training_device)
                    self.trainer.texture_codes.to(self.training_device)
                self.trainer.bound = checkpoint["bound"]
            else:
                self.trainer.extent_dict = checkpoint["bound"]
            # self.trainer.inst_id_to_index = checkpoint["instance_id_to_index"] # TODO: temporary, uncomment after train again
            
            self.trainer.obj_scale = checkpoint["obj_scale"]           
            self.start = checkpoint["global_step"]

    
class cameraInfo:

    def __init__(self, cfg) -> None:
        self.width = cfg.W  # Frame width
        self.height = cfg.H  # Frame height

        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy

        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self, depth_type="z"):
        idx_w = torch.arange(end=self.width)
        idx_h = torch.arange(end=self.height)

        dirs = torch.ones((self.width, self.height, 3))

        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None]
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy)

        if depth_type == "euclidean":
            raise Exception(
                "Get camera rays directions with euclidean depth not yet implemented"
            )
            norm = torch.norm(dirs, dim=-1)
            dirs = dirs * (1. / norm)[:, :, :, None]

        return dirs

# DEBUG
if __name__ == '__main__':
    from dataset import Replica, ScanNet
    
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--config', default='./configs/ScanNet/config_scannet0000_vMAP.json', type=str)
    # parser.add_argument('--config', default='./configs/Replica/config_replica_room0_vMAP.json', type=str)
    args = parser.parse_args()
    config_file = args.config
    cfg = Config(config_file)
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg) 

    cam_info = cameraInfo(cfg)
    
    category_dict = dataset.inst_dict
    sample_dict = dataset.sample_dict
    cls_id_list = [80] #[18, 29, 76, 44, 78, 65, 91, 63, 20, 11]
    for cls_id in category_dict.keys():
        if not cls_id in cls_id_list:
            continue
        inst_dict = category_dict[cls_id]
        scene_category = sceneCategory(cfg, cls_id, inst_dict, sample_dict, cam_info.rays_dir_cache)          
        scene_category.get_training_samples(50)
    print('hi')            
                
                
                