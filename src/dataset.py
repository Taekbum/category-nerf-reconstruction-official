from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
import glob
from torchvision import transforms
import image_transforms
import open3d
import time
import pickle

from utils import enlarge_bbox, get_bbox2d, get_bbox2d_batch, geometry_segmentation, refine_inst_data, unproject_pointcloud
from category_registration import *

def get_dataset(cfg, train_mode=True):
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg, train_mode=train_mode)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg) 
    else:
        print("Dataset format {} not found".format(cfg.dataset_format))
        exit(-1)
    return dataset     

class Replica(Dataset):
    def __init__(self, cfg, train_mode=True):
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

        self.get_all_frames()
        
        result_file = os.path.join(self.root_dir, "inst_dict.pkl")
        if cfg.load_registration_result and os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                self.inst_dict = pickle.load(f)
        else:
            bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict = {}, {}, {}, {}
            get_all_poses(self.inst_dict, self.sample_dict, self.intrinsic_open3d, name=self.name, depth_scale=cfg.depth_scale, max_depth=cfg.max_depth)
            get_uncertainty_fields(self.inst_dict, bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict, cfg,
                                   name=self.name, load_pretrained=cfg.load_pretrained)
            align_poses(self.inst_dict, bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict, 
                        name=self.name, multi_init_pose=cfg.multi_init_pose, eta1=cfg.eta1, eta2=cfg.eta2, eta3=cfg.eta3, device=self.device)

            with open(result_file, 'wb') as f:
                pickle.dump(self.inst_dict, f)
            # self.get_all_poses()
            # self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained)
            # self.align_poses(eta1=cfg.eta1, eta2=cfg.eta2, eta3=cfg.eta3)
        
    def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        self.sample_dict = {}
        cls_id_undefined = 1000
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
                    cls_list.append(inst_id+cls_id_undefined)
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

            T = self.Twc[idx]
            
            if self.depth_transform:
                depth = self.depth_transform(depth) 

            sample = {"image": image, "depth": depth, "obj_mask": inst, "T": T, "frame_id": idx}
                             
            if image is None or depth is None:
                print(rgb_file)
                print(depth_file)
                raise ValueError
            
            self.sample_dict[idx] = sample  
        
        t2 = time.time()
        print('get_all_frames takes {} seconds'.format(t2-t1))               
                    
    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        return self.sample_dict[idx]

class ScanNet(Dataset):
    def __init__(self, cfg):
        self.name = "scannet"
        self.device = cfg.data_device
        self.root_dir = cfg.dataset_dir
        self.color_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        if cfg.load_refined_mask:
            self.inst_paths = sorted(glob.glob(os.path.join(
                self.root_dir, 'instance-refined', '*.npy')), key=lambda x: int(os.path.basename(x)[:-4]))
            self.sem_paths = sorted(glob.glob(os.path.join(
                self.root_dir, 'label-refined', '*.npy')), key=lambda x: int(os.path.basename(x)[:-4]))
        else:
            self.inst_paths = sorted(glob.glob(os.path.join(
                self.root_dir, 'instance-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
            self.sem_paths = sorted(glob.glob(os.path.join(
                self.root_dir, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
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

        # from scannetv2-labels.combined.tsv
        #1-wall, 3-floor, 16-window, 41-ceiling, 232-light switch   0-unknown? 21-pillar 161-doorframe, shower walls-128, curtain-21, windowsill-141
        self.background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        self.bbox_scale = 0.2
        self.inst_filter_dict = {}
        self.inst_dict = {}
        
        self.use_refined_mask = cfg.use_refined_mask
        self.load_refined_mask = cfg.load_refined_mask
        
        self.get_all_frames()
        
        result_file = os.path.join(self.root_dir, "inst_dict.pkl")
        if cfg.load_registration_result and os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                self.inst_dict = pickle.load(f)
        else:
            bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict = {}, {}, {}, {}
            get_all_poses(self.inst_dict, self.sample_dict, self.intrinsic_open3d, name=self.name, depth_scale=self.depth_scale, max_depth=self.max_depth)
            get_uncertainty_fields(self.inst_dict, bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict, cfg,
                                    name=self.name, load_pretrained=cfg.load_pretrained)
            align_poses(self.inst_dict, bbox3d_dict, count_dict, pe_dict, fc_occ_map_dict, 
                        name=self.name, multi_init_pose=cfg.multi_init_pose, eta1=cfg.eta1, eta2=cfg.eta2, eta3=cfg.eta3, device=self.device) 
            
            with open(result_file, 'wb') as f:
                pickle.dump(self.inst_dict, f)
            # self.get_all_poses()
            # self.get_uncertainty_fields(cfg, load_pretrained=cfg.load_pretrained)
            # self.align_poses()

    def get_all_frames(self):
        print('get_all_frames')
        t1 = time.time()
        self.inst_dict = {}
        self.sample_dict = {}
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
            T_CW = np.linalg.inv(T)     

            H, W = depth_data.shape
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
            
            if self.edge:
                edge = self.edge # crop image edge, there are invalid value on the edge of the color image
                color_data = color_data[edge:-edge, edge:-edge]
                depth_data = depth_data[edge:-edge, edge:-edge]
            if self.depth_transform:
                depth_data = self.depth_transform(depth_data)
            
            if self.load_refined_mask and os.path.exists(inst_path) and os.path.exists(sem_path):
                inst_data = np.load(inst_path)
                sem_data = np.load(sem_path)
                
                cls_list = []
                inst_list = []
                inst_to_cls = {0:0}
                for inst_id in np.unique(inst_data):
                    inst_mask = inst_data == inst_id
                    sem_cls = np.unique(sem_data[inst_mask])  # sem label, only interested obj
                    assert sem_cls.shape[0] == 1
                    sem_cls = sem_cls[0]
                    cls_list.append(sem_cls)
                    inst_list.append(inst_id)
                    inst_to_cls[inst_id] = sem_cls
            else:    
                inst_data = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
                inst_data = cv2.resize(inst_data, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                sem_data = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
                sem_data = cv2.resize(sem_data, (W, H), interpolation=cv2.INTER_NEAREST)
            
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
                
                # semantically refine depth segmentation
                if self.use_refined_mask:
                    normal, geometry_label, segment_masks, segments = geometry_segmentation(color_data, depth_data, self.intrinsic_open3d)
                    inst_data = refine_inst_data(inst_data, segment_masks)
                    inst_path_new = os.path.join(self.root_dir, 'instance-refined', os.path.basename(inst_path)[:-4]+".npy")
                    sem_path_new = os.path.join(self.root_dir, 'label-refined', os.path.basename(sem_path)[:-4]+".npy")
                    np.save(inst_path_new, inst_data)
                    np.save(sem_path_new, sem_data)
            
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
            obj_ids = np.unique(inst_data)
            for obj_id in obj_ids:
                if obj_id == 0:
                    continue
                depth_data_copy = depth_data.copy()
                mask = inst_data == obj_id
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

            sample = {"image": color_data.transpose(1,0,2), "depth": depth_data.transpose(1,0), "obj_mask": inst_data.transpose(1,0), "T": T, "frame_id": index_reduced}#"sem_mask": sem_data, 
                  
            if color_data is None or depth_data is None:
                print(color_path)
                print(depth_path)
                raise ValueError
            
            self.sample_dict[index_reduced] = sample    

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
        return self.sample_dict[index]