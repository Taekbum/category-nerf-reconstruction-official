import torch
import torch.nn as nn
import model
import embedding
import render_rays
from utils import get_transform_from_tensor
import numpy as np
import vis
from tqdm import tqdm
import math

class Trainer:
    def __init__(self, cfg, cls_id, inst_ids, adaptive_obj_num=False):
        self.adaptive_obj_num = cfg.adaptive_obj_num
        self.object_wise_model = cfg.object_wise_model
        self.enlarge_scale = cfg.enlarge_scale
        self.use_nocs = cfg.use_nocs
        self.cls_id = cls_id
        self.use_zero_code = cfg.use_zero_code
        self.use_mean_code = cfg.use_mean_code
        if len(inst_ids) == 1 and cfg.use_zero_code:
            self.inst_id_to_index = {inst_ids[0]: 0}
        else:
            self.inst_id_to_index = {inst_id: inst_ids.index(inst_id) for inst_id in inst_ids}
        if cfg.codenerf and cfg.use_mean_code:
            self.n_obj = 4612
        else:
            self.n_obj = len(inst_ids)
        self.device = cfg.training_device
        self.obj_scale = cfg.obj_scale # 10 for bg and iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs

        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        if cls_id == 0:
            self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
            self.load_NeRF()
        elif cfg.object_wise_model:
            self.hidden_feature_size = cfg.hidden_feature_size
            self.load_NeRF_dict(inst_ids)
        else:
            self.net_hyperparams = cfg.net_hyperparams
            self.editnerf = cfg.editnerf
            self.load_codeNeRF()
            self.load_codes(adaptive_obj_num=adaptive_obj_num)

        self.extent_dict = None
        if self.cls_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9
            
        self.scale_template = None

    def load_codeNeRF(self):
        # self.fc_occ_map = model.CodeNeRF(**self.net_hyperparams).to(self.device)
        if self.editnerf:
            self.fc_occ_map = model.EditNeRF(self.emb_size1, self.emb_size2, **self.net_hyperparams).to(self.device)
        else:
            self.fc_occ_map = model.CodeNeRF(self.emb_size1, self.emb_size2, **self.net_hyperparams).to(self.device)
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        # def pe_func_obj(x):
        #     return model.PE(x, self.fc_occ_map.num_xyz_freq, obj_scale=self.obj_scale)
        # self.pe = pe_func_obj
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def load_NeRF(self):
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size
        )
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)
    
    def load_NeRF_dict(self, inst_ids):
        self.fc_occ_map_dict = {}
        self.pe_dict = {}
        for i in inst_ids:
            if not i in self.fc_occ_map_dict.keys():
                self.fc_occ_map_dict[i] = model.OccupancyMap(
                    self.emb_size1,
                    self.emb_size2,
                    hidden_size=self.hidden_feature_size
                )
                self.fc_occ_map_dict[i].apply(model.init_weights).to(self.device)
                self.pe_dict[i] = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def load_codes(self, adaptive_obj_num=False):
        embdim = self.net_hyperparams['latent_dim']
        d = 10 if adaptive_obj_num else self.n_obj
        self.shape_codes = nn.Embedding(d, embdim)
        self.texture_codes = nn.Embedding(d, embdim)
        self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
        self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))            
        self.shape_codes = self.shape_codes.to(self.device)
        self.texture_codes = self.texture_codes.to(self.device)
        if self.use_zero_code:
            self.zero_codes = nn.Embedding(1, embdim)
            self.zero_codes.weight = nn.Parameter(torch.zeros((1, embdim)), requires_grad=False)
            self.zero_codes = self.zero_codes.to(self.device)
    
    def meshing(self, inst_id=None, grid_dim=256, interpolate_mode=None, other_id=None, t=None, average_shape=False, average_texture=False):
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]
        if not self.adaptive_obj_num and (not self.use_nocs or self.cls_id == 0 or self.n_obj==1):
            if self.cls_id == 0:
                bound = self.bound
            else:
                bound = self.bound_dict[inst_id]
            # extent = bound[:,1] - bound[:,0]
            scale_np = bound.extent / (range_dist * self.bound_extent)
            scale = torch.from_numpy(scale_np).float().to(self.device)
            transform_np = np.eye(4, dtype=np.float32)
            transform_np[:3, 3] = bound.center#(bound[:,1] + bound[:,0])/2
            transform_np[:3, :3] = bound.R
            # transform_np = np.linalg.inv(transform_np)  #
            transform = torch.from_numpy(transform_np).to(self.device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                            scale=scale, transform=transform).view(-1, 3)
            if self.cls_id == 0:
                ret = self.eval_points(grid_pc)
            else:
                ret = self.eval_points(grid_pc, inst_id=inst_id, interpolate_mode=interpolate_mode, other_id=other_id, t=t, average_shape=average_shape, average_texture=average_texture)
        else:
            if interpolate_mode == 'shape' or average_shape:
                extent = np.array([2.0, 2.0, 2.0])
            else:
                extent = self.extent_dict[inst_id]
            if self.enlarge_scale:
                extent = extent/np.max(extent)
            else:
                extent = extent/np.max(extent/2)
                if self.scale_template is not None:
                    extent = extent * self.scale_template
            scale_np = extent / (range_dist * self.bound_extent)
            scale = torch.from_numpy(scale_np).float().to(self.device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device, scale=scale).view(-1, 3)
        # grid_pc -= obj_center.to(grid_pc.device)
            ret = self.eval_points(grid_pc, inst_id=inst_id, interpolate_mode=interpolate_mode, other_id=other_id, t=t, average_shape=average_shape, average_texture=average_texture)

        if ret is None:
            return None

        occ, _ = ret
        mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy())
        if mesh is None:
            print("marching cube failed")
            return None

        # Transform to [-1, 1] range
        mesh.apply_translation([-0.5, -0.5, -0.5])
        mesh.apply_scale(2)

        # Transform to scene coordinates
        mesh.apply_scale(scale_np)
        if not self.adaptive_obj_num and (not self.use_nocs or self.cls_id == 0 or self.n_obj==1):
            mesh.apply_transform(transform_np)

        vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        if self.cls_id == 0:
            ret = self.eval_points(vertices_pts)
        else:
            ret = self.eval_points(vertices_pts, inst_id=inst_id, interpolate_mode=interpolate_mode, other_id=other_id, t=t, average_shape=average_shape, average_texture=average_texture)
        if ret is None:
            return None
        _, color = ret
        mesh_color = color * 255
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        return mesh

    def eval_points(self, points, inst_id=None, chunk_size=500000, interpolate_mode=None, other_id=None, t=None, average_shape=False, average_texture=False):
        # 256^3 = 16777216
        if self.cls_id != 0 and not self.object_wise_model:
            obj_idx = torch.from_numpy(np.array(self.inst_id_to_index[inst_id])).to(self.device)
            shape_code, texture_code = self.shape_codes(obj_idx), self.texture_codes(obj_idx)
            if average_shape or average_texture:
                if self.use_zero_code:
                    obj_idx = torch.from_numpy(np.array([0])).to(self.device)
                    embdim = self.net_hyperparams['latent_dim']
                    self.zero_codes = nn.Embedding(1, embdim)
                    self.zero_codes.weight = nn.Parameter(torch.zeros((1, embdim)), requires_grad=False)            
                    self.zero_codes = self.zero_codes.to(self.device)
                    if average_shape: # zero code -> mean shape
                        shape_code = self.zero_codes(obj_idx)
                    if average_texture:
                        texture_code = self.zero_codes(obj_idx)
                elif self.use_mean_code:
                    if average_shape: # zero code -> mean shape
                        shape_code = torch.mean(self.shape_codes.weight, dim=0).reshape(1,-1)
                    if average_texture:
                        texture_code = torch.mean(self.texture_codes.weight, dim=0).reshape(1,-1)
        
        if other_id is not None:
            t = torch.from_numpy(np.array([t]).astype(np.float32)).to(self.device)
            other_idx = torch.from_numpy(np.array(self.inst_id_to_index[other_id])).to(self.device)
            shape_code_other, texture_code_other = self.shape_codes(other_idx), self.texture_codes(other_idx)
            shape_code_interpolate = (1-t) * shape_code + t * shape_code_other
            texture_code_interpolate = (1-t) * texture_code + t * texture_code_other
        
        alpha, color = [], []
        n_chunks = int(np.ceil(points.shape[0] / chunk_size))
        with torch.no_grad():
            for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                if self.cls_id != 0 and self.object_wise_model:
                    embedding_k = self.pe_dict[inst_id](points[chunk_idx, ...])
                else:
                    embedding_k = self.pe(points[chunk_idx, ...])
                if self.cls_id == 0:
                    alpha_k, color_k = self.fc_occ_map(embedding_k)
                else:
                    if self.object_wise_model:
                        alpha_k, color_k = self.fc_occ_map_dict[inst_id](embedding_k)  
                    else:                    
                        if average_shape and self.editnerf:
                            alpha_k, color_k = self.fc_occ_map(embedding_k, shape_code, texture_code, mean_shape=True)
                        else:
                            if interpolate_mode == 'texture':
                                alpha_k, color_k = self.fc_occ_map(embedding_k, shape_code, texture_code_interpolate)
                            elif interpolate_mode == 'shape':
                                alpha_k, color_k = self.fc_occ_map(embedding_k, shape_code_interpolate, texture_code)
                            else:
                                if self.editnerf:
                                    alpha_k, color_k = self.fc_occ_map(embedding_k, shape_code[None, :].repeat(embedding_k.shape[0],1), 
                                                                    texture_code[None, :].repeat(embedding_k.shape[0],1))
                                else:
                                    alpha_k, color_k = self.fc_occ_map(embedding_k, shape_code, texture_code)
                                
                alpha.extend(alpha_k.detach().squeeze())
                color.extend(color_k.detach().squeeze())
        alpha = torch.stack(alpha)
        color = torch.stack(color)

        occ = render_rays.occupancy_activation(alpha).detach()
        if occ.max() == 0:
            print("no occ")
            return None
        return (occ, color)
