import torch
import torch.nn as nn
import model
import embedding
import render_rays
import numpy as np
import vis
from tqdm import tqdm
import math

class Trainer:
    def __init__(self, cfg, cls_id, inst_ids):
        self.cls_id = cls_id
        self.inst_id_to_index = {inst_id: inst_ids.index(inst_id) for inst_id in inst_ids}
        self.n_obj = len(inst_ids)
        self.device = cfg.training_device
        self.obj_scale = cfg.obj_scale # 10 for bg
        self.n_unidir_funcs = cfg.n_unidir_funcs

        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        if cls_id == 0:
            self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 128 for seperate bg
            self.load_NeRF()
        else:
            self.net_hyperparams = cfg.net_hyperparams
            self.load_codeNeRF()
            self.load_codes()

        self.extent_dict = None
        if self.cls_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9
            
        self.scale_template = None

    def load_codeNeRF(self):
        self.fc_occ_map = model.CodeNeRF(self.emb_size1, self.emb_size2, **self.net_hyperparams).to(self.device)
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def load_NeRF(self):
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size
        )
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def load_codes(self):
        embdim = self.net_hyperparams['latent_dim']
        d = self.n_obj
        self.shape_codes = nn.Embedding(d, embdim)
        self.texture_codes = nn.Embedding(d, embdim)
        self.shape_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))
        self.texture_codes.weight = nn.Parameter(torch.randn(d, embdim) / math.sqrt(embdim/2))            
        self.shape_codes = self.shape_codes.to(self.device)
        self.texture_codes = self.texture_codes.to(self.device)
    
    def meshing(self, inst_id=None, grid_dim=256):
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]
        if self.cls_id == 0 or self.n_obj==1:
            if self.cls_id == 0:
                bound = self.bound
            else:
                bound = self.bound_dict[inst_id]
            
            scale_np = bound.extent / (range_dist * self.bound_extent)
            scale = torch.from_numpy(scale_np).float().to(self.device)
            transform_np = np.eye(4, dtype=np.float32)
            transform_np[:3, 3] = bound.center
            transform_np[:3, :3] = bound.R
            transform = torch.from_numpy(transform_np).to(self.device)
            
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                            scale=scale, transform=transform).view(-1, 3)
            if self.cls_id == 0:
                ret = self.eval_points(grid_pc)
            else:
                ret = self.eval_points(grid_pc, inst_id=inst_id)
        else:
            extent = self.extent_dict[inst_id]
            extent = extent/np.max(extent/2)
            scale_np = extent / (range_dist * self.bound_extent)
            scale = torch.from_numpy(scale_np).float().to(self.device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device, scale=scale).view(-1, 3)

            ret = self.eval_points(grid_pc, inst_id=inst_id)

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
        if self.cls_id == 0 or self.n_obj==1:
            mesh.apply_transform(transform_np)

        vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        if self.cls_id == 0:
            ret = self.eval_points(vertices_pts)
        else:
            ret = self.eval_points(vertices_pts, inst_id=inst_id)
        if ret is None:
            return None
        _, color = ret
        mesh_color = color * 255
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        return mesh

    def eval_points(self, points, inst_id=None, chunk_size=500000):
        # 256^3 = 16777216
        if self.cls_id != 0:
            obj_idx = torch.from_numpy(np.array(self.inst_id_to_index[inst_id])).to(self.device)
            shape_code, texture_code = self.shape_codes(obj_idx), self.texture_codes(obj_idx)
        
        alpha, color = [], []
        n_chunks = int(np.ceil(points.shape[0] / chunk_size))
        with torch.no_grad():
            for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                embedding_k = self.pe(points[chunk_idx, ...])
                if self.cls_id == 0:
                    alpha_k, color_k = self.fc_occ_map(embedding_k)
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
