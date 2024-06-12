"""
from CodeNeRF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_fn(m.weight)
        
def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

def PE(x, degree, obj_scale=1.0):
    x = x / obj_scale
    y = torch.cat([2.**i * x for i in range(degree)], -1)
    w = 1
    return torch.cat([x] + [torch.sin(y) * w, torch.cos(y) * w], -1)

# class CodeNeRF(nn.Module):
#     def __init__(self, shape_blocks = 2, texture_blocks = 1, W = 256, 
#                  num_xyz_freq = 10, latent_dim=256,
#                  use_semantic=False, num_semantic_classes=3):
#         super().__init__()
#         self.shape_blocks = shape_blocks
#         self.texture_blocks = texture_blocks
#         self.num_xyz_freq = num_xyz_freq
#         self.use_semantic = use_semantic
        
#         d_xyz = 3 + 6 * num_xyz_freq
#         self.encoding_xyz = nn.Sequential(nn.Linear(d_xyz, W), nn.ReLU())
#         for j in range(shape_blocks):
#             layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
#             setattr(self, f"shape_latent_layer_{j+1}", layer)
#             layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
#             setattr(self, f"shape_layer_{j+1}", layer)
#         self.encoding_shape = nn.Linear(W,W)
#         self.sigma = nn.Sequential(nn.Linear(W,1))#, nn.Softplus())
#         self.encoding_viewdir = nn.Sequential(nn.Linear(W, W), nn.ReLU())
#         for j in range(texture_blocks):
#             layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
#             setattr(self, f"texture_layer_{j+1}", layer)
#             layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
#             setattr(self, f"texture_latent_layer_{j+1}", layer)
#         self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3), nn.Sigmoid())
#         if use_semantic:
#             self.semantic = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, num_semantic_classes))
        
#     def forward(self, xyz, shape_latent, texture_latent):#, obj_scale):
#         # xyz = PE(xyz, self.num_xyz_freq, obj_scale)
#         y = self.encoding_xyz(xyz)
#         for j in range(self.shape_blocks):
#             z = getattr(self, f"shape_latent_layer_{j+1}")(shape_latent)
#             y = y + z
#             y = getattr(self, f"shape_layer_{j+1}")(y)
#         y = self.encoding_shape(y)
#         sigmas = self.sigma(y)
#         y = self.encoding_viewdir(y)
#         for j in range(self.texture_blocks):
#             z = getattr(self, f"texture_latent_layer_{j+1}")(texture_latent)
#             y = y + z
#             y = getattr(self, f"texture_layer_{j+1}")(y)
#         rgbs = self.rgb(y)
#         if self.use_semantic:
#             semantics = self.semantic(y)
#             return sigmas, rgbs, semantics
#         else:
#             return sigmas, rgbs

class StyleMLP(nn.Module):
    def __init__(self, style_dim=8, embed_dim=128, style_depth=1):
        super().__init__()
        self.activation = F.relu

        lin_block = nn.Linear
        first_block = nn.Linear(style_dim, embed_dim)
        self.mlp = nn.ModuleList([first_block] + [lin_block(embed_dim, embed_dim) for _ in range(style_depth - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.mlp):
            x = self.activation(layer(x))
        return x

# Model


class EditNeRF(nn.Module):
    def __init__(self,
                 emb_size1,
                 emb_size2,
                 shape_blocks = 2, 
                 texture_blocks = 1, 
                 W = 256,
                 num_xyz_freq=10,
                 latent_dim=256,
                 use_semantic=False, 
                 num_semantic_classes=3,
                 shared_shape=True,
                 do_cat=True): #W_bottleneck=8, input_ch=3, input_ch_views=3, output_ch=4, style_dim=64, embed_dim=128, style_depth=1, use_styles=True, separate_codes=True, use_viewdirs=True, **kwargs):
        super(EditNeRF, self).__init__()

        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.embedding_size1 = emb_size1
        self.embedding_size2 = emb_size2
        self.use_semantic = use_semantic
        self.shared_shape = shared_shape
        
        if shared_shape:
            mean_layer = [nn.Sequential(nn.Linear(self.embedding_size1, W), nn.ReLU())]
            for j in range(shape_blocks):
                mean_layer.append(nn.Sequential(nn.Linear(W,W), nn.ReLU()))
            self.mean_network = nn.Sequential(*mean_layer)

        instance_layer = [nn.Sequential(nn.Linear(self.embedding_size1+W, W), nn.ReLU())]
        for j in range(shape_blocks):
            instance_layer.append(nn.Sequential(nn.Linear(W,W), nn.ReLU()))
        self.instance_network = nn.Sequential(*instance_layer)
        
        self.fusion_network = nn.Sequential(nn.Linear(W,W),nn.ReLU()) # TODO:depth 2?

        self.sigma_linear = nn.Sequential(nn.Linear(W,1))
        self.encoding_viewdir = nn.Sequential(nn.Linear(W+self.embedding_size2, W), nn.ReLU())
        
        self.rgb_network = nn.Sequential(*[nn.Sequential(nn.Linear(2*W, W), nn.ReLU())],
                                                  *[nn.Sequential(nn.Linear(W, W), nn.ReLU()) for _ in range(texture_blocks - 1)])
        self.rgb_linear = nn.Linear(W, 3)
        if use_semantic:
            self.semantic = nn.Sequential(nn.Linear(W, num_semantic_classes))
            
        self.style_linears = nn.ModuleList([StyleMLP(latent_dim, W, 1) for _ in range(2)])

    def forward(self, x, shape_latent, texture_latent, mean_shape=None, noise_std=None):
        if mean_shape:
            y = self.mean_network(x[...,:self.embedding_size1])
        else:
            y = torch.cat([x[...,:self.embedding_size1], self.style_linears[0](shape_latent)], dim=-1)
            y = self.instance_network(y)
            if self.shared_shape:
                y_mean = self.mean_network(x[...,:self.embedding_size1])
                y = y + y_mean

        y = self.fusion_network(y)
        raw = self.sigma_linear(y)
        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        sigmas = raw * 10.
        
        y = torch.cat([y, x[...,self.embedding_size1:]], dim=-1)
        y = self.encoding_viewdir(y)
        y = torch.cat([y, self.style_linears[1](texture_latent)], dim=-1)
        y = self.rgb_network(y)
        
        rgbs = self.rgb_linear(y)
        if self.use_semantic:
            semantics = self.semantic(y)
            return sigmas, rgbs, semantics
        else:
            return sigmas, rgbs

class CodeNeRF(nn.Module):
    def __init__(self,
                 emb_size1,
                 emb_size2, 
                 shape_blocks = 2, 
                 texture_blocks = 1, 
                 W = 256, 
                 num_xyz_freq=10,
                 latent_dim=256,
                 use_semantic=False, 
                 num_semantic_classes=3):
        super().__init__()
        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.embedding_size1 = emb_size1
        self.embedding_size2 = emb_size2
        self.use_semantic = use_semantic
        
        self.encoding_xyz = nn.Sequential(nn.Linear(self.embedding_size1, W), nn.ReLU())
        for j in range(shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
            setattr(self, f"shape_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"shape_layer_{j+1}", layer)
        
        self.cat_layer = nn.Sequential(nn.Linear(W+self.embedding_size1,W),nn.ReLU())
        self.cat_latent_layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
        
        self.encoding_shape = nn.Linear(W,W)
        self.sigma = nn.Sequential(nn.Linear(W,1))#, nn.Softplus())
        self.encoding_viewdir = nn.Sequential(nn.Linear(W+self.embedding_size2, W), nn.ReLU())
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"texture_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3), nn.Sigmoid()) # TODO: reduce 1 layer here?
        if use_semantic:
            self.semantic = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, num_semantic_classes))
        
    def forward(self, x, shape_latent, texture_latent, 
                noise_std=None, 
                do_cat=True):#, obj_scale):
        y = self.encoding_xyz(x[...,:self.embedding_size1])
        for j in range(self.shape_blocks):
            if do_cat and j==1:
                z = self.cat_latent_layer(shape_latent)
                y = y + z
                y = torch.cat((y, x[...,:self.embedding_size1]), dim=-1)
                y = self.cat_layer(y)
            z = getattr(self, f"shape_latent_layer_{j+1}")(shape_latent)
            y = y + z    
            y = getattr(self, f"shape_layer_{j+1}")(y)
        
        y = self.encoding_shape(y) # TODO: comment this to get fair model
        raw = self.sigma(y)
        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        sigmas = raw * 10.
        
        y = torch.cat((y, x[..., self.embedding_size1:]), dim=-1)
        y = self.encoding_viewdir(y)
        for j in range(self.texture_blocks):
            z = getattr(self, f"texture_latent_layer_{j+1}")(texture_latent)
            y = y + z
            y = getattr(self, f"texture_layer_{j+1}")(y)
        rgbs = self.rgb(y)
        if self.use_semantic:
            semantics = self.semantic(y)
            return sigmas, rgbs, semantics
        else:
            return sigmas, rgbs
        
class OccupancyMap(torch.nn.Module):
    def __init__(
        self,
        emb_size1,
        emb_size2,
        hidden_size=256,
        do_color=True,
        hidden_layers_block=1
    ):
        super(OccupancyMap, self).__init__()
        self.do_color = do_color
        self.embedding_size1 = emb_size1
        self.in_layer = fc_block(self.embedding_size1, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)
        # self.embedding_size2 = 21*(5+1)+3 - self.embedding_size # 129-66=63 32
        self.embedding_size2 = emb_size2
        self.cat_layer = fc_block(
            hidden_size + self.embedding_size1, hidden_size)

        # self.cat_layer = fc_block(
        #     hidden_size , hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        if self.do_color:
            self.color_linear = fc_block(self.embedding_size2 + hidden_size, hidden_size)
            self.out_color = torch.nn.Linear(hidden_size, 3)

        # self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x,
                noise_std=None,
                do_alpha=True,
                do_color=True,
                do_cat=True):
        fc1 = self.in_layer(x[...,:self.embedding_size1])
        fc2 = self.mid1(fc1)
        # fc3 = self.cat_layer(fc2)
        if do_cat:
            fc2_x = torch.cat((fc2, x[...,:self.embedding_size1]), dim=-1)
            fc3 = self.cat_layer(fc2_x)
        else:
            fc3 = fc2
        fc4 = self.mid2(fc3)

        alpha = None
        if do_alpha:
            raw = self.out_alpha(fc4)   # todo ignore noise
            if noise_std is not None:
                noise = torch.randn(raw.shape, device=x.device) * noise_std
                raw = raw + noise

            # alpha = self.relu(raw) * scale    # nerf
            alpha = raw * 10. #self.scale     # unisurf

        color = None
        if self.do_color and do_color:
            fc4_cat = self.color_linear(torch.cat((fc4, x[..., self.embedding_size1:]), dim=-1))
            raw_color = self.out_color(fc4_cat)
            color = self.sigmoid(raw_color)

        return alpha, color