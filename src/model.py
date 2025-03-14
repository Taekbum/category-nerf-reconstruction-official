import torch
import torch.nn as nn

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


# Model
class CodeNeRF(nn.Module):
    def __init__(self,
                 emb_size1,
                 emb_size2, 
                 shape_blocks = 2, 
                 texture_blocks = 1, 
                 W = 32, 
                 latent_dim=32):
        super().__init__()
        self.shape_blocks = shape_blocks
        self.texture_blocks = texture_blocks
        self.embedding_size1 = emb_size1
        self.embedding_size2 = emb_size2
        
        self.encoding_xyz = nn.Sequential(nn.Linear(self.embedding_size1, W), nn.ReLU())
        for j in range(shape_blocks):
            layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
            setattr(self, f"shape_latent_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"shape_layer_{j+1}", layer)
        
        self.cat_layer = nn.Sequential(nn.Linear(W+self.embedding_size1,W),nn.ReLU())
        self.cat_latent_layer = nn.Sequential(nn.Linear(latent_dim,W),nn.ReLU())
        
        self.encoding_shape = nn.Linear(W,W)
        self.sigma = nn.Sequential(nn.Linear(W,1))
        self.encoding_viewdir = nn.Sequential(nn.Linear(W+self.embedding_size2, W), nn.ReLU())
        for j in range(texture_blocks):
            layer = nn.Sequential(nn.Linear(W,W), nn.ReLU())
            setattr(self, f"texture_layer_{j+1}", layer)
            layer = nn.Sequential(nn.Linear(latent_dim, W), nn.ReLU())
            setattr(self, f"texture_latent_layer_{j+1}", layer)
        self.rgb = nn.Sequential(nn.Linear(W, W//2), nn.ReLU(), nn.Linear(W//2, 3), nn.Sigmoid())
        
    def forward(self, x, shape_latent, texture_latent, 
                noise_std=None, 
                do_cat=True):
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
        
        y = self.encoding_shape(y)
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