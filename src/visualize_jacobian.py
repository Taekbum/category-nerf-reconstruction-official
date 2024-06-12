import argparse
import dataset
from scene_cateogries import *
from reconstruct import initialize_vis_dict, load_model
import render_rays
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_occupancy_jacobian(trainer, grid_pc, obj_id, out_dim=1):
    obj_idx = torch.from_numpy(np.array(trainer.inst_id_to_index[obj_id])).to(trainer.device)
    shape_code, texture_code = trainer.shape_codes(obj_idx), trainer.texture_codes(obj_idx)

    n = grid_pc.shape[0]
    input_shape_code = shape_code.expand(n, -1)
    input_shape_code = input_shape_code.unsqueeze(1).repeat(1, out_dim, 1)
    # input_shape_code.requires_grad = True # (n, out_dim, in_dim)
    grid_pc = grid_pc.unsqueeze(1).repeat(1, out_dim, 1)
    texture_code = texture_code.expand(n, -1).unsqueeze(1).repeat(1, out_dim, 1)
    
    input_shape_code.retain_grad()
    
    embedding = trainer.pe(grid_pc)
    alpha, _ = trainer.fc_occ_map(embedding, input_shape_code, texture_code)
    occ = render_rays.occupancy_activation(alpha) # (n, out_dim, out_dim)
    w = torch.eye(out_dim).view(1, out_dim, out_dim).repeat(n, 1, 1).to(trainer.device)
    occ.backward(w, retain_graph=False)
    
    return input_shape_code.grad.data.detach()
    
def visualize_jacobian(vis_dict, cfg, save_dir, grid_dim=16):
    cls_id_list = [20] #[18, 29, 76, 44, 78, 65, 91, 63, 20, 11]
    
    occ_range = [-1., 1.]
    range_dist = occ_range[1] - occ_range[0]
    for cls_id, cls_k in vis_dict.items():
        if not cls_id in cls_id_list:
            continue
        for obj_id in cls_k.obj_ids:
            extent = cls_k.trainer.extent_dict[obj_id]
            extent = extent/np.max(extent/2)
            scale_np = extent / (range_dist * cls_k.trainer.bound_extent)
            scale = torch.from_numpy(scale_np).float().to(cfg.training_device)
            grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=cfg.training_device, scale=scale).view(-1, 3)
            
            do_dc = get_occupancy_jacobian(cls_k.trainer, grid_pc, obj_id).squeeze(1)

            n_elements = 9#do_dc.shape[-1]
            subplot_rows = int(np.ceil(np.sqrt(n_elements)))
            subplot_cols = int(np.ceil(n_elements / subplot_rows))
            fig = plt.figure()
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            for code_idx in range(n_elements):
                ax = fig.add_subplot(subplot_rows, subplot_cols, code_idx + 1, projection='3d')
                ax.set_title(f'code idx {code_idx+1}')
                
                voxels = do_dc[:, code_idx].view(grid_dim, grid_dim, grid_dim).cpu().numpy()
                colors = plt.cm.viridis(voxels)
                ax.voxels(voxels, facecolors=colors, edgecolor='k')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_title('jacobian of ocde')
                sm.set_array(voxels)
                fig.colorbar(sm)  

            plt.tight_layout()
            # plt.show()
            filename = os.path.join(save_dir, f'code_jacobian_{obj_id}.png')
            plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="./logs/0624/room_0")
    parser.add_argument('--config', type=str, 
                        default="./configs/Replica/config_replica_room0_vMAP_device2.json")
    args = parser.parse_args()
    
    log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    
    cam_info = cameraInfo(cfg)
    data = dataset.get_dataset(cfg)
    
    vis_dict = {}
    initialize_vis_dict(vis_dict, data, cam_info, cfg)
    load_model(vis_dict, log_dir, cfg)
    
    visualize_jacobian(vis_dict, cfg, log_dir)