import loss
from scene_cateogries import *
from utils import get_transform_from_tensor
import utils
import dataset
from functorch import vmap
import argparse
from cfg import Config
import shutil
import os
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

def main(args):
    # setting params
    log_dir = args.logdir
    config_file = args.config
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    events_save_dir = os.path.join(log_dir, "events")
    os.makedirs(events_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=events_save_dir)
    n_sample_per_step_bg = cfg.n_per_optim_bg
    max_iter = cfg.max_iter
    
    # set camera
    cam_info = cameraInfo(cfg)
    
    # init cls_dict
    cls_dict = {}   # only objs
    vis_dict = {}   # including bg
    
    # init for training
    optimizer = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # init vmap
    # fc_models = []
    fc_models, pe_models = [], []
    scene_bg = None
    
    # parse dataset
    data = dataset.get_dataset(cfg)
    for cls_id in data.inst_dict.keys():
        assert len(cls_dict.keys()) < cfg.max_n_models
        inst_dict_cls = data.inst_dict[cls_id]
        if cls_id == 0:
            scene_bg = sceneCategory(cfg, cls_id, inst_dict_cls, data.sample_dict, cam_info.rays_dir_cache)
            optimizer.add_param_group({"params": scene_bg.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            optimizer.add_param_group({"params": scene_bg.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            vis_dict.update({cls_id: scene_bg})
        else:
            scene_category = sceneCategory(cfg, cls_id, inst_dict_cls, data.sample_dict, cam_info.rays_dir_cache)
            cls_dict.update({cls_id: scene_category})
            vis_dict.update({cls_id: scene_category})
            optimizer.add_param_group({"params": scene_category.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            optimizer.add_param_group({"params": scene_category.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            optimizer.add_param_group({"params": scene_category.trainer.shape_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
            optimizer.add_param_group({"params": scene_category.trainer.texture_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
                
    # load model if exists (resume training)
    start = 0
    ckpt_dir = os.path.join(log_dir, "ckpt")

    for cls_id in cls_dict.keys():
        fc_models.append(cls_dict[cls_id].trainer.fc_occ_map)
        pe_models.append(cls_dict[cls_id].trainer.pe)
    
    # ###################################
    # # measure trainable params in total
    # total_params = 0
    # for cls_id in cls_dict.keys():
    #     cls_k = cls_dict[cls_id]
    #     for p in cls_k.trainer.fc_occ_map.parameters():
    #         if p.requires_grad:
    #             total_params += p.numel()
    #     for p in cls_k.trainer.pe.parameters():
    #         if p.requires_grad:
    #             total_params += p.numel()
    # print("total param ", total_params)
    
    # add vmap
    fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimizer)
    pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimizer)
    
    # train
    n_cls = len(list(cls_dict.keys()))
    n_objs = 0
    for cls_id in cls_dict.keys():
        n_objs += len(cls_dict[cls_id].obj_ids)
    n_sample_per_step = n_objs * cfg.n_per_optim // n_cls

    for iteration in trange(start+1, max_iter):
        # get training samples for each class
        batch_gt_depth = []
        batch_gt_rgb = []
        batch_depth_mask = []
        batch_obj_mask = []
        batch_input_pcs = []
        batch_sampled_z = []
        batch_indices = []
        cls_ids = []
        
        batch_shape_codes = []
        batch_texture_codes = []
        
        # with performance_measure(f"Sampling over {len(cls_dict.keys())} classes,"):
        if scene_bg is not None:
            bg_gt_rgb, bg_gt_depth, bg_valid_depth_mask, bg_obj_mask, bg_input_pcs, bg_sampled_z, _ \
                = scene_bg.get_training_samples(n_sample_per_step_bg)
            bg_input_pcs = bg_input_pcs
            bg_gt_depth = bg_gt_depth
            bg_gt_rgb = bg_gt_rgb / 255.
            bg_valid_depth_mask = bg_valid_depth_mask
            bg_obj_mask = bg_obj_mask
            bg_sampled_z = bg_sampled_z
        
        for cls_id, cls_k in cls_dict.items():
            gt_rgb, gt_depth, depth_mask, obj_mask, input_pcs, sampled_z, indices \
                = cls_k.get_training_samples(n_sample_per_step)
            
            batch_gt_depth.append(gt_depth)
            batch_gt_rgb.append(gt_rgb)
            batch_depth_mask.append(depth_mask)
            batch_obj_mask.append(obj_mask)
            batch_input_pcs.append(input_pcs)
            batch_sampled_z.append(sampled_z)
            batch_indices.append(indices)
            cls_ids.append(np.array([cls_id], dtype=np.int32))
            
            shape_code_k = cls_k.trainer.shape_codes(indices)[:,None,:]
            texture_code_k = cls_k.trainer.texture_codes(indices)[:,None,:]
            
            batch_shape_codes.append(shape_code_k)
            batch_texture_codes.append(texture_code_k)
        
        batch_input_pcs = torch.stack(batch_input_pcs)
        batch_gt_depth = torch.stack(batch_gt_depth)
        batch_gt_rgb = torch.stack(batch_gt_rgb) / 255.
        batch_depth_mask = torch.stack(batch_depth_mask)
        batch_obj_mask = torch.stack(batch_obj_mask)
        batch_sampled_z = torch.stack(batch_sampled_z)
        batch_shape_codes = torch.stack(batch_shape_codes)
        batch_texture_codes = torch.stack(batch_texture_codes)
        cls_ids = np.concatenate(cls_ids, axis=0)
        
        # with performance_measure(f"Training over {len(cls_dict.keys())} classes,"):
        # batched training
        batch_input_embeddings = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
        batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_input_embeddings, batch_shape_codes, batch_texture_codes)

        # Loss
        batch_loss, batch_loss_dict, batch_loss_col = \
            loss.step_batch_loss(batch_alpha, batch_color,
                                    batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                    batch_obj_mask.detach(), batch_depth_mask.detach(),
                                    batch_sampled_z.detach())
        batch_loss_dict['cls_ids'] = cls_ids

        reg_scaling=0.0005
        reg_loss_shape, reg_loss_texture = loss.step_batch_loss_reg(cls_dict, torch.from_numpy(cls_ids).to(cfg.training_device))#obj_ids
        batch_loss += reg_scaling * (reg_loss_shape + reg_loss_texture).sum()
        batch_loss_dict['reg_shape'] = reg_loss_shape
        batch_loss_dict['reg_texture'] = reg_loss_texture
        
        # loss for background
        if scene_bg is not None:
            bg_embedding = scene_bg.trainer.pe(bg_input_pcs)
            bg_alpha, bg_color = scene_bg.trainer.fc_occ_map(bg_embedding)
            bg_loss, bg_loss_dict, bg_loss_col = loss.step_batch_loss(bg_alpha[None, ...], bg_color[None, ...],
                                                bg_gt_depth[None, ...].detach(), bg_gt_rgb[None, ...].detach(),
                                                bg_obj_mask[None, ...].detach(), bg_valid_depth_mask[None, ...].detach(),
                                                bg_sampled_z[None, ...].detach())
            batch_loss += bg_loss
            batch_loss_dict['background'] = bg_loss_dict
        # with performance_measure(f"Backward"):
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # log loss
        if iteration % cfg.log_iter == 0:
            loss.log_loss(writer, batch_loss_dict, iteration)
            if scene_bg is not None:
                loss.log_psnr(writer, cls_ids, batch_loss_col, iteration, bg_loss_col=bg_loss_col)
            else:
                loss.log_psnr(writer, cls_ids, batch_loss_col, iteration)
        
        # update each origin model params
        # todo find a better way    # https://github.com/pytorch/functorch/issues/280
        with torch.no_grad():
            for model_id, (cls_id, cls_k) in enumerate(cls_dict.items()):
                for i, param in enumerate(cls_k.trainer.fc_occ_map.parameters()):
                    param.copy_(fc_param[i][model_id])
                for i, param in enumerate(cls_k.trainer.pe.parameters()):
                    param.copy_(pe_param[i][model_id])
        
        # saving checkpoint
        if iteration % cfg.save_it == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            print(f'Saving ckpt at iteration {iteration}')       
            for cls_id in vis_dict.keys():
                cls_k = vis_dict[cls_id]
                ckpt_dir_cls = os.path.join(ckpt_dir, str(cls_id))
                os.makedirs(ckpt_dir_cls, exist_ok=True)
                cls_k.save_checkpoints(ckpt_dir_cls, iteration)
                    
        # meshing
        if iteration % cfg.mesh_it == 0:
            obj_mesh_output = os.path.join(log_dir, "scene_mesh")
            os.makedirs(obj_mesh_output, exist_ok=True)
            for cls_id, cls_k in vis_dict.items():
                if cls_id == 0:
                    bound = cls_k.trainer.bound
                    adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim)) 
                    mesh = scene_bg.trainer.meshing(grid_dim=adaptive_grid_dim)
                    mesh.export(os.path.join(obj_mesh_output, "iteration_{}_obj{}.obj".format(iteration, str(0))))
                else:
                    for obj_id in cls_k.obj_ids:
                        if len(cls_k.obj_ids) > 1:
                            extent = cls_k.trainer.extent_dict[obj_id]      
                        else:
                            extent = cls_k.trainer.bound_dict[obj_id].extent
                        adaptive_grid_dim = int(np.minimum(np.max(extent)//cfg.live_voxel_size+1, cfg.grid_dim))                        
                        
                        obj_tensor = cls_k.object_tensor_dict[obj_id]
                        mesh = cls_k.trainer.meshing(obj_id, grid_dim=adaptive_grid_dim)
                        scale_np = obj_tensor[0].detach().cpu().numpy()
                        transform_np = get_transform_from_tensor(obj_tensor[1:]).detach().cpu().numpy()

                        if mesh is None:
                            print("mesh failed obj ", obj_id)
                        else:
                            # Transform to scene coordinates
                            if len(cls_k.obj_ids) > 1:
                                mesh.apply_scale(scale_np)
                                mesh.apply_transform(transform_np)
                            mesh.export(os.path.join(obj_mesh_output, "iteration_{}_obj{}.obj".format(iteration, str(obj_id))))  
                 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default="./logs/commit1/Replica/room_0", type=str)
    parser.add_argument('--config', default="./configs/Replica/config_replica_room0.json", type=str)
    args = parser.parse_args()
    
    main(args)
    
    