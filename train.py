import sys
sys.path.append('/media/satassd_1/tblee-larr/CVPR24/vMap_plus_copy/src')
import loss
from scene_cateogries import *
from utils import get_transform_from_tensor
import utils
import open3d
import dataset
from functorch import vmap
import argparse
from cfg import Config
import shutil
import os
from tqdm import tqdm, trange
# from torch.utils.tensorboard import SummaryWriter

def main(args):
    # setting params
    log_dir = args.logdir
    config_file = args.config
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    events_save_dir = os.path.join(log_dir, "events")
    os.makedirs(events_save_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=events_save_dir)
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
    debug_dir = None
    if cfg.dataset_format == "ScanNet" and cfg.use_refined_mask:
        debug_dir = os.path.join(log_dir, "debug")
    data = dataset.get_dataset(cfg, debug_dir=debug_dir)
    ii = 1
    if cfg.codenerf:
        cls_id_list = [80]
    if cfg.object_wise_model:
        obj_id_to_idx = {}
    for cls_id in data.inst_dict.keys():
        if cfg.codenerf and not cls_id in cls_id_list:
            continue
        assert len(cls_dict.keys()) < cfg.max_n_models
        inst_dict_cls = data.inst_dict[cls_id]
        if cls_id == 0:
            scene_bg = sceneCategory(cfg, cls_id, inst_dict_cls, data.sample_dict, cam_info.rays_dir_cache)
            optimizer.add_param_group({"params": scene_bg.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            optimizer.add_param_group({"params": scene_bg.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            vis_dict.update({cls_id: scene_bg})
        else:
            if cfg.template_scale:
                scale_template = data.scale_template_dict[cls_id]
            else:
                scale_template = None
            scene_category = sceneCategory(cfg, cls_id, inst_dict_cls, data.sample_dict, cam_info.rays_dir_cache, 
                                           id_representative=data.id_representative_dict[cls_id], scale_template=scale_template, adaptive_obj_num=cfg.adaptive_obj_num)
            cls_dict.update({cls_id: scene_category})
            vis_dict.update({cls_id: scene_category})
            if cfg.object_wise_model:
                obj_id_to_idx[cls_id] = {}
                for iii, i in enumerate(scene_category.obj_ids):
                    obj_id_to_idx[cls_id][i] = iii
                    if not (cfg.codenerf and cls_id == 20 and cfg.fix_codenerf):
                        optimizer.add_param_group({"params": scene_category.trainer.fc_occ_map_dict[i].parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                    optimizer.add_param_group({"params": scene_category.trainer.pe_dict[i].parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
            else:
                if not (cfg.codenerf and cls_id == 20 and cfg.fix_codenerf):
                    optimizer.add_param_group({"params": scene_category.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                optimizer.add_param_group({"params": scene_category.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                optimizer.add_param_group({"params": scene_category.trainer.shape_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
                optimizer.add_param_group({"params": scene_category.trainer.texture_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
            if not cfg.fix_pose and len(list(inst_dict_cls.keys())) > 1 and not cfg.staged_poseoptim:
                for obj_id in scene_category.object_tensor_dict.keys():
                    if obj_id != data.id_representative_dict[cls_id]:
                        optimizer.add_param_group({"params": scene_category.object_tensor_dict[obj_id], "lr": cfg.pose_learning_rate, "weight_decay": cfg.pose_weight_decay})
    
    # load model if exists (resume training)
    start = 0
    ckpt_dir = os.path.join(log_dir, "ckpt")
    ckpt_dir_optimizer = os.path.join(ckpt_dir, "optimizer")
    # if os.path.isdir(ckpt_dir):
    #     print('Reloading from', ckpt_dir)
    #     # if os.path.isdir(ckpt_dir_optimizer):
    #     #     ckpt_paths = [os.path.join(ckpt_dir_optimizer, f) for f in sorted(os.listdir(ckpt_dir_optimizer))]
    #     #     ckpt_path = ckpt_paths[-1]
    #     #     ckpt = torch.load(ckpt_path)
    #     #     optimizer.load_state_dict(ckpt["optimizer"])
    #     for cls_id in vis_dict.keys():
    #         if cls_id == 0:
    #             scene_bg = vis_dict[cls_id]
    #             ckpt_dir_bg = os.path.join(ckpt_dir, str(0))
    #             if os.path.isdir(ckpt_dir_bg):
    #                 ckpt_paths = [os.path.join(ckpt_dir_bg, f) for f in sorted(os.listdir(ckpt_dir_bg))]
    #                 ckpt_path = ckpt_paths[-1]
    #                 scene_bg.load_checkpoints(ckpt_path)
    #                 start = scene_bg.start
    #         else:
    #             cls_k = vis_dict[cls_id]
    #             ckpt_dir_cls = os.path.join(ckpt_dir, str(cls_id))
    #             if os.path.isdir(ckpt_dir_cls):
    #                 ckpt_paths = [os.path.join(ckpt_dir_cls, f) for f in sorted(os.listdir(ckpt_dir_cls))]
    #                 ckpt_path = ckpt_paths[-1]
    #                 if cfg.codenerf and cls_id == 20:
    #                     cls_k.load_checkpoints(ckpt_path, load_codenerf=cfg.codenerf)
    #                 else:
    #                     cls_k.load_checkpoints(ckpt_path)
    #                 # start = cls_k.start
    #     print("start ", start)

    if cfg.training_strategy == "vmap":
        for cls_id in cls_dict.keys():
            # update_vmap_model = True
            if cfg.object_wise_model:
                for i in cls_dict[cls_id].obj_ids:
                    fc_models.append(cls_dict[cls_id].trainer.fc_occ_map_dict[i])
                    pe_models.append(cls_dict[cls_id].trainer.pe_dict[i])
            else:
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
    if cfg.training_strategy == "vmap":# and update_vmap_model == True:
        fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimizer)
        pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimizer)
        # update_vmap_model = False
    
    # train
    if cfg.same_amount:
        n_cls = len(list(cls_dict.keys()))
        n_objs = 0
        for cls_id in cls_dict.keys():
            n_objs += len(cls_dict[cls_id].obj_ids)
        n_sample_per_step = n_objs * cfg.n_per_optim // n_cls
    else:
        n_sample_per_step = cfg.n_per_optim
    
    frame_indices = np.array(list(data.sample_dict.keys()))
    update_vmap_model = False
    for iteration in trange(start+1, max_iter):
        
        
        # for revision
        if hasattr(cfg, 'it_add_obs'):
            if iteration-1 in cfg.it_add_obs[:-1]:
                data.add_additional_observation(cfg, cfg.frames[ii], iteration=cfg.it_add_obs[ii])
                if cfg.object_wise_model:
                    obj_id_to_idx = {}
                for cls_id in data.inst_dict.keys():
                    if cls_id == 0:
                        continue
                    inst_dict_cls = data.inst_dict[cls_id]
                    # if cls_id not in vis_dict.keys(): # for new subcategories if exists
                    if cls_id in cls_dict.keys():
                        if cfg.object_wise_model:
                            obj_ids = list(inst_dict_cls.keys())
                            cls_dict[cls_id].trainer.load_NeRF_dict(obj_ids)
                        trainer_copy = copy.deepcopy(cls_dict[cls_id].trainer)
                        
                    assert len(cls_dict.keys()) < cfg.max_n_models
                    scene_category = sceneCategory(cfg, cls_id, inst_dict_cls, data.sample_dict, cam_info.rays_dir_cache, 
                                                    id_representative=data.id_representative_dict[cls_id], scale_template=None, adaptive_obj_num=cfg.adaptive_obj_num)
                    if cls_id in cls_dict.keys():
                        trainer_copy.extent_dict = scene_category.extent_dict.copy()
                        trainer_copy.inst_id_to_index = scene_category.trainer.inst_id_to_index.copy()
                        scene_category.trainer = trainer_copy
                    
                    if cfg.object_wise_model:
                        obj_id_to_idx[cls_id] = {}
                        for iii, i in enumerate(scene_category.obj_ids):
                            obj_id_to_idx[cls_id][i] = iii
                    
                    if cfg.object_wise_model:
                        for i in scene_category.obj_ids:
                            if not i in cls_dict[cls_id].obj_ids:
                                optimizer.add_param_group({"params": scene_category.trainer.fc_occ_map_dict[i].parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                                optimizer.add_param_group({"params": scene_category.trainer.pe_dict[i].parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                                if cfg.training_strategy == "vmap":
                                    update_vmap_model = True
                                    fc_models.append(scene_category.trainer.fc_occ_map_dict[i])
                                    pe_models.append(scene_category.trainer.pe_dict[i])
                    elif not cls_id in cls_dict.keys():
                        optimizer.add_param_group({"params": scene_category.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                        optimizer.add_param_group({"params": scene_category.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                        optimizer.add_param_group({"params": scene_category.trainer.shape_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
                        optimizer.add_param_group({"params": scene_category.trainer.texture_codes.parameters(), "lr": cfg.code_learning_rate, "weight_decay": cfg.code_weight_decay})
                        if cfg.training_strategy == "vmap":
                            update_vmap_model = True
                            fc_models.append(scene_category.trainer.fc_occ_map)
                            pe_models.append(scene_category.trainer.pe)
                            
                    cls_dict.update({cls_id: scene_category})
                    vis_dict.update({cls_id: scene_category})

                if cfg.training_strategy == "vmap" and update_vmap_model == True:
                    fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimizer)
                    pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimizer)
                    update_vmap_model = False

                if cfg.same_amount:
                    n_cls = len(list(cls_dict.keys()))
                    n_objs = 0
                    for cls_id in cls_dict.keys():
                        n_objs += len(cls_dict[cls_id].obj_ids)
                    n_sample_per_step = n_objs * cfg.n_per_optim // n_cls
                else:
                    n_sample_per_step = cfg.n_per_optim
                    
                ii += 1
        # for revision
        
        
        # get training samples for each class
        batch_gt_depth = []
        batch_gt_rgb = []
        batch_depth_mask = []
        batch_obj_mask = []
        batch_input_pcs = []
        # batch_input_embeddings = []
        batch_sampled_z = []
        batch_indices = []
        cls_ids = []
        
        batch_shape_codes = []
        batch_texture_codes = []
        
        with performance_measure(f"Sampling over {len(cls_dict.keys())} classes,"):
            if scene_bg is not None:
                bg_gt_rgb, bg_gt_depth, bg_valid_depth_mask, bg_obj_mask, bg_input_pcs, bg_sampled_z, _ \
                    = scene_bg.get_training_samples(n_sample_per_step_bg)
                bg_input_pcs = bg_input_pcs#.to(cfg.training_device)
                bg_gt_depth = bg_gt_depth#.to(cfg.training_device)
                bg_gt_rgb = bg_gt_rgb / 255.#.to(cfg.training_device) / 255.
                bg_valid_depth_mask = bg_valid_depth_mask#.to(cfg.training_device)
                bg_obj_mask = bg_obj_mask#.to(cfg.training_device)
                bg_sampled_z = bg_sampled_z#.to(cfg.training_device)
            
            for cls_id, cls_k in cls_dict.items():
                if cfg.training_strategy == "forloop":
                    n_obj = len(cls_k.obj_ids)
                    n_sample_per_step = n_obj * cfg.n_per_optim
                if cfg.uncertainty_guided_sampling and len(cls_k.obj_ids) > 1: #cls_id in data.metric_dict.keys(): # class w/ more than 2 instances
                    if cfg.use_certain_data and not cfg.use_uncertainty:
                        mask_rate_dict = data.mask_rate_dict[cls_id]
                    else:
                        mask_rate_dict = None
                    gt_rgb, gt_depth, depth_mask, obj_mask, input_pcs, sampled_z, indices \
                        = cls_k.get_uncertainty_guided_training_samples(data.rgb_list, data.depth_list, data.obj_list, 
                            cam_info.rays_dir_cache, n_sample_per_step)
                        # = cls_k.get_uncertainty_guided_training_samples(data.rgb_list, data.depth_list, data.obj_list, 
                        #     cam_info.rays_dir_cache, data.metric_dict[cls_id], data.phi, data.theta, n_sample_per_step, mask_rate_dict=mask_rate_dict)
                else:
                    gt_rgb, gt_depth, depth_mask, obj_mask, input_pcs, sampled_z, indices \
                        = cls_k.get_training_samples(n_sample_per_step)
                
                if cfg.training_strategy == "forloop":
                    gt_rgb = gt_rgb.detach() / 255.
                    gt_depth = gt_depth.detach()
                    depth_mask = depth_mask.detach()
                    obj_mask = obj_mask.detach()
                    sampled_z = sampled_z.detach()
                
                batch_gt_depth.append(gt_depth)
                batch_gt_rgb.append(gt_rgb)
                batch_depth_mask.append(depth_mask)
                batch_obj_mask.append(obj_mask)
                batch_input_pcs.append(input_pcs)
                batch_sampled_z.append(sampled_z)
                if (cfg.use_zero_code or cfg.use_mean_code):
                    batch_indices.append(indices.view((-1)))
                else:
                    batch_indices.append(indices)
                cls_ids.append(np.array([cls_id], dtype=np.int32))
                
                # input_embeddings = cls_k.trainer.pe(input_pcs)
                # batch_input_embeddings.append(input_embeddings)
                
                if not cfg.object_wise_model:
                    if cfg.use_zero_code:
                        if n_objs > 1:
                            zero_index = torch.zeros((cfg.n_per_optim - n_sample_per_step,)).long().to(cfg.training_device)
                            shape_code_zero = cls_k.trainer.zero_codes(zero_index)
                            texture_code_zero = cls_k.trainer.zero_codes(zero_index)
                            
                            shape_code_k = cls_k.trainer.shape_codes(indices)
                            texture_code_k = cls_k.trainer.texture_codes(indices)
                            
                            shape_code_k = torch.cat([shape_code_k, shape_code_zero])[:,None,:]
                            texture_code_k = torch.cat([texture_code_k, texture_code_zero])[:,None,:]
                        else:
                            shape_code_k = cls_k.trainer.zero_codes(indices)[:,None,:]
                            texture_code_k = cls_k.trainer.zero_codes(indices)[:,None,:]
                    elif cfg.use_mean_code:
                        if n_objs > 1:
                            all_indices = torch.arange(n_objs).to(cfg.training_device)
                            shape_code_mean = torch.mean(cls_k.trainer.shape_codes(all_indices), dim=0).repeat(cfg.n_per_optim - n_sample_per_step, 1)
                            texture_code_mean = torch.mean(cls_k.trainer.texture_codes(all_indices), dim=0).repeat(cfg.n_per_optim - n_sample_per_step, 1)
                            
                            shape_code_k = cls_k.trainer.shape_codes(indices)
                            texture_code_k = cls_k.trainer.texture_codes(indices)
                            
                            shape_code_k = torch.cat([shape_code_k, shape_code_mean])[:,None,:]
                            texture_code_k = torch.cat([texture_code_k, texture_code_mean])[:,None,:]
                        else:
                            shape_code_k = cls_k.trainer.shape_codes(indices)[:,None,:]
                            texture_code_k = cls_k.trainer.texture_codes(indices)[:,None,:]
                    else:
                        shape_code_k = cls_k.trainer.shape_codes(indices)[:,None,:]
                        texture_code_k = cls_k.trainer.texture_codes(indices)[:,None,:]
                    
                    if cfg.editnerf:
                        shape_code_k = shape_code_k.repeat((1, input_pcs.shape[-2], 1))
                        texture_code_k = texture_code_k.repeat((1, input_pcs.shape[-2], 1))
                    
                    batch_shape_codes.append(shape_code_k)
                    batch_texture_codes.append(texture_code_k)
        
        if cfg.training_strategy == "vmap":
            batch_input_pcs = torch.stack(batch_input_pcs)#.to(cfg.training_device)
            batch_gt_depth = torch.stack(batch_gt_depth)#.to(cfg.training_device)
            batch_gt_rgb = torch.stack(batch_gt_rgb) / 255.#.to(cfg.training_device) / 255. # todo
            batch_depth_mask = torch.stack(batch_depth_mask)#.to(cfg.training_device)
            batch_obj_mask = torch.stack(batch_obj_mask)#.to(cfg.training_device)
            batch_sampled_z = torch.stack(batch_sampled_z)#.to(cfg.training_device)
            # batch_indices = torch.stack(batch_indices)#.to(cfg.training_device)
            if cfg.object_wise_model:
                batch_input_pcs = batch_input_pcs.squeeze(0).reshape(-1, cfg.n_per_optim, 10, 3)
                batch_gt_depth = batch_gt_depth.squeeze(0).reshape(-1, cfg.n_per_optim)
                batch_gt_rgb = batch_gt_rgb.squeeze(0).reshape(-1, cfg.n_per_optim, 3)
                batch_depth_mask = batch_depth_mask.squeeze(0).reshape(-1, cfg.n_per_optim)
                batch_obj_mask = batch_obj_mask.squeeze(0).reshape(-1, cfg.n_per_optim)
                batch_sampled_z = batch_sampled_z.squeeze(0).reshape(-1, cfg.n_per_optim, 10)
            else:
                batch_shape_codes = torch.stack(batch_shape_codes)#.to(cfg.training_device)
                batch_texture_codes = torch.stack(batch_texture_codes)#.to(cfg.training_device)
        cls_ids = np.concatenate(cls_ids, axis=0)
        
        # batch_input_embeddings = torch.stack(batch_input_embeddings)#.to(cfg.training_device)
        
        with performance_measure(f"Training over {len(cls_dict.keys())} classes,"):
            # with performance_measure(f"forward pass"):
            if cfg.training_strategy == "forloop":
                # for loop training
                batch_alpha = []
                batch_color = []
                for k, cls_id in enumerate(cls_dict.keys()):
                    cls_k = cls_dict[cls_id]
                    # embedding_k = batch_input_embeddings[k]
                    input_pcs = batch_input_pcs[k]
                    embedding_k = cls_k.trainer.pe(input_pcs)
                    shape_code_k = batch_shape_codes[k]
                    texture_code_k = batch_texture_codes[k]
                    alpha_k, color_k = cls_k.trainer.fc_occ_map(embedding_k, shape_code_k, texture_code_k)
                    batch_alpha.append(alpha_k)
                    batch_color.append(color_k)

                # batch_alpha = torch.stack(batch_alpha) # (num_cls, num_samples_per_class (m instances), 2)
                # batch_color = torch.stack(batch_color) # (num_cls, num_samples_per_class (m instances), 3, 2)
            elif cfg.training_strategy == "vmap":
                # batched training
                batch_input_embeddings = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                if cfg.object_wise_model:
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_input_embeddings)
                else:
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_input_embeddings, batch_shape_codes, batch_texture_codes)
            else:
                print("training strategy {} is not implemented ".format(cfg.training_strategy))
                exit(-1)

            # Loss
            # loss for objects : choose lower loss from (two) possible inputs for each instance
            # with performance_measure(f"Batch LOSS"):
            if cfg.training_strategy == "vmap":
                batch_loss, batch_loss_dict, batch_loss_col = \
                    loss.step_batch_loss(batch_alpha, batch_color,
                                            batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                            batch_obj_mask.detach(), batch_depth_mask.detach(),
                                            batch_sampled_z.detach())
                batch_loss_dict['cls_ids'] = cls_ids#obj_ids
            elif cfg.training_strategy == "forloop":
                batch_loss_dict = {}
                batch_loss = 0
                for i in range(len(batch_alpha)):
                    class_loss, _, _ = loss.step_batch_loss(batch_alpha[i][None, ...], batch_color[i][None, ...],
                                                batch_gt_depth[i][None, ...], batch_gt_rgb[i][None, ...],
                                                batch_obj_mask[i][None, ...], batch_depth_mask[i][None, ...],
                                                batch_sampled_z[i][None, ...])
                    batch_loss += class_loss
            
            # regularization for shape / texture code - TODO: find good configuration
            # if iteration % 8:
            # if iteration % (max_iter//len(list(data.sample_dict.keys()))):
            if not cfg.object_wise_model:
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
            
            # # log loss
            # if iteration % cfg.log_freq == 0 and cfg.training_strategy == "vmap":
            #     loss.log_loss(writer, batch_loss_dict, iteration)
            #     if scene_bg is not None:
            #         loss.log_psnr(writer, cls_ids, batch_loss_col, iteration, bg_loss_col=bg_loss_col)#obj_ids
            #     else:
            #         loss.log_psnr(writer, cls_ids, batch_loss_col, iteration)#obj_ids
        
        # update each origin model params
        # todo find a better way    # https://github.com/pytorch/functorch/issues/280
        if cfg.training_strategy == "vmap":
            with torch.no_grad():
                for model_id, (cls_id, cls_k) in enumerate(cls_dict.items()):
                    if cfg.object_wise_model:
                        for obj_id in cls_k.obj_ids:
                            for i, param in enumerate(cls_k.trainer.fc_occ_map_dict[obj_id].parameters()):
                                obj_idx = obj_id_to_idx[cls_id][obj_id]
                                param.copy_(fc_param[i][obj_idx])
                            for i, param in enumerate(cls_k.trainer.pe_dict[obj_id].parameters()):
                                obj_idx = obj_id_to_idx[cls_id][obj_id]
                                param.copy_(pe_param[i][obj_idx])
                    else:
                        for i, param in enumerate(cls_k.trainer.fc_occ_map.parameters()):
                            param.copy_(fc_param[i][model_id])
                        for i, param in enumerate(cls_k.trainer.pe.parameters()):
                            param.copy_(pe_param[i][model_id])
        
        if not cfg.fix_pose and cfg.staged_poseoptim and iteration == 5000:
            for cls_id, cls_k in cls_dict.items():
                if len(cls_k.obj_ids) > 1:
                    for obj_id in cls_k.object_tensor_dict.keys():
                        if obj_id != data.id_representative_dict[cls_id]:
                            object_tensor = cls_k.object_tensor_dict[obj_id]
                            object_tensor = Variable(object_tensor.to(cls_k.data_device), requires_grad=True)
                            optimizer.add_param_group({"params": cls_k.object_tensor_dict[obj_id], "lr": cfg.pose_learning_rate, "weight_decay": cfg.pose_weight_decay})

        # render image - support only for nocs now
        if cfg.use_nocs and iteration % cfg.img_it == 0:
            view_synthesis_dir = os.path.join(log_dir, "novel_view_synthesis")
            os.makedirs(view_synthesis_dir, exist_ok=True)
            with torch.no_grad():
                # sample_indices = np.random.choice(len(data.render_poses), size=2, replace=False)
                sample_indices = np.array([100])
                for sample_idx in sample_indices:
                    filename = os.path.join(view_synthesis_dir, 'render_{}.png'.format(sample_idx))
                    c2w = data.render_poses[sample_idx]
                    utils.render_image(c2w, vis_dict, cam_info.rays_dir_cache, cfg, filename)
        
        # saving checkpoint
        if iteration % cfg.save_it == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            # os.makedirs(ckpt_dir_optimizer, exist_ok=True)
            print(f'Saving ckpt at iteration {iteration}')
            # optimizer_path = os.path.join(ckpt_dir_optimizer, "iteration_{:05d}.pth".format(iteration))
            # torch.save({"optimizer": optimizer.state_dict()}, optimizer_path)
            for cls_id in vis_dict.keys():
                if cls_id == 0:
                    scene_bg = vis_dict[cls_id]
                    ckpt_dir_bg = os.path.join(ckpt_dir, str(0))
                    os.makedirs(ckpt_dir_bg, exist_ok=True)
                    if not cfg.object_wise_model:
                        scene_bg.save_checkpoints(ckpt_dir_bg, iteration, chamfer_dict=data.chamfer_dict, chamfer_opposite_dict=data.chamfer_opposite_dict)
                else:
                    cls_k = vis_dict[cls_id]
                    ckpt_dir_cls = os.path.join(ckpt_dir, str(cls_id))
                    os.makedirs(ckpt_dir_cls, exist_ok=True)
                    cls_k.save_checkpoints(ckpt_dir_cls, iteration)
                    
        # meshing
        if iteration % cfg.mesh_it == 0:
            obj_mesh_output = os.path.join(log_dir, "scene_mesh")
            os.makedirs(obj_mesh_output, exist_ok=True)
            # DEBUG
            # if not cfg.fix_pose:
            #     data.visualize_coords()
            for cls_id, cls_k in vis_dict.items():
                if cls_id == 0:
                    bound = cls_k.trainer.bound
                    adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim)) 
                    mesh = scene_bg.trainer.meshing(grid_dim=adaptive_grid_dim)
                    mesh.export(os.path.join(obj_mesh_output, "iteration_{}_obj{}.obj".format(iteration, str(0))))
                else: # first mesh both poses to debug
                    for obj_id in cls_k.obj_ids:
                        if len(cls_k.obj_ids) > 1:
                            if not cfg.fix_pose and len(list(inst_dict_cls.keys())) > 1:
                                object_tensor = cls_k.object_tensor_dict[obj_id]
                                T_obj = get_transform_from_tensor_sim3(object_tensor)
                                data.inst_dict[cls_id][obj_id]['T_obj'] = T_obj.detach().cpu().numpy()
                                utils.get_obb(data.inst_dict[cls_id][obj_id])
                                cls_k.trainer.extent_dict[obj_id] = data.inst_dict[cls_id][obj_id]['bbox3D'].extent
                            extent = cls_k.trainer.extent_dict[obj_id]      
                        else:
                            extent = cls_k.trainer.bound_dict[obj_id].extent
                        adaptive_grid_dim = int(np.minimum(np.max(extent)//cfg.live_voxel_size+1, cfg.grid_dim))                        
                        
                        obj_tensor = cls_k.object_tensor_dict[obj_id]
                        mesh = cls_k.trainer.meshing(obj_id, grid_dim=adaptive_grid_dim)
                        scale_np = obj_tensor[0].detach().cpu().numpy()#/self.bound_extent
                        transform_np = get_transform_from_tensor(obj_tensor[1:]).detach().cpu().numpy()

                        if mesh is None:
                            print("mesh failed obj ", obj_id)
                        else:
                            # Transform to scene coordinates
                            if cfg.use_nocs and len(cls_k.obj_ids) > 1:
                                mesh.apply_scale(scale_np)
                                mesh.apply_transform(transform_np)
                            mesh.export(os.path.join(obj_mesh_output, "iteration_{}_obj{}.obj".format(iteration, str(obj_id))))  
            print('hi')
            # # DEBUG
            # if not cfg.fix_pose:
            #     data.visualize_coords()
                 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default="./logs/debug_", type=str)
    parser.add_argument('--config', default="./configs/Replica/config_replica_room1_vMAP_device1.json", type=str)
    args = parser.parse_args()
    
    main(args)
    
    