import torch
import render_rays
import torch.nn.functional as F
import numpy as np

def step_batch_loss_reg(cls_dict, cls_ids):#obj_ids
    loss_reg_shape = torch.zeros(cls_ids.shape[0]).to(cls_ids.device)
    loss_reg_texture = torch.zeros_like(loss_reg_shape)
    for idx, cls_k in enumerate(cls_dict.values()):
        obj_ids = torch.arange(cls_k.trainer.n_obj).to(cls_k.training_device)
        shape_codes = cls_k.trainer.shape_codes(obj_ids)
        texture_codes = cls_k.trainer.texture_codes(obj_ids)
        if len(cls_k.obj_ids) > 1:
            loss_reg_shape[idx] = torch.norm(shape_codes, dim=-1).sum()
            loss_reg_texture[idx] = torch.norm(texture_codes, dim=-1).sum()
    return loss_reg_shape, loss_reg_texture


def step_batch_loss(alpha, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0):
    """
    SLIGHTLY CHANGED FROM ORIGINAL vMAP CODE
    """
    
    """
    apply depth where depth are valid                                       -> mask_depth
    apply depth, color loss on this_obj                                     -> mask_obj
    apply occupancy/opacity loss on this_obj & background                   -> mask_sem

    output:
    loss for training
    loss_all for per sample, could be used for active sampling, replay buffer
    """
    mask_obj = sem_labels != 0
    mask_obj = mask_obj.detach()
    mask_sem = sem_labels != 2
    mask_sem = mask_sem.detach()

    alpha = alpha.squeeze(dim=-1)
    color = color.squeeze(dim=-1)

    occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]

    render_depth = render_rays.render(termination, z_vals)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    var = render_rays.render(termination, diff_sq).detach()  # must detach here!
    render_color = render_rays.render(termination[..., None], color, dim=-2)
    render_opacity = torch.sum(termination, dim=-1)     # similar to obj-nerf opacity loss
    
    # 2D depth loss: only on valid depth & mask
    # [mask_depth & mask_obj]
    loss_depth_raw = render_rays.render_loss(render_depth, gt_depth, loss="L1", normalise=False)
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    # [mask_obj]
    loss_col_raw = render_rays.render_loss(render_color, gt_color, loss="L1", normalise=False)
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)# / 3.
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)

    # 2D occupancy/opacity loss: apply except unknown area
    # [mask_sem]
    loss_opacity_raw = render_rays.render_loss(render_opacity, mask_obj.float(), loss="L1", normalise=False)
    loss_opacity = torch.mul(loss_opacity_raw, mask_sem)  # but ignore -1 unkown area e.g., mask edges   # todo var
    loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem)   # todo var

    # loss for bp
    l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling
    loss = l_batch.sum()
    
    loss_dict = {'depth': loss_depth, 'color': loss_col, 'opacity': loss_opacity}
    
    return loss, loss_dict, loss_col

def log_loss(writer, batch_loss_dict, iteration):
    cls_ids = batch_loss_dict['cls_ids']
    
    if 'background' in batch_loss_dict.keys():
        background_loss = batch_loss_dict['background']
        for key in background_loss.keys():
            loss = background_loss[key]
            writer.add_scalar('background/'+key, loss, iteration)
    
    for key in batch_loss_dict.keys():
        if key == 'background' or key == 'cls_ids':
            continue
        losses = batch_loss_dict[key]
        for i in range(losses.shape[0]):
            cls_id = cls_ids[i]#.item()
            loss = losses[i].item()
            writer.add_scalar('cls_'+str(cls_id)+'/'+key, loss, iteration)
            
def log_psnr(writer, cls_ids, batch_loss_col, iteration, bg_loss_col=None):#obj_ids
    if bg_loss_col is not None:
        psnr_bg = -10*np.log(bg_loss_col.item()) / np.log(10)
        writer.add_scalar('background/psnr', psnr_bg, iteration)
    for i in range(cls_ids.shape[0]):
        cls_id = cls_ids[i]#.item()
        loss_col = batch_loss_col[i].item()
        psnr = -10*np.log(loss_col) / np.log(10)
        writer.add_scalar('cls_'+str(cls_id)+'/psnr', psnr, iteration)