import os
import dataset
from scene_cateogries import *
from utils import get_transform_from_tensor, BoundingBox
import open3d
import trimesh
import scipy

def average_shape_or_code(cls_id, cls_k, log_dir, iteration=10000):
    obj_mesh_output = os.path.join(log_dir, "mean_mesh")
    os.makedirs(obj_mesh_output, exist_ok=True)
    cls_id_list = [80] #[80] table #[29] cushion
    # if not cls_id in cls_id_list:
    #     continue
    inst_id = cls_k.obj_ids[0] # any feasible id is OK
    mesh = cls_k.trainer.meshing(inst_id=inst_id, average_shape=True)
    # mesh = cls_k.trainer.meshing(inst_id=70, average_texture=True)
    
    if mesh is None:
        print("mesh failed cls ", cls_id)
    else:
        mesh.export(os.path.join(obj_mesh_output, f"mean_shape_cls{cls_id}_iteration_{iteration:05d}.obj"))
        # mesh.export(os.path.join(obj_mesh_output, "mean_texture_cls{}_{}.obj".format(str(cls_id), str(70))))
        print(f"save mean mesh of cls {cls_id} at iteration {iteration}")

def get_bound(obj_id, data, intrinsic_open3d, voxel_size=0.01, n_img=200):
    pcs = open3d.geometry.PointCloud()
    for idx in range(n_img):
        mask = data.obj_list[idx].squeeze() == obj_id
        depth = np.copy(data.depth_list[idx])
        twc = data.Twc[idx]
        depth[~mask] = 0
        depth = depth.transpose(1,0).astype(np.float32)
        T_CW = np.linalg.inv(twc)
        pc = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(np.asarray(depth, order="C")), intrinsic=intrinsic_open3d, extrinsic=T_CW)
        pcs += pc
    pcs = pcs.voxel_down_sample(voxel_size)
        
    try:
        transform, extents = trimesh.bounds.oriented_bounds(np.array(pcs.points))  # pc
        transform = np.linalg.inv(transform)
    except scipy.spatial._qhull.QhullError:
        print("too few pcs obj ")
        return None

    for i in range(extents.shape[0]):
        extents[i] = np.maximum(extents[i], 0.10)  # at least rendering 10cm
    extents = np.maximum(0.05, extents)
    return transform, extents

def initialize_vis_dict(vis_dict, log_dir, inst_dict, cam_info, cfg):
    cls_id_list = [18] #[18, 29, 76, 44, 78, 65, 91, 63, 20, 11]
    ckpt_dir = os.path.join(log_dir, "ckpt")
    for cls_id_str in os.listdir(ckpt_dir):
        cls_id = int(cls_id_str)
        # if not cls_id in cls_id_list:
        #     continue
        assert len(vis_dict.keys()) < cfg.max_n_models
        if cls_id == 0:
            continue
            scene_bg = sceneCategory(cfg, cls_id, inst_dict_cls, None, cam_info.rays_dir_cache, train_mode=False)
            vis_dict.update({cls_id: scene_bg})
        else:
            inst_dict_cls = inst_dict[cls_id]
            scene_category = sceneCategory(cfg, cls_id, inst_dict_cls, None, cam_info.rays_dir_cache, train_mode=False, adaptive_obj_num=cfg.adaptive_obj_num)
            vis_dict.update({cls_id: scene_category})
    return vis_dict

def mesh_all(vis_dict, log_dir, cfg, iteration=10000, mean_shape=False):
    obj_mesh_output = os.path.join(log_dir, "scene_mesh")
    os.makedirs(obj_mesh_output, exist_ok=True)
    cls_id_list = [18] #[18, 29, 76, 44, 78, 65, 91, 63, 20, 11]
    obj_id_list = []
    for cls_id, cls_k in vis_dict.items():
        # if not cls_id in cls_id_list:
        #     continue
        if cls_id == 0:
            continue
            bound = cls_k.trainer.bound
            adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim)) 
            mesh = vis_dict[0].trainer.meshing(grid_dim=adaptive_grid_dim)
            mesh.export(os.path.join(obj_mesh_output, f"iteration_{iteration:05d}_obj{obj_id}.obj"))
            # mesh.export(os.path.join(obj_mesh_output, "meshonly_obj{}.obj".format(str(0))))
        elif mean_shape:
            average_shape_or_code(cls_id, cls_k, log_dir, iteration=iteration)
        else: # first mesh both poses to debug
            for obj_id in cls_k.obj_ids:
                # if not obj_id in obj_id_list:
                #     continue
                if cfg.adaptive_obj_num or len(cls_k.obj_ids) > 1:
                    extent = cls_k.trainer.extent_dict[obj_id]
                else:
                    extent = cls_k.trainer.bound_dict[obj_id].extent
                adaptive_grid_dim = int(np.minimum(np.max(extent)//cfg.live_voxel_size+1, cfg.grid_dim))     
                        
                obj_tensor = cls_k.object_tensor_dict[obj_id]
                mesh = cls_k.trainer.meshing(inst_id=obj_id, grid_dim=adaptive_grid_dim, average_shape=False, average_texture=False) # for codenerf init weight test
                scale_np = obj_tensor[0].detach().cpu().numpy()#/bound_extent
                transform_np = get_transform_from_tensor(obj_tensor[1:]).detach().cpu().numpy()
                    
                # Transform to scene coordinates
                if not cfg.use_mean_code:
                    mesh.apply_scale(scale_np)
                    mesh.apply_transform(transform_np)
                
                if mesh is None:
                    print("mesh failed obj ", obj_id)
                else:
                    mesh.export(os.path.join(obj_mesh_output, f"iteration_{iteration:05d}_obj{obj_id}.obj"))
                    # mesh.export(os.path.join(obj_mesh_output, "meshonly_obj{}.obj".format(str(obj_id))))
                    print(f"save mesh of {obj_id} at iteration {iteration}")
                    
def load_model(vis_dict, log_dir, cfg, iteration=10000):
    ckpt_dir = os.path.join(log_dir, "ckpt")
    if os.path.isdir(ckpt_dir):
        print('Reloading from', ckpt_dir)
        cls_id_list = [18] #[18, 29, 76, 44, 78, 65, 91, 63, 20, 11]
        for cls_id in vis_dict.keys():
            cls_id = int(cls_id)
            # if not cls_id in cls_id_list:
            #     continue
            if cls_id == 0:
                continue
                scene_bg = vis_dict[cls_id]
                ckpt_dir_bg = os.path.join(ckpt_dir, str(0))
                if os.path.isdir(ckpt_dir_bg):
                    ckpt_paths = [os.path.join(ckpt_dir_bg, f) for f in sorted(os.listdir(ckpt_dir_bg))]
                    ckpt_path = ckpt_paths[-1]
                    scene_bg.load_checkpoints(ckpt_path)
            else:
                cls_k = vis_dict[cls_id]
                ckpt_dir_cls = os.path.join(ckpt_dir, str(cls_id))
                if os.path.isdir(ckpt_dir_cls):
                    ckpt_paths = [os.path.join(ckpt_dir_cls, f) for f in sorted(os.listdir(ckpt_dir_cls))]
                    # ckpt_path = ckpt_paths[-1]
                    ckpt_path = os.path.join(ckpt_dir_cls, f"cls_{cls_id}_iteration_{iteration:05d}.pth")
                    cls_k.load_checkpoints(ckpt_path, load_codenerf=cfg.codenerf)

def load_inst_dict(ckpt_dir, intrinsic_open3d, iteration=10000, data=None):
    inst_dict = {}
    cls_ids = [int(cls) for cls in os.listdir(ckpt_dir) if int(cls) != 0]
    cls_id_list = [18]
    for cls_id in cls_ids:
        if cls_id == 0:
            continue
        # if not cls_id in cls_id_list:
        #     continue
        inst_dict[cls_id] = {}
        ckpt_file = os.path.join(ckpt_dir, str(cls_id), f'cls_{cls_id}_iteration_{iteration:05d}.pth') # 10000
        ckpt = torch.load(ckpt_file)
        obj_ids = ckpt['obj_tensor_dict'].keys()
        for obj_id in obj_ids:
            inst_dict[cls_id][obj_id] = {}      
            if ckpt['bound'] is None:
                transform, extents = get_bound(obj_id, data, intrinsic_open3d)
            else:
                extents = ckpt['bound'][obj_id]
                transform = get_transform_from_tensor(ckpt['obj_tensor_dict'][obj_id][1:]).cpu().numpy()
            inst_dict[cls_id][obj_id]['T_obj'] = np.copy(transform)
            # scale = np.linalg.det(transform[:3,:3])**(1/3)
            # transform[:3,:3] /= scale
            bbox3D = BoundingBox()
            bbox3D.center = transform[:3, 3]
            bbox3D.R = transform[:3, :3]
            bbox3D.extent = extents
            inst_dict[cls_id][obj_id]['bbox3D'] = bbox3D
                     
    return inst_dict
                    
def main(args):
    log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    mean_shape = cfg.use_mean_code
    
    cam_info = cameraInfo(cfg)
    # data = dataset.get_dataset(cfg, train_mode=False)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)
    
    mesh_iterations = np.arange(0, cfg.max_iter-1, cfg.save_it) + cfg.save_it
    for i in range(mesh_iterations.shape[0]):
        iteration = mesh_iterations[i]
        # for candidate_1_10000_2
        if not ((iteration > 3000 and iteration < 3100) or (iteration > 6000 and iteration < 6100)): continue
        # if log_dir.split('/')[-1] == 'candidate_1_10000':
        #     if cfg.use_mean_code and iteration < 2000: continue
        #     if not cfg.use_mean_code and iteration < 6200: continue

        vis_dict = {}
        ckpt_dir = os.path.join(log_dir, "ckpt")
        inst_dict = load_inst_dict(ckpt_dir, intrinsic_open3d, iteration=iteration)
        vis_dict = initialize_vis_dict(vis_dict, log_dir, inst_dict, cam_info, cfg)
        load_model(vis_dict, log_dir, cfg, iteration=iteration)
        mesh_all(vis_dict, log_dir, cfg, iteration=iteration, mean_shape=mean_shape)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, 
                        default="/media/satassd_1/tblee-larr/CVPR24/vMap_plus_copy/logs/0816/Replica/random_batch_for_comp/1/room_0")
    parser.add_argument('--config', type=str, 
                        default="/media/satassd_1/tblee-larr/CVPR24/vMap_plus_copy/configs/Replica/1129/config_replica_align_room0_vMAP_use_random_batch_device0.json")
    # parser.add_argument('--config', type=str, 
    #                     default="./configs/ScanNet/config_scannet0000_vMAP.json")
    # parser.add_argument('--ids', type=int, nargs="+", default=[73, 74])
    args = parser.parse_args()
    
    main(args)