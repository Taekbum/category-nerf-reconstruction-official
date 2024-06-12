import argparse
import dataset
from scene_cateogries import *
from reconstruct import initialize_vis_dict, load_model

def code_interpolate(vis_dict, log_dir):
    cls_id_list = [80] # table #[29] # cushion
    ids = [7, 11] #[5, 72]
    shape_interpolate_dir = os.path.join(log_dir, "shape_interpolate")
    texture_interpolate_dir = os.path.join(log_dir, "texture_interpolate")
    os.makedirs(shape_interpolate_dir, exist_ok=True)
    os.makedirs(texture_interpolate_dir, exist_ok=True)
    for cls_id, cls_k in vis_dict.items():
        if not cls_id in cls_id_list:
            continue
        
        id = ids[0]
        id_other = ids[1]
        
        for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
            mesh_shape_interpolate = cls_k.trainer.meshing(inst_id=id, interpolate_mode="shape", other_id=id_other, t=t)
            # mesh_texture_interpolate = cls_k.trainer.meshing(inst_id=id, interpolate_mode="texture", other_id=id_other, t=t)
            
            mesh_shape_interpolate.export(os.path.join(shape_interpolate_dir, f"original_{id}_interpolate_{id_other}_{t}.obj"))
            # mesh_texture_interpolate.export(os.path.join(texture_interpolate_dir, f"original_{id}_interpolate_{id_other}_{t}.obj"))

def average_shape_or_code(vis_dict, log_dir):
    obj_mesh_output = os.path.join(log_dir, "scene_mesh")
    os.makedirs(obj_mesh_output, exist_ok=True)
    cls_id_list = [80] #[80] table #[29] cushion
    for cls_id, cls_k in vis_dict.items():
        # if not cls_id in cls_id_list:
        #     continue
        inst_id = cls_k.obj_ids[0] # any feasible id is OK
        mesh = cls_k.trainer.meshing(inst_id=inst_id, average_shape=True)
        # mesh = cls_k.trainer.meshing(inst_id=70, average_texture=True)
        
        if mesh is None:
            print("mesh failed cls ", cls_id)
        else:
            mesh.export(os.path.join(obj_mesh_output, "mean_shape_cls{}.obj".format(str(cls_id))))
            # mesh.export(os.path.join(obj_mesh_output, "mean_texture_cls{}_{}.obj".format(str(cls_id), str(70))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="./logs/0609")
    parser.add_argument('--config', type=str, 
                        default="./configs/Replica/config_replica_room0_vMAP_device0.json")
    parser.add_argument('--task', type=str, 
                        default="average_shape_or_code")
    args = parser.parse_args()
    
    log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    
    cam_info = cameraInfo(cfg)
    data = dataset.get_dataset(cfg)
    
    vis_dict = {}
    initialize_vis_dict(vis_dict, data, cam_info, cfg)
    load_model(vis_dict, log_dir)
    
    if args.task == "code_interpolate":
        code_interpolate(vis_dict, log_dir)
    elif args.task == "average_shape_or_code":
        average_shape_or_code(vis_dict, log_dir)