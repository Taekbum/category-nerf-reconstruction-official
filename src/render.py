import os
import dataset
from scene_cateogries import *
from reconstruct import initialize_vis_dict, load_model
import utils
import time
                    
def main(args):
    log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    
    cam_info = cameraInfo(cfg)
    data = dataset.get_dataset(cfg)
    
    vis_dict = {}
    initialize_vis_dict(vis_dict, data, cam_info, cfg)
    load_model(vis_dict, log_dir)
    
    view_synthesis_dir = os.path.join(log_dir, "novel_view_synthesis")
    os.makedirs(view_synthesis_dir, exist_ok=True)
    with torch.no_grad():
        # sample_indices = np.random.choice(len(data.render_poses), size=2, replace=False)
        sample_indices = np.array([100])
        t1 = time.time()
        for sample_idx in sample_indices:
            filename = os.path.join(view_synthesis_dir, 'render_{}.png'.format(sample_idx))
            c2w = data.render_poses[sample_idx]
            utils.render_image(c2w, vis_dict, cam_info.rays_dir_cache, cfg, filename)
        t2 = time.time()
        print(f"rendering time : {(t2-t1)/len(sample_indices)} sec / img")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_indices', nargs='+', required=True)
    parser.add_argument('--logdir', type=str, default="./logs/0522/room_1")
    parser.add_argument('--config', type=str, 
                        default="./configs/Replica/config_replica_room1_vMAP_device1.json")
    args = parser.parse_args()
    
    main(args)