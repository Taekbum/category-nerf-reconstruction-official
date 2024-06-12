from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import dataset
from scene_cateogries import *
from reconstruct import initialize_vis_dict, load_model

colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]

def plot_tsne(vis_dict, log_dir):
    cls_id_list = [80]
    shape_tsne_dir = os.path.join(log_dir, "tsne", "shape")
    texture_tsne_dir = os.path.join(log_dir, "tsne", "texture")
    os.makedirs(shape_tsne_dir, exist_ok=True)
    os.makedirs(texture_tsne_dir, exist_ok=True)
    for cls_id, cls_k in vis_dict.items():
        if not cls_id in cls_id_list:
            continue
        n_obj = cls_k.trainer.n_obj
        latent_dim = cls_k.trainer.net_hyperparams['latent_dim']
        if n_obj < 2:
            continue
        shape_code_data = np.zeros((n_obj, latent_dim))
        texture_code_data = np.zeros((n_obj, latent_dim))
        obj_ids = []
        shape_codes = cls_k.trainer.shape_codes
        texture_codes = cls_k.trainer.texture_codes
        for obj_id in cls_k.obj_ids:
            idx = torch.from_numpy(np.array([cls_k.trainer.inst_id_to_index[obj_id]])).to(cls_k.training_device)
            shape_code = shape_codes(idx).detach().cpu().numpy()
            texture_code = texture_codes(idx).detach().cpu().numpy()
            shape_code_data[idx] = shape_code
            texture_code_data[idx] = texture_code
            obj_ids.append(obj_id)
        
        dbscan = DBSCAN(eps=0.5, min_samples=1)
        # shape_labels = dbscan.fit(shape_code_data).labels_
        # texture_labels = dbscan.fit(texture_code_data).labels_
        
        # tsne = TSNE(perplexity=n_obj-1)    
        # shape_tsne = tsne.fit_transform(shape_code_data)
        # texture_tsne = tsne.fit_transform(texture_code_data)
        
        # shape_fig = plt.figure(figsize=(10,10))
        # plt.xlim(shape_tsne[:,0].min(), shape_tsne[:,0].max()+1)
        # plt.ylim(shape_tsne[:,1].min(), shape_tsne[:,1].max()+1)
        # for i in range(n_obj):
        #     if shape_labels[i] == -1:
        #         continue
        #     plt.text(shape_tsne[i,0], shape_tsne[i,1], str(obj_ids[i]),
        #              color = colors[shape_labels[i]%10],
        #              fontdict = {'weight':'bold','size':9})
        # shape_filename = os.path.join(shape_tsne_dir, str(cls_id)+'.png')
        # plt.savefig(shape_filename)
        # plt.close(shape_fig)
        
        # texture_fig = plt.figure(figsize=(10,10))
        # plt.xlim(texture_tsne[:,0].min(), texture_tsne[:,0].max()+1)
        # plt.ylim(texture_tsne[:,1].min(), texture_tsne[:,1].max()+1)
        # for i in range(n_obj):
        #     if texture_labels[i] == -1:
        #         continue
        #     plt.text(texture_tsne[i,0], texture_tsne[i,1], str(obj_ids[i]),
        #              color = colors[texture_labels[i]%10],
        #              fontdict = {'weight':'bold','size':9})
        # texture_filename = os.path.join(texture_tsne_dir, str(cls_id)+'.png')
        # plt.savefig(texture_filename)
        # plt.close(texture_fig)

def main(args):
    log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    
    cam_info = cameraInfo(cfg)
    data = dataset.get_dataset(cfg)
    
    vis_dict = {}
    initialize_vis_dict(vis_dict, data, cam_info, cfg)
    load_model(vis_dict, log_dir)
    plot_tsne(vis_dict, log_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="./logs/debug_background")
    parser.add_argument('--config', type=str, 
                        default="./configs/Replica/config_replica_room0_vMAP_device1.json")
    # parser.add_argument('--config', type=str, 
    #                     default="./configs/ScanNet/config_scannet0000_vMAP.json")
    # parser.add_argument('--ids', type=int, nargs="+", default=[73, 74])
    args = parser.parse_args()
    
    main(args)