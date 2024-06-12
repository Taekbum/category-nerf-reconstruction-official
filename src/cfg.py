import json
import numpy as np
import os
import utils

class Config:
    def __init__(self, config_file):
        # setting params
        with open(config_file) as json_file:
            config = json.load(json_file)

        # training strategy
        self.do_bg = bool(config["trainer"]["do_bg"])
        self.training_device = config["trainer"]["train_device"]
        self.data_device = config["trainer"]["data_device"]
        self.max_n_models = config["trainer"]["n_models"]
        self.live_mode = bool(config["dataset"]["live"])
        self.keep_live_time = config["dataset"]["keep_alive"]
        self.imap_mode = config["trainer"]["imap_mode"]
        self.training_strategy = config["trainer"]["training_strategy"]  # "forloop" "vmap"
        self.obj_id = -1
        self.max_iter = config["trainer"]["max_iter"]

        # dataset setting
        self.dataset_format = config["dataset"]["format"]
        self.dataset_dir = config["dataset"]["path"]
        self.depth_scale = 1 / config["trainer"]["scale"]
        # camera setting
        self.max_depth = config["render"]["depth_range"][1]
        self.min_depth = config["render"]["depth_range"][0]
        self.mh = config["camera"]["mh"]
        self.mw = config["camera"]["mw"]
        self.height = config["camera"]["h"]
        self.width = config["camera"]["w"]
        self.H = self.height - 2 * self.mh
        self.W = self.width - 2 * self.mw
        if "fx" in config["camera"]:
            self.fx = config["camera"]["fx"]
            self.fy = config["camera"]["fy"]
            self.cx = config["camera"]["cx"] - self.mw
            self.cy = config["camera"]["cy"] - self.mh
        else:   # for scannet
            intrinsic = utils.load_matrix_from_txt(os.path.join(self.dataset_dir, "intrinsic/intrinsic_depth.txt"))
            self.fx = intrinsic[0, 0]
            self.fy = intrinsic[1, 1]
            self.cx = intrinsic[0, 2] - self.mw
            self.cy = intrinsic[1, 2] - self.mh
        if "distortion" in config["camera"]:
            self.distortion_array = np.array(config["camera"]["distortion"])
        elif "k1" in config["camera"]:
            k1 = config["camera"]["k1"]
            k2 = config["camera"]["k2"]
            k3 = config["camera"]["k3"]
            k4 = config["camera"]["k4"]
            k5 = config["camera"]["k5"]
            k6 = config["camera"]["k6"]
            p1 = config["camera"]["p1"]
            p2 = config["camera"]["p2"]
            self.distortion_array = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        else:
            self.distortion_array = None

        # training setting
        self.win_size = config["model"]["window_size"]
        self.n_iter_per_frame = config["render"]["iters_per_frame"]
        self.n_per_optim = config["render"]["n_per_optim"]
        self.n_samples_per_frame = self.n_per_optim // self.win_size
        self.win_size_bg = config["model"]["window_size_bg"]
        self.n_per_optim_bg = config["render"]["n_per_optim_bg"]
        self.n_samples_per_frame_bg = self.n_per_optim_bg // self.win_size_bg
        self.keyframe_buffer_size = config["model"]["keyframe_buffer_size"]
        self.keyframe_step = config["model"]["keyframe_step"]
        self.keyframe_step_bg = config["model"]["keyframe_step_bg"]
        self.obj_scale = config["model"]["obj_scale"]
        self.bg_scale = config["model"]["bg_scale"]
        self.hidden_feature_size = config["model"]["hidden_feature_size"]
        self.hidden_feature_size_bg = config["model"]["hidden_feature_size_bg"]
        self.n_bins_cam2surface = config["render"]["n_bins_cam2surface"]
        self.n_bins_cam2surface_bg = config["render"]["n_bins_cam2surface_bg"]
        self.n_bins = config["render"]["n_bins"]
        self.n_unidir_funcs = config["model"]["n_unidir_funcs"]
        self.surface_eps = config["model"]["surface_eps"]
        self.stop_eps = config["model"]["other_eps"]
        self.net_hyperparams = config["model"]["net_hyperparams"]

        # optimizer setting
        self.learning_rate = config["optimizer"]["args"]["lr"]
        self.pose_learning_rate = config["optimizer"]["args"]["pose_lr"]
        self.code_learning_rate = config["optimizer"]["args"]["code_lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]
        self.pose_weight_decay = config["optimizer"]["args"]["pose_weight_decay"]
        self.code_weight_decay = config["optimizer"]["args"]["code_weight_decay"]

        # vis setting
        self.vis_device = config["vis"]["vis_device"]
        self.n_vis_iter = config["vis"]["n_vis_iter"]
        self.live_voxel_size = config["vis"]["live_voxel_size"]
        self.grid_dim = config["vis"]["grid_dim"]
        self.mesh_it = config["vis"]["mesh_it"]
        self.img_it = config["vis"]["img_it"]
        self.save_it = config["save_it"]
        if "bound" in config["vis"].keys():
            self.scene_bound = config["vis"]["bound"]
        
        # for initial pose
        if self.dataset_format == "Replica":
            info_file = os.path.join(config["dataset"]["habitat_dir"], "info_semantic.json")
            with open(info_file) as f:
                info = json.load(f)
            self.scene_mesh_file = os.path.join(config["dataset"]["habitat_dir"], "mesh_semantic.ply")
        elif self.dataset_format == "ScanNet":
            mesh_name = self.dataset_dir.split("/")[-3] + "_vh_clean_2.ply"
            self.scene_mesh_file = os.path.join(config["dataset"]["habitat_dir"], mesh_name)
            ## scannet option    
            self.use_refined_mask = config["use_refined_mask"]

            
        # log
        self.log_freq = config["log_freq"]
        
        # setting related to forgetting
        if "use_equal_batch" in config.keys():
            self.use_equal_batch = config["use_equal_batch"]
        else:
            self.use_equal_batch = True
        
        # codenerf baslines
        if "codenerf" in config.keys():
            self.codenerf = config["codenerf"]
        else:
            self.codenerf = False
        if "fix_codenerf" in config.keys():
            self.fix_codenerf = config["fix_codenerf"]
        else:
            self.fix_codenerf = False
            
        # fix pose
        if "fix_pose" in config.keys():
            self.fix_pose = config["fix_pose"]
        else:
            self.fix_pose = True
            
        # use editnerf or codenerf
        if "editnerf" in config.keys():
            self.editnerf = config["editnerf"]
        else:
            self.editnerf = False
            
        # instance-free guide explicitly
        if "use_zero_code" in config.keys():
            self.use_zero_code = config["use_zero_code"]
        else:
            self.use_zero_code = False
        
        if "use_mean_code" in config.keys():
            self.use_mean_code = config["use_mean_code"]
        else:
            self.use_mean_code = False
            
        if "use_nocs" in config.keys():
            self.use_nocs = config["use_nocs"]
        else:
            self.use_nocs = True
            
        if "enlarge_scale" in config.keys():
            self.enlarge_scale = config["enlarge_scale"]
        else:
            self.enlarge_scale = False
        
        if "align_poses" in config.keys():
            self.align_poses = config["align_poses"]
        else:
            self.align_poses = False
        
        if "load_pretrained" in config.keys():
            self.load_pretrained = config["load_pretrained"]
        else:
            self.load_pretrained = False
            
        if self.load_pretrained:
            self.weight_root = config["weight_root"]
            
        if "uncertainty_guided_sampling" in config.keys():
            self.uncertainty_guided_sampling = config["uncertainty_guided_sampling"]
        else:
            self.uncertainty_guided_sampling = False
            
        if "no_shuffle" in config.keys():
            self.no_shuffle = config["no_shuffle"]
        else:
            self.no_shuffle = False
            
        if "suppress_unseen" in config.keys():
            self.suppress_unseen = config["suppress_unseen"]
        else:
            self.suppress_unseen = False
            
        if "same_amount" in config.keys():
            self.same_amount = config["same_amount"]
        else:
            self.same_amount = True

        if "smooth" in config.keys():
            self.smooth = config["smooth"]
        else:
            self.smooth = False
            
        if "use_certain_data" in config.keys():
            self.use_certain_data = config["use_certain_data"]
        else:
            self.use_certain_data = False
            
        if "use_uncertainty" in config.keys():
            self.use_uncertainty = config["use_uncertainty"]
        else:
            self.use_uncertainty = False
            
        if "select_minimum" in config.keys():
            self.select_minimum = config["select_minimum"]
        else:
            self.select_minimum = False
            
        if "staged_poseoptim" in config.keys():
            self.staged_poseoptim = config["staged_poseoptim"]
        else:
            self.staged_poseoptim = False
            
        if "template_scale" in config.keys():
            self.template_scale = config["template_scale"]
        else:
            self.template_scale = False
            
        if "representative_metric" in config.keys():
            self.representative_metric = config["representative_metric"]
        else:
            self.representative_metric = "uncertainty"
            
        if "subcategorize" in config.keys():
            self.subcategorize = config["subcategorize"]
        else:
            self.subcategorize = True
            
        if "uniform_region" in config.keys():
            self.uniform_region = config["uniform_region"]
        else:
            self.uniform_region = False
            
        if "large_concent" in config.keys():
            self.large_concent = config["large_concent"]
        else:
            self.large_concent = False
            
        if "object_wise_model" in config.keys():
            self.object_wise_model = config["object_wise_model"]
        else:
            self.object_wise_model = False
            
        if "eta1" in config.keys():
            self.eta1 = config["eta1"]
        else:
            self.eta1 = 0.06
        if "eta2" in config.keys():
            self.eta2 = config["eta2"]
        else:
            self.eta2 = 0.15
        if "eta3" in config.keys():
            self.eta3 = config["eta3"]
        else:
            self.eta3 = 0.12
            
        # revision r1_2
        if "adaptive_obj_num" in config.keys():
            self.adaptive_obj_num = config["adaptive_obj_num"]
            self.get_frames_incremental = config["adaptive_obj_num"]
            self.it_add_obs = config["it_add_obs"]
            self.frames = config["frames"]
        
        else:
            self.adaptive_obj_num = False