class BaseDataSet(Dataset):
    def __init__(self):
        pass
    
    def get_all_poses(self):
        print('get_all_poses')
        t1 = time.time()
        for cls_id in self.inst_dict.keys():
            inst_dict_cls = self.inst_dict[cls_id]
            obj_ids = list(inst_dict_cls.keys())
            n_obj = len(obj_ids)
            if cls_id == 0:
                background_list = inst_dict_cls['frame_info']
                if self.name == "replica":
                    background_pcs = accumulate_pointcloud(0, background_list, self.sample_dict, self.intrinsic_open3d)
                else:
                    background_pcs = accumulate_pointcloud_tsdf(0, background_list, self.sample_dict, self.intrinsic_open3d, depth_scale=self.depth_scale, max_depth=self.max_depth)
                transform, extents = trimesh.bounds.oriented_bounds(np.asarray(background_pcs.points))  # pc
                transform = np.linalg.inv(transform)
                bbox3D = BoundingBox()
                bbox3D.center = transform[:3, 3]
                bbox3D.R = transform[:3, :3]
                bbox3D.extent = extents
                inst_dict_cls['bbox3D'] = bbox3D
                inst_dict_cls['pcs'] = background_pcs
            else:
                for idx in range(n_obj):
                    inst_id = obj_ids[idx]
                    inst_list = inst_dict_cls[inst_id]['frame_info']
                    if self.name == "replica":
                        inst_pcs = accumulate_pointcloud(inst_id, inst_list, self.sample_dict, self.intrinsic_open3d)
                    else:
                        if 'pcs' not in inst_dict_cls[inst_id].keys():
                            print(f"{inst_id} is not detected from semantically refined geometry segmentations")
                            inst_pcs = None
                            inst_dict_cls[inst_id]['T_obj'] = np.eye(4)
                        else:
                            inst_pcs = inst_dict_cls[inst_id]['pcs']
                            inst_pcs = inst_pcs.voxel_down_sample(0.01)
                    inst_dict_cls[inst_id]['pcs'] = inst_pcs
                    
        t2 = time.time()
        print('get_all_poses takes {} seconds'.format(t2-t1))

    def get_uncertainty_fields(self, cfg, load_pretrained=False, use_reliability=True):
        
        emb_size1 = 21*(3+1)+3
        emb_size2 = 21*(5+1)+3 - emb_size1
        
        self.count_dict = {}
        self.bbox3d_dict = {}
        if load_pretrained:
            self.fc_occ_map_dict = {}
            self.pe_dict = {}
            for cls_id in self.inst_dict.keys():
                if cls_id == 0:
                    continue
                inst_dict_cls = self.inst_dict[cls_id]
                obj_ids = list(inst_dict_cls.keys())
                if cls_id not in self.fc_occ_map_dict.keys():
                    self.fc_occ_map_dict[cls_id] = {}
                if cls_id not in self.pe_dict.keys():
                    self.pe_dict[cls_id] = {}
                if cls_id not in self.bbox3d_dict.keys():
                    self.bbox3d_dict[cls_id] = {}
                for obj_id in obj_ids:
                    ckpt_dir = os.path.join(cfg.weight_root, "ckpt", str(obj_id))
                    # if not os.path.isdir(ckpt_dir):
                    #     continue
                    ckpt_paths = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))]
                    ckpt_path = ckpt_paths[-1]
                    ckpt = torch.load(ckpt_path, map_location = torch.device('cpu'))
                    
                    self.fc_occ_map_dict[cls_id][obj_id] = model.OccupancyMap(
                        emb_size1,
                        emb_size2,
                        hidden_size=cfg.hidden_feature_size
                    )
                    self.fc_occ_map_dict[cls_id][obj_id].apply(model.init_weights).to(cfg.data_device)
                    self.pe_dict[cls_id][obj_id] = embedding.UniDirsEmbed(max_deg=cfg.n_unidir_funcs, scale=ckpt["obj_scale"]).to(cfg.data_device)
                    self.fc_occ_map_dict[cls_id][obj_id].load_state_dict(ckpt["FC_state_dict"])
                    self.pe_dict[cls_id][obj_id].load_state_dict(ckpt["PE_state_dict"])
                    self.bbox3d_dict[cls_id][obj_id] = ckpt["bbox"]
        
        # world coord
        phi = torch.linspace(0, np.pi, 100)
        theta = torch.linspace(0, 2*np.pi, 100)
        phi, theta = torch.meshgrid(phi, theta)
        phi = phi.t()
        theta = theta.t()

        x_norm = torch.sin(phi) * torch.cos(theta)
        y_norm = torch.sin(phi) * torch.sin(theta)
        z_norm = torch.cos(phi)
        self.phi = phi.reshape(-1)
        self.theta = theta.reshape(-1)

        for cls_id in self.fc_occ_map_dict.keys():
            if not cls_id in self.count_dict.keys():
                self.count_dict[cls_id] = {}

            bounds = []
            obj_ids = list(self.inst_dict[cls_id].keys())
            for obj_id in obj_ids:
                points = np.asarray(self.inst_dict[cls_id][obj_id]['pcs'].points)
                bound = points.max(axis=0) - points.min(axis=0) # aabb
                bound = np.maximum(bound, 0.10) # scale at least 10cm
                bounds.append(torch.from_numpy((bound/2).astype(np.float32)))
            
            bounds = torch.stack(bounds, dim=0) 
            rs = 1.2*torch.sqrt(torch.square(bounds).sum(dim=-1))
            
            entropies_max_list = []
            metric_list = []
            obj_ids = list(self.fc_occ_map_dict[cls_id].keys())
            
            for idx in range(len(obj_ids)):
                obj_id = obj_ids[idx]
                r = rs[idx]
                x = r * x_norm
                y = r * y_norm
                z = r * z_norm
                
                rays_o_o = torch.stack([x, y, z], dim=-1).reshape(-1,3)
                viewdir = -rays_o_o/r               
                
                points = np.asarray(self.inst_dict[cls_id][obj_id]['pcs'].points)
                if self.name == "replica":
                    center_np = ((points.max(axis=0) + points.min(axis=0))/2).astype(np.float32)
                else: # for noisy point cloud
                    center_np = (points.mean(axis=0)).astype(np.float32)
                center = torch.from_numpy(center_np)
                rays_o = center + rays_o_o
                
                far = 2*r
                z_vals = stratified_bins(0, far, 96, rays_o.shape[0], device='cpu', z_fixed=True)
                xyz = rays_o[..., None, :] + (viewdir[:, None, :] * z_vals[..., None])
                embedding_ = self.pe_dict[cls_id][obj_id](xyz.to(cfg.data_device))
                sigmas, _ = self.fc_occ_map_dict[cls_id][obj_id](embedding_)
                occupancies = torch.sigmoid(10*sigmas.squeeze(-1)).detach().cpu()
                
                term_probs = render_rays.occupancy_to_termination(occupancies).numpy()                                      
                
                entropies = np.sum(-term_probs*np.log(term_probs + 1e-10), axis=-1)
                entropies_max_list.append(entropies.max())
                    
                if use_reliability:
                    heuristic = np.sum(term_probs, axis=-1) * np.exp(-0.5*entropies)
                    reliability = calculate_reliability(heuristic, eta=0.9, m1=0.1, m2=0.15, M1=0.57, M2=0.65)
                    metric_list.append(1-reliability)
                else:
                    metric_list.append(entropies)
            
            # rate of below threshold
            if use_reliability:
                for i in range(len(obj_ids)):
                    obj_id = obj_ids[i]
                    metric = metric_list[i]
                    measure = metric[metric<0.5].shape[0]
                    self.count_dict[cls_id][obj_id] = measure
            else:
                threshold = 0.8 * min(entropies_max_list)
                for i in range(len(obj_ids)):
                    obj_id = obj_ids[i]
                    entropies = metric_list[i]
                    measure = entropies[entropies<threshold].shape[0]
                    self.count_dict[cls_id][obj_id] = measure
    
    def align_poses(self, multi_init_pose=True, eta1=0.06, eta2=0.15, eta3=0.12):
        from teaser_utils.teaser_fpfh_icp import TEASER_FPFH_ICP
        print('align_poses')
        t1 = time.time()
        
        if self.name == "replica":
            cls_id_add = 100
        else:
            cls_id_add = 10000
        
        self.chamfer_dict = {}
        self.chamfer_opposite_dict = {}
        self.id_representative_dict = {}
        while self.bbox3d_dict:
            for cls_id in self.bbox3d_dict.copy().keys():
                self.chamfer_dict[cls_id] = {}
                self.chamfer_opposite_dict[cls_id] = {}
                obj_ids = list(self.bbox3d_dict[cls_id].keys())
                counts = [self.count_dict[cls_id][obj_id] for obj_id in self.count_dict[cls_id].keys()]
                if len(counts) > 1:
                    counts = np.array(counts)
                    idx_representative = np.argmax(counts)
                else:
                    idx_representative = 0

                inst_dict_cls = self.inst_dict[cls_id]
                
                # get pose for representative
                obj_id_representative = obj_ids[idx_representative]
                inst_pcs_template = inst_dict_cls[obj_id_representative]['pcs']
                T_obj, bbox3D = get_pose_from_pointcloud(inst_pcs_template)
                inst_dict_cls[obj_id_representative]['T_obj'] = T_obj
                if bbox3D is not None:
                    inst_dict_cls[obj_id_representative]['bbox3D'] = bbox3D
                
                self.id_representative_dict[cls_id] = obj_id_representative
                
                other_obj_ids = []
                for idx in range(len(obj_ids)):
                    if idx != idx_representative:
                        obj_id = obj_ids[idx]
                        other_obj_ids.append(obj_id)

                if len(other_obj_ids) == 0:
                    self.bbox3d_dict.pop(cls_id) 
                    continue

                T_obj_template = np.copy(inst_dict_cls[obj_id_representative]['T_obj'])
                scale_template = np.linalg.det(T_obj_template[:3, :3]) ** (1/3)
                T_obj_template[:3, :3] = T_obj_template[:3, :3]/scale_template
                template_np_w = np.array(inst_pcs_template.points)
                    
                template = torch.from_numpy(template_np_w.transpose(1,0)).unsqueeze(0).to(self.device)
                if multi_init_pose:
                    transform_list = get_possible_transform_from_bbox()
                    template_np_w_list = []
                    for transform in transform_list:
                        template_np_w_transformed = transform_pointcloud(template_np_w, transform)
                        template_np_w_list.append(template_np_w_transformed)
                    template = torch.from_numpy(np.stack(template_np_w_list).transpose(0,2,1)).to(self.device)

                for idx in range(len(other_obj_ids)):
                    obj_id = other_obj_ids[idx]
                    inst_pcs = inst_dict_cls[obj_id]['pcs']
                    source_np_w = np.array(inst_pcs.points)
                    
                    scale_source = np.max(source_np_w.max(axis=0)-source_np_w.min(axis=0))/2 # 0720
                    
                    # use TEASER++
                    source = torch.from_numpy(source_np_w.transpose(1,0)).unsqueeze(0).to(self.device)
                    teaser = TEASER_FPFH_ICP(source, voxel_size=0.1, spc=True, visualize=False)
                    R_rel, t_rel = teaser.forward(template)
                    if multi_init_pose:
                        T_rel_multi = np.repeat(np.eye(4)[None, ...], template.shape[0], axis=0)
                        T_rel_multi[:, :3, :3] = R_rel.detach().cpu().numpy()
                        T_rel_multi[:, :3, 3:] = t_rel.detach().cpu().numpy()
                        chamfer_unidir_list = np.zeros(T_rel_multi.shape[0])
                        for idx_cand in range(T_rel_multi.shape[0]):
                            T_rel = np.linalg.inv(transform_list[idx_cand]) @ T_rel_multi[idx_cand]
                            source_transformed = transform_pointcloud(source_np_w, T_rel)
                            inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                            chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source # important: normalize cd metric!!
                            chamfer_unidir_list[idx_cand] = chamfer_unidir

                        idx_sel = np.argmin(chamfer_unidir_list)
                        T_rel = np.linalg.inv(transform_list[idx_sel]) @ T_rel_multi[idx_sel]
                        chamfer_unidir = chamfer_unidir_list[idx_sel]
                        
                    else:
                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R_rel.squeeze(0).detach().cpu().numpy()
                        T_rel[:3, 3:] = t_rel.squeeze(0).detach().cpu().numpy()
                        source_transformed = transform_pointcloud(source_np_w, T_rel)
                        inst_pcs_transformed = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(source_transformed))
                        chamfer_unidir = np.asarray(inst_pcs_transformed.compute_point_cloud_distance(inst_pcs_template)).mean()/scale_source                 
                    
                    # consider as other subcategory if observed point cloud is note aligned well
                    self.chamfer_dict[cls_id][obj_id] = chamfer_unidir
                    if chamfer_unidir > eta2:
                        subcategorize = True
                    elif chamfer_unidir < eta1:
                        subcategorize = False
                    else:
                        chamfer_opposite = np.asarray(inst_pcs_template.compute_point_cloud_distance(inst_pcs_transformed)).mean()/scale_template
                        self.chamfer_opposite_dict[cls_id][obj_id] = chamfer_opposite
                        subcategorize = True if chamfer_opposite > eta3 else False
                    
                    if subcategorize:
                        cls_id_sub = cls_id + cls_id_add
                        inst_dict = inst_dict_cls[obj_id]
                        metric = self.count_dict[cls_id][obj_id]
                        bbox3d = self.bbox3d_dict[cls_id][obj_id]
                        
                        if not cls_id_sub in self.inst_dict.keys():
                            self.inst_dict[cls_id_sub] = {}
                        self.inst_dict[cls_id_sub].update({obj_id: inst_dict})
                        if not cls_id_sub in self.count_dict.keys():
                            self.count_dict[cls_id_sub] = {}
                        self.count_dict[cls_id_sub].update({obj_id: metric})
                        if not cls_id_sub in self.bbox3d_dict.keys():
                            self.bbox3d_dict[cls_id_sub] = {}
                        self.bbox3d_dict[cls_id_sub].update({obj_id: bbox3d})
                        if not cls_id_sub in self.pe_dict.keys():
                            self.pe_dict[cls_id_sub] = {}
                        self.pe_dict[cls_id_sub].update({obj_id: self.pe_dict[cls_id][obj_id]})
                        if not cls_id_sub in self.fc_occ_map_dict.keys():
                            self.fc_occ_map_dict[cls_id_sub] = {}
                        self.fc_occ_map_dict[cls_id_sub].update({obj_id: self.fc_occ_map_dict[cls_id][obj_id]})
                        
                        inst_dict_cls.pop(obj_id, None)
                        self.count_dict[cls_id].pop(obj_id, None)
                        self.bbox3d_dict[cls_id].pop(obj_id, None)
                        self.pe_dict[cls_id].pop(obj_id, None)
                        self.fc_occ_map_dict[cls_id].pop(obj_id, None)
                    else:
                        T_obj = np.linalg.inv(T_rel) @ T_obj_template # if template = T_rel @ source
                        inst_dict_cls[obj_id]['T_obj'] = T_obj # center to aligned position
                        
                        # bound to obb w.r.t aligned pose
                        get_obb(inst_dict_cls[obj_id])
                
                self.bbox3d_dict.pop(cls_id)       

        t2 = time.time()
        print('align_poses takes {} seconds'.format(t2-t1))
   