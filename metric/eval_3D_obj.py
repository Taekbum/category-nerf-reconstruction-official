import sys
sys.path.append('/media/satassd_1/tblee-larr/CVPR24/vMap_plus_copy/src')
import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio
import os
import json
import argparse
import pandas as pd
import torch

def calc_3d_metric(obj_id, mesh_rec, mesh_gt, mesh_vMAP, metric_dict, N=200000, same_file=False):
    """
    3D reconstruction metric.
    """
    transform, extents = trimesh.bounds.oriented_bounds(mesh_gt)#mesh_vMAP)
    # extents = extents / 0.9  # enlarge 0.9
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    if same_file:
        mesh_rec_for_acc = mesh_rec
    else:
        mesh_rec_for_acc = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
        if mesh_rec_for_acc.vertices.shape[0] == 0:
            print(f"{obj_id} no mesh found")
            return
    rec_pc = trimesh.sample.sample_surface(mesh_rec, N)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])
    rec_pc_for_acc = trimesh.sample.sample_surface(mesh_rec_for_acc, N)
    rec_pc_tri_for_acc = trimesh.PointCloud(vertices=rec_pc_for_acc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri_for_acc.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.05)
    completion_ratio_rec_1 = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.01)

    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    completion_ratio_rec_1 *= 100  # convert to %

    metric_dict['acc'].append(accuracy_rec)
    metric_dict['comp'].append(completion_rec)
    metric_dict['<1'].append(completion_ratio_rec_1)
    metric_dict['<5'].append(completion_ratio_rec)
    return metric_dict

def get_gt_bg_mesh(gt_dir, background_cls_list):
    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        label_obj_list = json.load(f)["objects"]

    bg_meshes = []
    for obj in label_obj_list:
        if int(obj["class_id"]) in background_cls_list:
            obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(int(obj["id"])) + ".ply")
            obj_mesh = trimesh.load(obj_file)
            bg_meshes.append(obj_mesh)

    bg_mesh = trimesh.util.concatenate(bg_meshes)
    return bg_mesh

def get_obj_ids(obj_dir):
    files = os.listdir(obj_dir)
    obj_ids = []
    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '' or obj_id == '0':
            continue
        obj_ids.append(int(obj_id))
    return obj_ids

def get_obj_ids_of_selected_class(gt_dir, cls_id_list):
    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        label_obj_list = json.load(f)["objects"]
    obj_ids = []
    for obj in label_obj_list:
        if int(obj["class_id"]) in cls_id_list:
            obj_ids.append(int(obj["id"]))
    return obj_ids

def get_obj_ids_per_classes(gt_dir, obj_dir):
    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        id_to_label = np.array(json.load(f)["id_to_label"])
    label_unique = np.unique(id_to_label)
    label_more_than_five_objects = []
    for label in label_unique:
        n_obj = np.count_nonzero(id_to_label==label)
        if n_obj >= 2:
            label_more_than_five_objects.append(label)
       
    files = os.listdir(obj_dir)
    obj_ids = []
    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '' or obj_id == '0':
            continue
        if id_to_label[int(obj_id)] in label_more_than_five_objects:
            obj_ids.append(int(obj_id))
    print(obj_ids)
    return obj_ids

def get_obj_ids_with_bool(ckpt_dir, obj_dir):
    single_bool = {}
    labels = [int(label) for label in os.listdir(ckpt_dir) if int(label) != 0]
    ckpt_files = [os.path.join(ckpt_dir, str(label), f'cls_{label}_iteration_10000.pth') for label in labels]
    for ckpt_file in ckpt_files:
        ckpt = torch.load(ckpt_file)
        obj_ids = ckpt["obj_tensor_dict"].keys()
        if len(obj_ids) > 1:
            for obj_id in obj_ids:
                single_bool[obj_id] = False
        else:
            single_bool[list(obj_ids)[0]] = True

    files = os.listdir(obj_dir)
    obj_ids = []

    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '' or obj_id == '0':
            continue
        obj_ids.append(int(obj_id))
    return obj_ids, single_bool

def obj_ids_for_objsdf_plus(scene):
    if scene == 'room_0':
        obj_ids = [2, 5, 6, 7, 8, 9, 11, 12, 13, 18, 20, 24, 27, 32, 35, 39, 41, 43, 54, 55, 59, 63, 69, 70, 71, 72, 73, 74, 75, 77, 83, 86, 87, 92]
    elif scene == 'room_1':
        obj_ids = [3, 4, 7, 8, 9, 10, 11, 12, 13, 18, 22, 28, 32, 36, 39, 40, 43, 45, 46, 48, 51, 52, 53, 54]
    elif scene == 'room_2':
        obj_ids = [3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 27, 28, 35, 40, 46, 47, 48, 49, 51, 56]
    elif scene == 'office_0':
        obj_ids = [2, 3, 4, 7, 9, 12, 19, 22, 23, 30, 34, 42, 49, 58, 61, 64, 66]
    elif scene == 'office_1':
        obj_ids = [3, 5, 8, 12, 17, 23, 29, 31, 40, 42, 44, 45]
    elif scene == 'office_2':
        obj_ids = [2, 8, 12, 17, 23, 28, 32, 38, 39, 41, 43, 44, 51, 58, 65, 69, 73, 80, 84, 85, 86, 87, 90, 93]
    elif scene == 'office_3':
        obj_ids = [1, 8, 9, 10, 12, 14, 15, 16, 22, 23, 25, 26, 27, 30, 51, 54, 55, 61, 72, 76, 83, 88, 89, 91, 95, 98, 100]
    elif scene == 'office_4':
        obj_ids = [2, 3, 4, 10, 12, 13, 15, 16, 17, 18, 19, 20, 31, 48, 49, 51, 52, 53, 57, 58, 61, 69]
        
    return obj_ids

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Datasets/Replica/vmap', type=str)
parser.add_argument('--log_dir', default='logs/0816/Replica/random_batch', type=str)
parser.add_argument('--log_dir_vMAP', default='../vMAP_offline/logs/0819/Replica', type=str)
parser.add_argument('--iteration', default=10000, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    background_cls_list = [5, 12, 30, 31, 40, 60, 92, 93, 95, 97, 98, 79]
    scene_name = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]#["room_2", "office_0", "office_3", "office_4"] #["room_0", "room_1", "room_2", "office_0"]#["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
    exp_name = ["1","2","3","4","5"]
    data_dir = args.data_dir #"Datasets/Replica/vmap/"
    log_dir = args.log_dir #"logs/vMAP/"
    log_dir_vMAP = args.log_dir_vMAP
    excel_file = os.path.join(log_dir, "r3_3_objsdfplus_gt_bound.xlsx")

    metric_dict = {'acc': [], 'comp': [], '<1': [], '<5': []}
    for scene in tqdm(scene_name):
        gt_dir = os.path.join(data_dir, scene, "habitat")
        for exp in tqdm(exp_name):
            exp_dir = os.path.join(log_dir, exp)
            scene_dir = os.path.join(exp_dir, scene)
            ckpt_dir = os.path.join(scene_dir, "ckpt")
            mesh_dir = os.path.join(scene_dir, "scene_mesh")
            mesh_dir_vMAP = os.path.join(log_dir_vMAP, "1", scene, "scene_mesh")

            metric_dict_single = {'acc': [], 'comp': [], '<1': [], '<5': []}
            # get obj ids
            # cls_id_list = [80]
            # obj_ids = get_obj_ids_of_selected_class(gt_dir, cls_id_list)
            # obj_ids = get_obj_ids_per_classes(gt_dir, mesh_dir)
            # print(f"scene: {scene}, n_obj: {len(obj_ids)}")
            # if len(obj_ids) < 1:
            #     continue
            # obj_ids = get_obj_ids(mesh_dir)
            obj_ids = obj_ids_for_objsdf_plus(scene)
            # obj_ids, single_bool = get_obj_ids_with_bool(ckpt_dir, mesh_dir)
            for obj_id in tqdm(obj_ids):
                if obj_id == 0: # for bg
                    print("include background")
                    N = 200000
                    mesh_gt = get_gt_bg_mesh(gt_dir, background_cls_list)
                else:   # for obj
                    N = 10000
                    obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(obj_id) + ".ply")
                    mesh_gt = trimesh.load(obj_file)

                rec_meshfile = os.path.join(mesh_dir, "iteration_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
                rec_meshfile_vMAP = os.path.join(mesh_dir_vMAP, "it_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
                # if single_bool[obj_id]:
                #     rec_meshfile = os.path.join(log_dir_vMAP, exp, scene, "scene_mesh", "it_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
                #     mesh_rec = trimesh.load(rec_meshfile)
                #     same_file = True
                # else:
                #     mesh_rec = trimesh.load(rec_meshfile)
                #     same_file = False
                mesh_rec = trimesh.load(rec_meshfile)
                mesh_vMAP = trimesh.load(rec_meshfile_vMAP)

                calc_3d_metric(obj_id, mesh_rec, mesh_gt, mesh_vMAP, metric_dict_single, N=N)#, same_file=same_file)  # for objs use 10k, for scene use 200k points

            for metric in metric_dict_single.keys():
                metric_ = np.array(metric_dict_single[metric])
                metric_dict[metric].append(np.mean(metric_))
        print("finish scene ", scene)
        
    combinations = [(scene_, exp_) for scene_ in scene_name for exp_ in exp_name]    
    excel_data = {
        "scene": [combi[0] for combi in combinations],
        "exp": [combi[1] for combi in combinations],
    }
    for metric in metric_dict.keys():
        excel_data[metric] = metric_dict[metric]
    
    df = pd.DataFrame(excel_data)
    pivot_table = df.pivot_table(values=list(metric_dict.keys()), index="exp", columns="scene")
    pivot_table.to_excel(excel_file)