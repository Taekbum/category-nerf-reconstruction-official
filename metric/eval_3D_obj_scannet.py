import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio, accuracy_ratio
import os
import json
import argparse
import csv
import pandas as pd

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def calc_3d_metric(obj_id, mesh_rec, mesh_gt, mesh_vMAP, metric_dict, N=200000):
    """
    3D reconstruction metric.
    """
    transform, extents = trimesh.bounds.oriented_bounds(mesh_vMAP)
    # extents = extents / 0.9  # enlarge 0.9
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    mesh_rec_for_acc = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
    if mesh_rec_for_acc.vertices.shape[0] == 0:
        print(f"{obj_id} no mesh found")
        return False
    rec_pc_for_acc = trimesh.sample.sample_surface(mesh_rec_for_acc, N)
    rec_pc_tri_for_acc = trimesh.PointCloud(vertices=rec_pc_for_acc[0])
    rec_pc = trimesh.sample.sample_surface(mesh_rec, N)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri_for_acc.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    precision_rec = accuracy_ratio(gt_pc_tri.vertices, rec_pc_tri_for_acc.vertices, 0.05)
    recall_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.05)
    f_score_rec = 2*precision_rec*recall_rec/(precision_rec+recall_rec)
    
    metric_dict['acc'].append(accuracy_rec)
    metric_dict['comp'].append(completion_rec)
    metric_dict['prec'].append(precision_rec)
    metric_dict['recal'].append(recall_rec)
    metric_dict['f-score'].append(f_score_rec)
    return True

def get_gt_bg_mesh(gt_dir, exp, background_cls_list):
    label_map_file = "/media/satassd_1/tblee-larr/CVPR24/dataset/ScanNet/scannetv2-labels.combined.tsv"
    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='id')
    agg_file = os.path.join(gt_dir, exp+".aggregation.json")
    with open(agg_file) as f:
        label_obj_list = json.load(f)["segGroups"]

    bg_meshes = []
    for obj in label_obj_list:
        if label_map[obj["label"]] in background_cls_list:
            obj_file = os.path.join(gt_dir, exp+"_vh_clean_2.ply_" + str(int(obj["id"])+2) + ".ply")
            obj_mesh = trimesh.load(obj_file)
            bg_meshes.append(obj_mesh)
    # scannetv2-labels.combined.tsv  does not have label for "unknown" (inst_id=0)
    obj_file = os.path.join(gt_dir, exp+"_vh_clean_2.ply_0.ply")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='Datasets/ScanNet', type=str)
    parser.add_argument('--log_dir', default='logs/0816/ScanNet/random_batch/refined_mask', type=str)
    parser.add_argument('--log_dir_vMAP', default='../vMAP_offline/logs/vMAP_0819/ScanNet/refined_mask', type=str)
    parser.add_argument('--iteration', default=10000, type=int)
    args = parser.parse_args()
    
    background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
    scene_name = ["scene0024_00", "scene0084_00"]#["scene0013_02", "scene0281_00"]#["scene0059_00", "scene0066_00", "scene0247_00", "scene0266_00"]#["scene0011_00", "scene0059_00", "scene0066_00", "scene0247_00"]
    exp_name = ["1", "2", "3"]#, "4", "5"]
    data_dir = args.data_dir
    log_dir = args.log_dir
    log_dir_vMAP = args.log_dir_vMAP
    excel_file = os.path.join(log_dir, "revision_ours.xlsx")

    metric_dict = {'acc': [], 'comp': [], 'prec': [], 'recal': [], 'f-score': []}
    for scene in tqdm(scene_name):
        gt_dir = os.path.join(data_dir, scene+"/habitat")
        for exp in tqdm(exp_name):
            exp_dir = os.path.join(log_dir, exp)
            scene_dir = os.path.join(exp_dir, scene)
            mesh_dir = os.path.join(scene_dir, "scene_mesh")
            mesh_dir_vMAP = os.path.join(log_dir_vMAP, "1", scene, "scene_mesh")

            metric_dict_single = {'acc': [], 'comp': [], 'prec': [], 'recal': [], 'f-score': []}
            # get obj ids
            obj_ids = get_obj_ids(mesh_dir)#[49, 2, 35, 56, 26, 4, 51, 22, 40, 28, 27, 31, 14, 58, 17, 5, 18, 24, 21, 46, 33, 47, 54]#get_obj_ids(mesh_dir.replace("iMAP", "vMAP"))#[3,4,5,6,7,8,9,10,12,13,18,32,39,40,41,49,69,71,72,73,74,77,78,83,85,87,90,92,93]#[3,6,7,9,11,12,13,48] #get_obj_ids(mesh_dir.replace("iMAP", "vMAP"))
            for obj_id in tqdm(obj_ids):
                if obj_id == 0: # for bg
                    N = 200000
                    mesh_gt = get_gt_bg_mesh(gt_dir, scene, background_cls_list)
                else:   # for obj
                    N = 10000
                    obj_file = os.path.join(gt_dir, scene+"_vh_clean_2.ply_" + str(obj_id) + ".ply")
                    mesh_gt = trimesh.load(obj_file)

                rec_meshfile = os.path.join(mesh_dir, "iteration_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
                mesh_rec = trimesh.load(rec_meshfile)
                rec_meshfile_vMAP = os.path.join(mesh_dir_vMAP, "it_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
                mesh_vMAP = trimesh.load(rec_meshfile_vMAP)

                calc_3d_metric(obj_id, mesh_rec, mesh_gt, mesh_vMAP, metric_dict_single, N=N)  # for objs use 10k, for scene use 200k points
            
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