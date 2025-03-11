import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio
import os
import json
import argparse
import csv

def calc_3d_metric(mesh_rec, mesh_ref, N=200000):
    """
    3D reconstruction metric.
    """
    metrics = [[] for _ in range(3)]
    transform, extents = trimesh.bounds.oriented_bounds(mesh_ref)
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    mesh_rec_for_acc = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
    if mesh_rec_for_acc.vertices.shape[0] == 0:
        print("no mesh found")
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

    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec)
    return metrics

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

def get_gt_bg_mesh_scannet(gt_dir, exp, background_cls_list, label_map_file):
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

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def get_obj_ids(obj_dir):
    files = os.listdir(obj_dir)
    obj_ids = []
    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '' or obj_id == '0':
            continue
        obj_ids.append(int(obj_id))
    return obj_ids

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='Datasets/Replica', type=str)
parser.add_argument('--log_dir', default='logs/Replica', type=str)
parser.add_argument('--log_dir_ref', default='', type=str)
parser.add_argument('--iteration', default=10000, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    data_dir = args.data_dir
    log_dir = args.log_dir
    log_dir_ref = args.log_dir_ref
    
    dataset = args.data_dir.split('/')[-1]
    if dataset == "Replica":
        background_cls_list = [5, 12, 30, 31, 40, 60, 92, 93, 95, 97, 98, 79]
        exp_name = ["room_0", "room_1", "room_2", "office_0", "office_1", "office_2", "office_3", "office_4"]
    elif dataset == "ScanNet":
        background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        exp_name = ["scene0013_02", "scene0059_00", "scene0066_00", "scene0281_00"]
        label_map_file = os.path.join(data_dir, "scannetv2-labels.combined.tsv")
    else:
        print(f"Dataset {dataset} is not supported")
        NotImplementedError()
        
    for exp in tqdm(exp_name):
        gt_dir = os.path.join(data_dir, exp, "habitat")
        exp_dir = os.path.join(log_dir, exp)
        mesh_dir = os.path.join(exp_dir, "scene_mesh")
        mesh_dir_ref = os.path.join(log_dir_ref, exp, "scene_mesh")
        output_path = os.path.join(exp_dir, "eval_mesh")
        os.makedirs(output_path, exist_ok=True)
        metrics_3D = [[] for _ in range(3)]

        # get obj ids
        obj_ids = get_obj_ids(mesh_dir)
        for obj_id in tqdm(obj_ids):
            if obj_id == 0: # for bg
                N = 200000
                mesh_gt = get_gt_bg_mesh(gt_dir, background_cls_list) if dataset == "Replica"  \
                    else get_gt_bg_mesh_scannet(gt_dir, exp, background_cls_list, label_map_file)
            else:   # for obj
                N = 10000
                obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(obj_id) + ".ply") if dataset == "Replica" \
                    else os.path.join(gt_dir, exp+"_vh_clean_2.ply_" + str(obj_id) + ".ply")
                mesh_gt = trimesh.load(obj_file)

            rec_meshfile = os.path.join(mesh_dir, "iteration_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")
            rec_meshfile_ref = os.path.join(mesh_dir_ref, "it_"+str(args.iteration)+"_obj"+str(obj_id)+".obj")

            mesh_rec = trimesh.load(rec_meshfile)
            mesh_ref = trimesh.load(rec_meshfile_ref) if os.path.exists(rec_meshfile_ref) else mesh_gt

            metrics = calc_3d_metric(mesh_rec, mesh_ref, N=N)  # for objs use 10k, for scene use 200k points
            if metrics is None:
                continue
            np.save(output_path + '/metric_obj{}.npy'.format(obj_id), np.array(metrics))
            metrics_3D[0].append(metrics[0])    # acc
            metrics_3D[1].append(metrics[1])    # comp
            metrics_3D[2].append(metrics[2])    # comp ratio 5cm
        metrics_3D = np.array(metrics_3D)
        np.save(output_path + '/metrics_3D_obj.npy', metrics_3D)
        print("metrics 3D obj \n Acc | Comp | Comp Ratio 5cm \n", metrics_3D.mean(axis=1))
        print("-----------------------------------------")
        print("finish exp ", exp)