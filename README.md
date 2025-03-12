[comment]: <> (# Category-Level Neural Field for Reconstruction of Partially Observed Objects in Indoor Environment)

<p align="center">

  <h1 align="center">Category-Level Neural Field for Reconstruction of Partially Observed Objects in Indoor Environment</h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://ieeexplore.ieee.org/document/10565984">Paper</a> | <a href="https://arxiv.org/abs/2406.08176">arXiv</a> | <a href="https://www.youtube.com/watch?v=f1YA10qoAwc">Video</a></h3>
  <div align="center"></div>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="60%">
  </a>
</p>
<p align="center">
Our method reconstructs objects using category-level models. Objects belonging to the same category share common shape properties, which help to reconstruct unobserved parts plausibly. On the other hand, unobserved parts of objects that are reconstructed by the object-level model tend to be over-smooth or fail to recover complete geometry.



## Install
Create virtual environment with 
```bash
conda env create -f environment.yml
```

We use [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) for category-level registration process.  
Follow [Instructions for installing TEASER++ python bindings](https://teaser.readthedocs.io/en/latest/installation.html#compilation-and-installation) after you activate the virtual environment. 


## Data
### Dataset
We tested our method on Replica and ScanNet. We follow [vMAP](https://github.com/kxhit/vMAP?tab=readme-ov-file#dataset) to prepare the dataset.
The structure of the dataset is outlined as follows:
```
Datasets/
├── Replica/
│   ├── room_0/
│   │   └── habitat/
│   │   └── sequences/
│   └── ...
└── ScanNet/
│   ├── scene0066_00/
│   │   └── habitat/
│   │   └── sequences/
│   └── ...
```

### Checkpoints of pretrained vMAP (offline version we implemented) 
Our method requires pretrained vMAP models, which serve as pretrained object-level neural field models. We provide [checkpoints](https://drive.google.com/file/d/1e1nrER3aLdI4WjLXqfI6DAh1FYeo15pV/view?usp=sharing) for all scenes we reported in our paper.

### Sample dataset
Our system applies category-level registration (subcategorization, registration) before train each category-level models.
To help you avoid this tedious process and see the result quickly, we provide a [sample dataset](https://drive.google.com/file/d/1loKzrhiug4yPVDcynMVSK35868IvUY6P/view?usp=sharing) for **Replica room_0** and **ScanNet scene0066_00**.


## Config

Update dataset paths (and / or) other training hyper-parameters in the config files in `configs/.json`.
```json
"dataset": {
        "path": "path/to/ims/folder/",
    }
```


## Run
To train category-level neural field models, run
```bash
python train.py --config configs/Replica/config_replica_room0.json --logdir logs/Replica/room0
```


## Evaluation
To evaluate the quality of reconstructed scenes, run 
```bash
python metric/eval_3D_scene.py --data_dir /path/to/dataset --log_dir /path/to/log_directory
```


## Acknowledgement
Our code is sourced and modified from [vMAP](https://github.com/kxhit/vMAP). We thank a lot for their open-source code.<br/>
Our category-level neural field models are sourced and modified from [CodeNeRF](https://github.com/wbjang/code-nerf).


## Citation
If you found this code/work to be useful in your own research, please considering citing the following:
```bibtex
@article{lee2024category,
  title={Category-Level Neural Field for Reconstruction of Partially Observed Objects in Indoor Environment},
  author={Lee, Taekbeom and Jang, Youngseok and Kim, H Jin},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
