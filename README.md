# spe2018

## Contents
* [1. Environment preparation](#1-environment-preparation)
* [2. Data preparation](#2-data-preparation)
* [3. Training](#3-training)

## 1 Environment preparation
a) Install miniconda 3
[//]: # "[install miniconda3-4.5.4, which is for python 3.6, not 3.7.] (https://repo.continuum.io/miniconda/) you can try install latested miniconda3 with python 3.7"
update it if necessary.
``` shell
conda update conda
```
b) Prepare a python 3.6 environment
Create a python 3.6 environment (denoted by py36), if the base environment of miniconda3 is not python 3.6, which is usually 3.7 now.
``` shell
conda create -n py36 python=3.6
```
c) Intall some libraries: trimesh 2.33, pytorch >= 0.4, opencv. (trimesh installation, refer to https://trimsh.org/install.html)
``` shell
    # 
    conda install -c anaconda numpy scipy

    # h5py for pytorch_HMR, matplotlib for pointnet, pyglet for trimesh    
    conda install -c conda-forge h5py matplotlib trimesh pyglet
 
    # pytorch 
    conda install pytorch torchvision -c pytorch

```
d) [It is not used now. so ignore this step.] Install pointnet++(pointnet2) for pytorch: https://github.com/erikwijmans/Pointnet2_PyTorch. It will compile some c++ code.
``` shell
    mkdir build && cd build
    cmake .. && make
    git submodule update --init --recursive
```
``` shell
    # tqdm and natsort for pointnet ++. 
    # conda install -c conda-forge tqdm natsort
    # pip install sacred
```
    
## 2 Data preparation
Refer to [./datageneration/readme.md](/datageneration/readme.md)

## 3 Training 
a) in `./`, unzip the [model.zip](https://pan.baidu.com/s/1PUv5kUydmx5RG1E0KsQBkw)
Only `neutral_smpl_with_cocoplus_reg.txt` (112M) and `neutral_smpl_mean_params.h5` are necessary.

b) Train
``` shell
python ./train.py
```

## mesh & pointcloud library or display
    1. open3d 0.3 不可调试，显示极棒，只能读取某种ply（obj，和meshlab的ply都读不了）,改写费劲。
        conda install -c open3d-admin open3d
    2. trimesh 2.33 不可调试，实用。
        python 2.7 ok, python 3.6 ok
        conda install -c conda-forge trimesh
        conda install -c conda-forge pyglet

    3. mayavi
        its dependencies are unfortunately rather heavy and failed to install for python 3.6;
        works for python 2.7
        conda install -c anaconda mayavi

    4. pymesh 没试，没时间，trimesh+mayavi暂时够了。
