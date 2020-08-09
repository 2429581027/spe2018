# Data generation
    
## 1 Download smpl_data.npz
a) [Apply username & passowrd for downloading SMPL](http://smpl.is.tue.mpg.de/downloads).

c) With the credentials (username & passowrd), download smpl_data.npz (2.5G). It contains `femaleshapes.npy`, `maleshapes.npy` & massive poses, such as `pose_01_01.npy`.
and place it in `../../../data/smpl_data` or `/Users/jjcao/data/smpl_data`.
``` shell
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```

## 2 database generation 
``` shell
python ./datageneration/generate_data.py
# python ./datageneration/2smpl_mesh.py (outdated)
# python ./datageneration/generate_surreal.py #(outdated, & this need python 2)
```

<!--- 注释，bak
## 1 Python 2 environment preparation
Create an environment: python 2.7 + chumpy 0.67.6 + Numpy & Scipy + opencv 3.4
a) Install miniconda3
b) After miniconda is installed, create an python 2.7 environment: 
``` shell
conda update conda
conda create -n py27 python=2.7
```
c) In the py27 environment, install opencv, numpy and chumpy: 
``` shell
conda install opencv 
conda install -c anaconda numpy
pip install chumpy
``` 
References: [Install opencv on Mac Mac](https://medium.com/init27-labs/installation-of-opencv-using-anaconda-mac-faded05a4ef6)
--->

<!---
## 2 Install SMPL (which need python2 + chumpy 0.67.6 + Numpy & Scipy + opencv 3.4)
a) Download [SMPL for python users, 1.0.0](http://smpl.is.tue.mpg.de/downloads). 
b) Unzip downloaded SMPL_python_v.1.0.0.zip to `./datageneration` folder, then you have `./datageneration/smpl`
c) With the same credentials (username & passowrd), download the remaining necessary SMPL data (smpl_data.npz, 2.5G). It contains `femaleshapes.npy`, `maleshapes.npy` & massive poses, such as `pose_01_01.npy`.
and place it in `../../../data/smpl_data` or `/Users/jjcao/data/smpl_data`.
``` shell
./download_smpl_data.sh /path/to/smpl_data yourusername yourpassword
```
--->
