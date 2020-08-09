import sys
import os
from pickle import load
import numpy as np
#import matplotlib.pyplot as plt
import colorsys
import trimesh

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def test_colormap():
    N = 5

    x = np.random.randn(40)
    y = np.random.randn(40)
    c = np.random.randint(N, size=40)

    # Edit: don't use the default ('jet') because it makes @mwaskom mad...
    plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    plt.show()

def test_sdf():
    with open('model/sdf_smpl.txt') as fp:
        ov = np.loadtxt(fp)
        v = ov[:,0]
        #v= v/v.max() + 0.1
        v = 1 - v/v.max()*0.9

        np.savez_compressed('pervertex_weight_sdf', pervertex_weight=v)

        mesh = trimesh.load('mesh.obj', process=False)
        for i in range(len(mesh.vertices)):
            mesh.visual.vertex_colors[i] = np.array([v[i]*255,0,0,255], dtype = np.uint8)
        mesh.show()         

    loaded = np.load('model/pervertex_weight_sdf.npz')
    v = loaded['pervertex_weight']
    #v[v>0.5] = v[v>0.5]*2
    v = v/v.max()
    hist = np.histogram(v,10)
    print( 'min error-{}, max error-{}, after normalization'.format(v.min(), v.max()) )

    mesh = trimesh.load('mesh.obj', process=False)
    for i in range(len(mesh.vertices)):
        mesh.visual.vertex_colors[i] = np.array([v[i]*255,0,0,255], dtype = np.uint8)
    mesh.show()  

def test_trimesh():
    mesh = trimesh.load('mesh.obj', process=False)

    for i in range(len(mesh.vertices)):
        mesh.visual.vertex_colors[i] = np.array([0,255,0,150], dtype = np.uint8)
    mesh.show()

    # for i in range(len(mesh.faces)):
    #     mesh.visual.face_colors[i] = np.array([0,255,0,255], dtype = np.uint8)
    # mesh.show()

def list_xmap(N):
    hue = np.linspace(0, 1, N, endpoint=True)
    # saturation = np.ones(N)
    # value = np.ones(N)
    rgb = np.ones((N,4), dtype=np.uint8)
    for i in range(N):
        rgb[i,0:3] = np.array(colorsys.hsv_to_rgb(hue[i], 1.0, 1.0)) * 255

    rgb[:,3] = 255
    return rgb

# Python program to illustrate the intersection 
# of two lists using set() method 
def intersection(lst1, lst2): 
	return list(set(lst1) & set(lst2)) 

def show_segment():

    with open('data/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = load(f)

    parts = sorted(vsegm.keys())
    print(intersection(vsegm['leftFoot'], vsegm['leftToeBase'])) 

    N = len(parts)
    cmap = list_xmap(N)
    # totalLen = 0
    # tmin = 0
    # tmax = 0
    # for part in parts:
    #     vs = vsegm[part]
    #     if tmin > min(vs):
    #         tmin = min(vs)
    #     if tmax < max(vs):
    #         tmax = max(vs)
    #     totalLen = totalLen + len(vs)
    #     print len(vs)

    # print totalLen, tmin, tmax

    mesh = trimesh.load('mesh.obj', process=False)
    i = 0
    for part in parts: 
        idx = vsegm[part]       
        tmp = np.zeros((len(idx), 4), dtype = np.uint8)
        color = tmp + cmap[i,:]
        mesh.visual.vertex_colors[idx] = color
        i = i + 1

    mesh.show()
    
if __name__ == '__main__':
    #test_colormap()
    show_segment()