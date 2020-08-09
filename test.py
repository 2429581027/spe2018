import os
import random
import torch
import torch.nn as nn
import numpy as np
import trimesh

from config import args

def batch_3d_l2_loss(verts, predict_verts, ave_pervert, pvw):
    B = verts.shape[0]
    N = verts.shape[1]
    k = B * N * 3 + 1e-8    

    dif = abs(verts - predict_verts)
    tmp = dif.sum(2) * pvw
    tmp = tmp.sum(0)
    
    ave_pervert += tmp.numpy()/B
    #self.ave_pervert
    return  tmp.sum() / k

def test_batch_3d_l2_loss():
    from torch.autograd import Variable
    pvw = np.ones(6890) #*2
    #np.savez_compressed(os.path.join(args.outf, 'pervertex_weight'), pervertex_weight=pvw)
    pvw = torch.from_numpy(pvw).float()
    pvw = pvw.repeat(2,1)

    ################
    # verts = Variable(torch.ones(2,6890,3))
    # predict_verts = Variable(torch.zeros(2,6890,3))
    verts = torch.from_numpy( np.random.rand(2,6890,3) ).float()
    predict_verts = torch.from_numpy( np.random.rand(2,6890,3) ).float()   
    ave_pervert = np.zeros(6890)

    nruns = 3
    for i in range(nruns):
        loss = batch_3d_l2_loss(verts, predict_verts, ave_pervert, pvw)

    v = ave_pervert / nruns

    v=(v-v.min())/(v.max()-v.min())*255
    print(v.max())
    
    np.savez_compressed('pervertex_weight', pervertex_weight=v)
    loaded = np.load('pervertex_weight.npz')
    vv = loaded['pervertex_weight']
    print(vv.max())

    mesh = trimesh.load('mesh.obj', process=False)

    for i in range(len(mesh.vertices)):
        mesh.visual.vertex_colors[i] = np.array([vv[i],0,0,255], dtype = np.uint8)
    mesh.show()

def test_query_group1():
    #(B, C=3, npoint, nsample) 2 (B, npoint, nsample*C)
    B, C, npoint, nsample = 2, 3, 4, 2
    a1 = torch.range(1, B*npoint*nsample*C) # 1--48
    target = a1.reshape((B, npoint, nsample*C))
    print(target)
    # target[0,0,:] # 1--6  # target[0,3,:] # 19--24  # target[1,0,:] # 25--30

    tmp = target.reshape(B, npoint, nsample, C)
    #tmp = target.reshape(B, npoint, C, nsample)
    # print(tmp[0,0]) # 1--6  # print(tmp[0,1]) # 7--12
    source = tmp.transpose(2,3).transpose(1,2)
    print(source)
    print(source[0,:,0,0])
    print(source[0,:,0,1])
    
    target = source.transpose(1,2).transpose(2,3).reshape((B, npoint, nsample*C))
    print(target)

    print('test_query_group1 finish')

def test_query_group():
    from pointnet2.utils.pointnet2_utils import QueryAndGroup
    #mesh_gt = trimesh.load('0_p_79_f_1700_s_0_m.obj', process=False)
    mesh = trimesh.load('0_p_79_f_1700_s_0_m_.obj', process=False)    
    verts = torch.tensor(mesh.vertices).float() 
    pts = np.loadtxt('0_p_79_f_1700_s_0_m.pts').astype(np.float32)    
    pts = torch.tensor(pts).float()
    if torch.cuda.is_available():
        verts = verts.cuda()  
        pts = pts.cuda()  
    verts = verts.view(-1, verts.shape[0], 3)
    verts = torch.cat([verts, verts], 0)
    pts = pts.view(-1, pts.shape[0], 3)
    pts = torch.cat([pts, pts], 0)
    B = 2

    npoint = verts.shape[1]
    qg = QueryAndGroup(radius = 0.2, nsample = 32, use_xyz = False)
    nfeat = qg(xyz = pts, new_xyz = verts.contiguous(), features = pts.transpose(1,2).contiguous())
    nfeat = nfeat.transpose(1,2).transpose(2,3).reshape((B, npoint, qg.nsample*3))
    nfeat = nfeat[0]
    samples = nfeat.view(-1, 3).cpu().numpy()

    scene = mesh.scene()
    #for i in range(npoint):
        
    scene.add_geometry(trimesh.PointCloud(samples))
    scene.show()

    print('test_query_group finish')

if __name__ == '__main__':
    test_query_group()
    #test_query_group1()
    #test_sdf()
    #test_batch_3d_l2_loss()