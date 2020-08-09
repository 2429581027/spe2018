import numpy as np
import torch.utils.data
import torch.optim as optim
import sys
import os
sys.path.append(os.getcwd())
sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer =  ext.chamferDist()
from util.ply import *
from model import *

import global_variables
import trimesh

def log(info, flag='a'):
    '''
        info: a string
        flag: w or a
    '''
    print(info)
    with open(os.path.join(args.outf, 'log.txt'), flag) as fp:
        fp.write(info + '\n')

def test_orientation(input_mesh):
    """
    This fonction tests wether widest axis of the input mesh is the Z axis
    input mesh
    output : boolean or warning
    """
    point_set = input_mesh.vertices
    bbox = np.array([[np.max(point_set[:,0]), np.max(point_set[:,1]), np.max(point_set[:,2])], [np.min(point_set[:,0]), np.min(point_set[:,1]), np.min(point_set[:,2])]])
    extent = bbox[0] - bbox[1]
    if not np.argmax(np.abs(extent)) == 1:
        log("The widest axis is not the Y axis, you should make sure the mesh is aligned on the Y axis for the autoencoder to work (check out the example in /data)")
    return 

def clean(input_mesh):
    """
    This function remove faces, and vertex that doesn't belong to any face. Intended to be used before a feed forward pass in pointNet
    Input : mesh
    output : cleaned mesh
    """
    print("cleaning ...")
    print("number of point before : " , np.shape(input_mesh.vertices)[0])
    pts = input_mesh.vertices
    faces = input_mesh.faces
    faces = faces.reshape(-1)
    unique_points_index = np.unique(faces)
    unique_points = pts[unique_points_index]
    print("number of point after : " , np.shape(unique_points)[0])
    mesh = trimesh.Trimesh(vertices=unique_points, faces=np.array([[0,0,0]]), process=False)
    return mesh

def center(input_mesh):
    """
    This function center the input mesh using it's bounding box
    Input : mesh
    output : centered mesh and translation vector
    """
    bbox = np.array([[np.max(input_mesh.vertices[:,0]), np.max(input_mesh.vertices[:,1]), np.max(input_mesh.vertices[:,2])], [np.min(input_mesh.vertices[:,0]), np.min(input_mesh.vertices[:,1]), np.min(input_mesh.vertices[:,2])]])

    translation = (bbox[0] + bbox[1]) / 2
    translation[1] = bbox[1][1] # make the lowest y = 0 after centerization
    #points = input_mesh.vertices - translation    
    #t2 = np.array([ 0.00057989, -0.30316264,  0.02279274]) # center of smpl template
    #t2 = np.array([5.79890743e-04, -1.16184305e+00,  2.27927418e-02]) # centerB of smpl template
    
    t2 = np.array([ 0.0, -0.0,  0.0])
    points = input_mesh.vertices - translation + t2

    mesh = trimesh.Trimesh(vertices=points, faces=input_mesh.faces, process= False)
    return mesh, translation

def scale(input_mesh, mesh_ref):
    """
    This function scales the input mesh to have the same volume as a reference mesh Intended to be used before a feed forward pass in pointNet
    Input : file path
    mesh_ref : reference mesh path
    output : scaled mesh
    """
    area = np.power(mesh_ref.volume / input_mesh.volume, 1.0/3)
    mesh= trimesh.Trimesh( vertices =  input_mesh.vertices * area, faces= input_mesh.faces, process = False)
    return mesh, area

def regress(points, predict_theta):
    """
    search the latent space to global_variables. Optimize reconstruction using the Chamfer Distance
    :param points: input points to reconstruct
    :return pointsReconstructed: final reconstruction after optimisation
    
    note: far worse than the direct output of the network!!! Why?
    """
    points = points.data
    #points = points.detach()
    lrate = 0.001  # learning rate
    # define parameters to be optimised and optimiser
    input_param = nn.Parameter(predict_theta.data, requires_grad=True)
    global_variables.optimizer = optim.Adam([input_param], lr=lrate)
    #global_variables.optimizer = optim.Adam([input_param], lr=lrate,weight_decay = 0.0001)

    loss = 10
    i = 0

    #learning loop
    while np.log(loss) > -9 and i < global_variables.opt.nepoch:
        global_variables.optimizer.zero_grad()
        pointsReconstructed = global_variables.network.decode(input_param)  # forward pass
        dist1, dist2 = distChamfer(points.contiguous(), pointsReconstructed)
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        loss_net.backward()
        global_variables.optimizer.step()
        loss = loss_net.item()        
        if i % 100 == 0:
            print("log loss of reg: {}, loop: {}".format(np.log(loss),i))
        i = i + 1

    print("loss reg: {}, loop: {}".format(np.log(loss),i))

    with torch.no_grad():
        #theta, verts, j3d, codeBeta, codePose = global_variables.network.encode(points)
        if global_variables.opt.network == 'SPENet':
            pointsReconstructed = global_variables.network.decode_full(input_param) 
        else:
            pointsReconstructed = global_variables.network.decode(input_param) 

    return pointsReconstructed

def run(input, scalefactor):
    """
    :param input: input mesh to reconstruct optimally.
    :return: final reconstruction after optimisation

    """

    input, translation = center(input)

    ## Extract points and put them on GPU    
    points = torch.from_numpy(input.vertices.astype(np.float32)).contiguous().unsqueeze(0)
    points = points.contiguous()
    points = points.cuda()

    # Get a low resolution PC to find the best reconstruction after a rotation on the Y axis
    random_sample = np.random.choice(np.shape(input.vertices)[0], size=10000)
    points_LR = torch.from_numpy(input.vertices[random_sample].astype(np.float32)).contiguous().unsqueeze(0)
    points_LR = points_LR.contiguous()
    points_LR = points_LR.cuda()

    with torch.no_grad():
        #(predict_theta, predict_verts, predict_j3d, decoded_verts) = global_variables.network(points_LR)
        beta, theta, verts, j3d, codeBeta, codePose = global_variables.network.encode(points_LR)

        if global_variables.opt.network == 'SPENet':
            decoded_verts = global_variables.network.decode(codePose)
        else: # SPENetSiam
            decoded_verts = global_variables.network.decode(torch.cat([codeBeta, codePose], 1))

    pointsReconstructed = verts 
    # create initial guess
    mesh = trimesh.Trimesh(vertices=(pointsReconstructed[0].data.cpu().numpy() + translation)/scalefactor, 
                            faces=global_variables.network.smpl.faces, process = False)


    # print("start regression...")
    # if global_variables.opt.network == 'SPENet':
    #     pointsReconstructed1 = regress(points, codePose)   
    # else: # SPENetSiam
    #     pointsReconstructed1 = regress(points, torch.cat([codeBeta, codePose], 1))   

    with torch.no_grad():
        beta, theta, verts, j3d, codeBeta, codePose = global_variables.network.encode(points)
        if global_variables.opt.network == 'SPENet':
            if global_variables.opt.HR:
                decoded_verts = global_variables.network.decode_full(codePose)
            else:
                decoded_verts = global_variables.network.decode(codePose)
        else: # SPENetSiam
            if global_variables.opt.HR:
                decoded_verts = global_variables.network.decode_full(torch.cat([codeBeta, codePose], 1))
            else:
                decoded_verts = global_variables.network.decode(torch.cat([codeBeta, codePose], 1))
    pointsReconstructed1 =  decoded_verts
       
    # create optimal reconstruction    
    # bbox_max, _ = torch.max(pointsReconstructed1, 1)
    # bbox_min, _ = torch.min(pointsReconstructed1, 1)
    # t1 = (bbox_max + bbox_min) / 2.0 #(B, 1, 3)
    # pointsReconstructed1 = pointsReconstructed1 - t1.unsqueeze(1)

    if global_variables.opt.HR:
        faces_tosave = global_variables.network.mesh_HR.faces
    else:
        faces_tosave = global_variables.network.smpl.faces
    meshReg = trimesh.Trimesh(vertices=(pointsReconstructed1[0].data.cpu().numpy()  + translation)/scalefactor, 
                                faces=faces_tosave, process=False)                        

    print("... Done!")
    return mesh, meshReg

def save(mesh, mesh_color, path, red, green, blue):
    """
    Home-made function to save a ply file with colors. A bit hacky
    """
    to_write = mesh.vertices
    b = np.zeros((len(mesh.faces),4)) + 3
    b[:,1:] = np.array(mesh.faces)
    points2write = pd.DataFrame({
        'lst0Tite': to_write[:,0],
        'lst1Tite': to_write[:,1],
        'lst2Tite': to_write[:,2],
        'lst3Tite': red,
        'lst4Tite': green,
        'lst5Tite': blue,
        })
    write_ply(filename=path, points=points2write, as_text=True, text=False, faces = pd.DataFrame(b.astype(int)), color = True)    

def reconstruct(input_p):
    """
    Recontruct a 3D shape by deforming a template
    :param input_p: input path
    :return: None (but save reconstruction)
    """
    input = trimesh.load(input_p, process=False)
    scalefactor = 1.0
    if global_variables.opt.scale:
        input, scalefactor = scale(input, global_variables.mesh_ref_LR) #scale input to have the same volume as mesh_ref_LR
    if global_variables.opt.clean:
        input = clean(input) #remove points that doesn't belong to any edges
    test_orientation(input)

    mesh, meshReg = run(input, scalefactor)

    if not global_variables.opt.HR:
        red = global_variables.red_LR
        green = global_variables.green_LR
        blue = global_variables.blue_LR
        mesh_ref = global_variables.mesh_ref_LR
    else:
        blue = global_variables.blue_HR
        red = global_variables.red_HR
        green = global_variables.green_HR
        mesh_ref = global_variables.mesh_ref

    save(mesh, global_variables.mesh_ref_LR, input_p[:-4] + "InitialGuess.ply", global_variables.red_LR, global_variables.green_LR, global_variables.blue_LR )
    #save(meshReg, global_variables.mesh_ref_LR, input_p[:-4] + "FinalReconstruction.ply", global_variables.red_LR, global_variables.green_LR, global_variables.blue_LR )
    save(meshReg, mesh_ref, input_p[:-4] + "FinalReconstruction.ply", red, green, blue)    
    #meshReg.export(input_p[:-4] + 'FinalReconstruction.ply')