import random
import numpy as np
import torch
import os
import sys
sys.path.append(os.getcwd())
# from datasetFaust import *
#from model import *

import time
from sklearn.neighbors import NearestNeighbors
import trimesh
import visdom
import global_variables

from config import args
from model import SPENetSiam, SPENet
from util.util import copy_state_dict, weights_init
import reconstruct

def log(info, flag='a'):
    '''
        info: a string
        flag: w or a
    '''
    print(info)
    with open(os.path.join(args.outf, 'log.txt'), flag) as fp:
        fp.write(info + '\n')

def compute_correspondances(source_p, source_reconstructed_p, target_p, target_reconstructed_p):
    """
    Given 2 meshes, and their reconstruction, compute correspondences between the 2 meshes through neireast neighbors
    :param source_p: path for source mesh
    :param source_reconstructed_p: path for source mesh reconstructed
    :param target_p: path for target mesh
    :param target_reconstructed_p: path for target mesh reconstructed
    :return: None but save a file with correspondences
    """
    # inputs are all filepaths
    with torch.no_grad():
        source = trimesh.load(source_p, process=False)
        source_reconstructed = trimesh.load(source_reconstructed_p, process=False)
        target = trimesh.load(target_p, process=False)
        target_reconstructed = trimesh.load(target_reconstructed_p, process=False)

        # project on source_reconstructed
        neigh.fit(source_reconstructed.vertices)
        idx_knn = neigh.kneighbors(source.vertices, return_distance=False)

        #correspondances throught template
        closest_points = target_reconstructed.vertices[idx_knn]
        closest_points = np.mean(closest_points, 1, keepdims=False)


        # project on target
        if global_variables.opt.project_on_target:
            print("projection on target...")
            neigh.fit(target.vertices)
            idx_knn = neigh.kneighbors(closest_points, return_distance=False)
            closest_points = target.vertices[idx_knn]
            closest_points = np.mean(closest_points, 1, keepdims=False)

        # save output
        mesh = trimesh.Trimesh(vertices=closest_points, faces=source.faces, process=False)
        trimesh.io.export.export_mesh(mesh, "results/correspondences.ply")
        np.savetxt("results/correspondences.txt", closest_points, fmt='%1.10f')
        return

if __name__ == '__main__':

    args.batch_size = 1
    global_variables.opt = args
    vis = visdom.Visdom(port=8097, env=args.vis)

    # load network
    if args.network == 'SPENetPose':
        spenet = SPENetPose()
    elif args.network == 'SPENetBeta':
        spenet = SPENetBeta()
    elif args.network == 'SPENetSiam':
        spenet = SPENetSiam()
    else: #'SPENet'
        spenet = SPENet()
    global_variables.network = spenet

    if torch.cuda.is_available():
        global_variables.network.cuda()

    #global_variables.network.apply(weights_init)
    model_path = args.model
    if os.path.exists(model_path):
        copy_state_dict(
            global_variables.network.state_dict(), 
            torch.load(model_path),
            prefix = 'module.'
        )
    else:
        info = 'model {} not exist!'.format(model_path)
        log(info)

    global_variables.network.eval()

    neigh = NearestNeighbors(1, 0.4)
    args.manualSeed = random.randint(1, 10000) # fix seed
    # print("Random Seed: ", opt.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    start = time.time()
    print("computing correspondences for " + args.inputA + " and " + args.inputB)

    # Reconstruct meshes
    reconstruct.reconstruct(args.inputA)
    reconstruct.reconstruct(args.inputB)

    # Compute the correspondences through the recontruction
    compute_correspondances(args.inputA, args.inputA[:-4] + "FinalReconstruction.ply", args.inputB, args.inputB[:-4] + "FinalReconstruction.ply")
    end = time.time()
    print("ellapsed time is ", end - start, " seconds !")