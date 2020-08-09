import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../')
sys.path.append('../datageneration/')
import copy
import random
import datetime
import json
import trimesh
import numpy as np
import torch
import torch.nn as nn
from config import args
from SMPL import SMPL
from datageneration.generate_data import SpeDataGenerator
from model import SPENet, SPENetSiam, SPENetBeta, SPENetPose
from util.util import copy_state_dict, batch_rodrigues, center_pts_tensor, center_pts_np_array
from util.robustifiers import GMOf

def bbox_pts_np_array(inpts):
    bbox = np.array([[np.max(inpts[:,0]), np.max(inpts[:,1]), np.max(inpts[:,2])], [np.min(inpts[:,0]), np.min(inpts[:,1]), np.min(inpts[:,2])]])
    return bbox

class SmplPoseTransfer(SpeDataGenerator):
    def __init__(self, dataroot):
        self.data_root = dataroot       
        self.smpl_model = args.smpl_model
        self.gender = args.gender

        # load pose database
        database_path = os.path.join(self.data_root, 'smpl_data.npz')
        self.database = np.load(database_path)

        # load model
        self.smpl = SMPL(self.smpl_model, obj_saveable = True) #'../model/neutral_smpl_with_cocoplus_reg.txt'
        if torch.cuda.is_available():
            self.smpl = self.smpl.cuda()    

        self.network = SPENet()
        if torch.cuda.is_available():
            self.network.cuda()

        model_path = args.model
        if os.path.exists(model_path):
            copy_state_dict(
                self.network.state_dict(), 
                torch.load(model_path),
                prefix = 'module.'
            )
        else:
            info = 'model {} not exist!'.format(model_path)
            print(info)

        self.network.eval()


    def transfer_para_lr(self):
        if self.gender == 'm':
            betas = self.database['maleshapes']
        else:
            betas = self.database['femaleshapes'] 

        pose_files = [i for i in self.database.keys() if "pose" in i]

        #beta_id = np.random.randint(np.shape(betas)[0]) 
        beta_id = 558 # thin
        beta = betas[beta_id]

        #beta_id_to = np.random.randint(np.shape(betas)[0]) 
        beta_id_to = 410 # fat
        beta_to = betas[beta_id_to]

        pose_file_id = np.random.randint(len(pose_files))           
        pose_ = self.database[pose_files[pose_file_id]]
        frame_id = np.random.randint(np.shape(pose_)[0])  
        scene = trimesh.Scene()
        j = 0
        for i in range(frame_id, np.shape(pose_)[0], 10):
            pose = pose_[i]
            points = self._generate_surreal_(pose, beta, bRotate = False) 
            points, translation = center_pts_np_array( points)
            mesh = trimesh.Trimesh(points, self.smpl.faces)

            points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
            points = points.contiguous().cuda()
            with torch.no_grad():
                betaLearnt, thetaLearnt, verts, j3d, codeBeta, codePose = self.network.encode(points)

            points_to = self._generate_surreal_(thetaLearnt[0].detach().cpu().numpy(), beta_to, bRotate = False) 
            points_to, translation = center_pts_np_array( points_to)
            x = bbox_pts_np_array(mesh.vertices)[0][0]
            points_to[:,0] = points_to[:,0] - 2*x
            mesh_to = trimesh.Trimesh(points_to, self.smpl.faces)     
            

            scene.add_geometry(mesh)

            # tmp = verts[0].detach().cpu().numpy()
            # tmp[:,0] = tmp[:,0] + 2*x
            # samples = trimesh.PointCloud(tmp)
            # scene.add_geometry(samples)
            scene.add_geometry(mesh_to)
            
            
            try:
                # save a render of the object as a png
                png = scene.save_image(resolution=[480, 480], visible=True)                
                imname = os.path.join(args.outf, '{}{}'.format(j,'.png'))
                with open(imname, 'wb') as f:
                    f.write(png)
                    f.close()
            except BaseException as E:
                print("unable to save image", str(E))  
                            
            #scene.show()
            scene.geometry.clear()
            j = j + 1

    def transfer_para_hr(self):
        scene = trimesh.Scene()
        if self.gender == 'm':
            betas = self.database['maleshapes']
        else:
            betas = self.database['femaleshapes'] 

        pose_files = [i for i in self.database.keys() if "pose" in i]

        #beta_id = np.random.randint(np.shape(betas)[0]) 
        beta_id = 558 # thin
        beta = betas[beta_id]

        #beta_id_to = np.random.randint(np.shape(betas)[0]) 
        beta_id_to = 410 # fat
        #beta_to = np.zeros(10)
        beta_to = betas[beta_id_to]         
        pose_to = np.zeros(72)
        points_to = self._generate_surreal_(pose_to, beta_to, bRotate = False) 
        points_to, translation = center_pts_np_array( points_to)
        points_to = torch.from_numpy(points_to.astype(np.float32)).contiguous().unsqueeze(0)
        points_to = points_to.contiguous().cuda()        
        with torch.no_grad():
            betaLearnt, thetaLearnt, verts, j3d, codeBeta_to, codePose_to = self.network.encode(points_to)  
            verts_to = self.network.decode(torch.cat([codeBeta_to, codePose_to], 1))       
        # mesh = trimesh.Trimesh(verts[0].cpu().numpy(), self.smpl.faces)
        # mesh_to = trimesh.Trimesh(verts_to[0].cpu().numpy(), self.smpl.faces)
        # #scene.add_geometry(mesh)
        # scene.add_geometry(mesh_to)
        # scene.show()
        # scene.geometry.clear()


        pose_file_id = np.random.randint(len(pose_files))           
        pose_ = self.database[pose_files[pose_file_id]]
        #frame_id = np.random.randint(np.shape(pose_)[0])  
        frame_id = 0
        
        for i in range(frame_id, np.shape(pose_)[0], 100):
            pose = pose_[i]
            points = self._generate_surreal_(pose, beta, bRotate = False) 
            points, translation = center_pts_np_array( points)
            mesh = trimesh.Trimesh(points, self.smpl.faces)

            points = torch.from_numpy(points.astype(np.float32)).contiguous().unsqueeze(0)
            points = points.contiguous().cuda()
            with torch.no_grad():
                betaLearnt, thetaLearnt, verts, j3d, codeBeta, codePose = self.network.encode(points)
                verts_to = self.network.decode(torch.cat([codeBeta_to, codePose], 1))                   
            
            verts_to = verts_to[0].cpu().numpy()
            x = bbox_pts_np_array(mesh.vertices)[0][0]
            verts_to[:,0] = verts_to[:,0] - 2*x
            mesh_to = trimesh.Trimesh(verts_to, self.smpl.faces)     
            
            points_too = self._generate_surreal_(thetaLearnt[0].data.cpu().numpy(), beta_to, bRotate = False) 
            points_too, translation = center_pts_np_array( points_too)
            x = bbox_pts_np_array(mesh.vertices)[0][0]
            points_too[:,0] = points_too[:,0] - 4*x
            mesh_too = trimesh.Trimesh(points_too, self.smpl.faces)     

            scene.add_geometry(mesh)
            scene.add_geometry(mesh_to)
            scene.add_geometry(mesh_too)
            scene.show()
            scene.geometry.clear()

if __name__ == '__main__':
    args.model = './model/SPENet_pointnetmini_PointGenCon_108_0.043558_s0.039577_p0.002866_3d0.000499_decoded0.000263_j0.00035_r0.000010.pkl'
    if not os.path.exists(args.smpl_model):
        args.smpl_model = '../model/neutral_smpl_with_cocoplus_reg.txt'      

    spt = SmplPoseTransfer(args.data_root)
    
    args.outf = '{}_{}_{}'.format(args.outf, 'pose_transfer', datetime.datetime.now().strftime('%m-%d-%H:%M'))
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    spt.transfer_para_lr()
    #spt.transfer_para_hr()