import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from util.util import batch_rodrigues
from config import args
from SMPL import SMPL

class SurrealDataset(data.Dataset):
    '''
        load synthesized data:
                point clouds (synthesize) + shape & pose parameters

        Data argumentation: 
            1. randomly select npts points from input points

        todo: 
            more supervision
                use segmentation label
            data argumentatin
                noise
                sampling with one or two view directions  

        note:   
            normalization of pts
                should not have normalization, since we wish to learn shape, & there are adults and children.
                pointnet.pytorch does not have normalization, to be verified again 

    '''
    def __init__(self, root, npts = 2400, classification = 'regression', train = True):
        self.npts = npts
        self.root = root
        self.classification = classification
        self.data = []

        self.smpl = SMPL(args.smpl_model, obj_saveable = True)

        self.train = train
        if self.train:
            # self.path = ['spe_dataset_train_w_m3','spe_dataset_train_w_m4',
            #             'spe_dataset_train_w_m5','spe_dataset_train_w_m6']#, 'spe_dataset_train_w_m_selectPose']#, 'spe_dataset_train_w_m_bent']
            self.path = ['spe_dataset_train_w_m3', 'spe_dataset_train_w_m4', 'spe_dataset_train_w_m5']
            args.train_data = self.path
        else:
            self.path = ['spe_dataset_val_w_m_1']
            args.val_data = self.path

        for path in self.path:                
            tmp = os.path.join(self.root, path)
            jsonfile = os.path.join(tmp, 'data.json')

            with open(jsonfile) as json_file:  
                data = json.load(json_file)
                for p in data['people']:
                    self.data.append((p['name'], p['beta'], p['pose'], tmp))

    def data_arg_sampling(self, pts):
        ''' data argumentation 1 - random sampling input point cloud
                                the order of the index, i.e. choice, is also random    
            pts: (N, 3)
        '''
        # replace=False for avoiding to sample a point multiple times
        N = pts.shape[0]
        if self.npts > N:
            pts = torch.tensor( np.array([ pts ]) ).float()
        else:
            choice = np.random.choice(N, self.npts, replace=False) 
            pts = pts[choice, :]
            pts = torch.tensor( np.array([ pts ]) ).float()
        
        return pts

    def data_arg_rotate(self, pts, shape, pose, bRotate = False, 
                                anglex = 0.0, angley = 0.5 * np.pi, anglez = 0.0):
        '''
        data argumentation 2: rotation around x,y,z axis 
        up orientation is y+; face (front) orientation is z+

        output:
            pts: rotated points via matrix multiplication
            gtverts: reconstructed vertices by smpl using pose parameters according to the rotation

        Note: do not set bRotate = True, since surface of pts and that of gverts does not coincide exactly
        '''
        if bRotate:        
            pose[0,0]= anglex * np.random.random() # x axis
            pose[0,1]= angley * np.random.random()
            pose[0,2]= anglez * np.pi * np.random.random()
        
        gtverts, j3d, r = self.smpl(shape, pose, get_skin = True)
        
        def make_A(R, t):
            N = R.shape[0]
            # Rs is N x 3 x 3, ts is N x 3 x 1
            R_homo = nn.functional.pad(R, [0, 0, 0, 1, 0, 0])
            if t.is_cuda:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)
            else:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1))], dim = 1)
            return torch.cat([R_homo, t_homo], 2)    
        
        if bRotate:
            tmpA = self.smpl.J_transformed[0,0,:]
            tmpA[1]=0 #y
            # tmpA[0]=0    
            Rs = r[0,0,:,:].view(1,3,3)
            A = make_A(Rs, tmpA.view(1,3,1))    
            A = A.transpose(1,2)
            pts_homo = torch.cat([pts, torch.ones(Rs.shape[0], pts.shape[1], 1, device = self.smpl.cur_device)], dim = 2)        
            pts_homo = torch.matmul(pts_homo, A)
            pts = pts_homo[:, :, :3]# (N x 6890 x 4) =>  (N x 6890 x 3)

            # this is bad, since we do not know the root location of the human body
            # Rs = batch_rodrigues(pose_para[:3].view(-1,3)).view(3, 3)
            # Rs = Rs.transpose(1,2)
            # pts = torch.matmul(opts, Rs)

        return pts, pose, gtverts, j3d

    def __getitem__(self, index):
        '''
            output: 
                pts: N x 3;
                shape: 10
                pose: 72
                gtverts: 6890 x 3
                j3d: 19 x 3

        transfer the data to batch 1 format, handle, then back to normal format, such as
            pts: (1500*3) => (1*1500*3) =>(1500*3)
        '''
        data = self.data[index]
        shape = torch.tensor( np.array([data[1]]) ).float()
        pose = torch.tensor( np.array([data[2]]) ).float()

        fn = os.path.join(data[3], data[0] + '.pts')
        pts = np.loadtxt(fn).astype(np.float32)         
        #print("pts in getitem of datasets.py_", pts.shape)
        #print(fn)

        #################################
        pts = self.data_arg_sampling(pts)
        pts, pose, gtverts, j3d = self.data_arg_rotate(pts, shape, pose, False,
                                    0.5 * np.pi, # angle x
                                    np.pi, # angle y
                                    0.5 * np.pi)       
        
        results = {
                    'name': data[0],
                    'pts': torch.squeeze(pts, 0),
                    'shape': torch.squeeze(shape, 0),
                    'pose': torch.squeeze(pose, 0),
                    'gtmesh': torch.squeeze(gtverts, 0),
                    'j3d': torch.squeeze(j3d, 0),
                }
        if self.npts == 6890:
            results['pts'] = torch.squeeze(gtverts, 0)

        return results

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print('test')
    root = '/data/spe_database'
    data_set = SurrealDataset(root=root)
    
    print(len(data_set))
    data = data_set[0]
    pts = data['pts']
    shapepara, posepara = data['shape'], data['pose']
    print(pts.size(),shapepara.size(), shapepara.type(), posepara.size())
    if 'gtmesh' in data:
        verts, j3d = data['gtmesh'], data['j3d']

    data_set.smpl.save_obj(verts.cpu().numpy(), 'test.obj') 
    data_set.smpl.save_png(pts.cpu().numpy(), verts.cpu().numpy(), verts.cpu().numpy(), 'test.png')    
    print('dataset over')


    from torch.utils.data import DataLoader
    loader = DataLoader(
            dataset = data_set,
            batch_size = 16,
            shuffle = True,
            drop_last = True,
            pin_memory = True,
            num_workers = 4
        )
    loader = iter(loader)
    data = loader.next()
    pts, real_shape, real_pose = data['pts'], data['shape'], data['pose']
    if 'gtmesh' in data:
        verts, j3d = data['gtmesh'], data['j3d']
    print('dataloader over')