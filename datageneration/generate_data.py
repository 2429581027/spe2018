'''
    file:   generate_data.py

    Generate human database from pose database, smpl_data.npz, 
        which contains: 2,667 (1261 pose_* + 1406 pose_ung_*) pose sequence files, 
        each contains a series of continue poses. There are 4323000 pose in total.
    We randomly sample the pose and beta parameters in the same way as 3D-CODED.

    date:   2018_11_06
    author: Junjie Cao
'''


import sys
import os
sys.path.append(os.getcwd())
sys.path.append('../')
import copy
import random
import json
import trimesh
import numpy as np
import torch
import torch.nn as nn
from config import args
from SMPL import SMPL
from datageneration.resampling import mesh2points

def resample(mesh, targetnpts, ptsname, imgname):
    '''
    '''
    samples, _, _ = mesh2points(mesh, targetnpts)      
    with open(ptsname, 'w') as fp:
        for v in samples:
            fp.write('%f %f %f\n' % (v[0], v[1], v[2]))  
        fp.close()
 
    # print("vertices of mesh".format(mesh.vertices.shape[0]) )
    # print("pts after".format(samples.shape[0]) )

    ## same image
    scene = mesh.scene()
    scene.add_geometry(trimesh.PointCloud(samples))

    # viewer = scene.show()
    # viewer.set_location(10, 10)
    
    try:
        # save a render of the object as a png
        png = scene.save_image(resolution=[480, 480], visible=True)
        with open(imgname, 'wb') as f:
            f.write(png)
            f.close()
    except BaseException as E:
        print("unable to save image", str(E))  

def print_pose_statics(betas, poses, database):
    print(len(poses))
    num_poses= 0
    for i in poses:
        num_poses = num_poses + np.shape(database[i])[0]
    print('Number of poses ' + str(num_poses))
    print('Number of betas ' + str(np.shape(betas)[0]))
    
    return

class SpeDataGenerator(object):
    def __init__(self, dataroot, outpath_train, outpath_val, op = 'generate'):
        self.data_root = dataroot
        self.human_count = args.human_count
        self.sample_count = args.sample_count
        self.outpath_train = outpath_train
        self.outpath_val = outpath_val
        self.gender = args.gender
        self.data_type = args.data_type
        self.smpl_model = args.smpl_model
        self.view_directions = None # None for do not use it
        self.bent = False

        if op == 'generate':
            # load pose database
            database_path = os.path.join(DATA_ROOT, 'smpl_data.npz')
            self.database = np.load(database_path)

            # load model
            self.smpl = SMPL(self.smpl_model, obj_saveable = True) #'../model/neutral_smpl_with_cocoplus_reg.txt'
            if torch.cuda.is_available():
                self.smpl = self.smpl.cuda()      

    def setup_jason(self):
        #################### setting
        train_dict = {} 
        
        train_dict['info'] = {} # dict
        train_dict['info']['nb_generated_humans'] = self.human_count # 20000 for male and 20000 for female
        train_dict['info']['nb_resample'] = self.sample_count
        train_dict['info']['resample_method'] = 'uniform'
        train_dict['info']['smpl'] = self.smpl_model

        train_dict['info']['male_shape'] = 'maleshapes' # maleshapes.npy
        train_dict['info']['female_shape'] = 'femaleshapes' # femaleshapes.npy
        # if self.gender == 'm':
        #     train_dict['info'].pop('female_shape', None)
        # elif self.gender == 'f':
        #     train_dict['info'].pop('male_shape', None)      

        train_dict['info']['type'] = 'train'
        train_dict['people'] = [] # list

        #####
        val_dict = copy.deepcopy(train_dict)
        val_dict['info']['nb_generated_humans'] = min(self.human_count, 100) # 100 for male and 100 for female
        val_dict['info']['type'] = 'validation'
            
        return train_dict, val_dict 

    def _generate_surreal_(self, pose, beta, bRotate = True, 
                                    anglex = 0.0, angley = 0.5 * np.pi, anglez = 0.0):
        '''
        This function generation 1 human using a random pose and shape estimation from surreal
        
        todo:
            1. generate segmentation/part label at the same time.

        '''
        pose[0:3]=0.0 # cancel global transformation, since the SMPL coordinate frame is very strange!!!
        if bRotate:        
            pose[0]= anglex * np.random.random() # x axis
            pose[1]= angley * np.random.random()
            pose[2]= anglez * np.random.random()

        tpose = torch.tensor(np.array([pose])).float()
        tbeta = torch.tensor(np.array([beta])).float()
        if torch.cuda.is_available():
            tbeta = tbeta.cuda() 
            tpose = tpose.cuda()  
            
        #############
        verts, j, r = self.smpl(tbeta, tpose, get_skin = True)

        verts = verts[0].cpu().numpy()

        #normalize
        # centroid = np.expand_dims(np.mean(point_set[:,0:3], axis = 0), 0) #Useless because dataset has been normalised already
        # point_set[:,0:3] = point_set[:,0:3] - centroid

        return verts

    def _get_random_(self, poses, betas, beta_pose_info):
        '''     
            info is a list, like [{'poseid': 168, 'frameid': 175, 'betaid': -1}]
        '''
        if beta_pose_info is None:
            beta_id = np.random.randint(np.shape(betas)[0])        
            pose_id = np.random.randint(len(poses))                
            pose_ = self.database[poses[pose_id]]
            frame_id = np.random.randint(np.shape(pose_)[0])        
        else:
            id = np.random.randint(len(beta_pose_info))
            info = beta_pose_info[id]

            beta_id = info['betaid']
            if beta_id == -1: 
                beta_id = np.random.randint(np.shape(betas)[0])            

            pose_id = info['poseid']
            pose_ = self.database[poses[pose_id]]
            frame_id = info['frameid']
            
        beta = betas[beta_id]
        pose = pose_[frame_id]
        name = 'p_{}_f_{}_s_{}'.format(pose_id, frame_id, beta_id)
        return pose, beta, name
    def _bent_pose(self, pose):
        '''
            from 3D-CODED
        '''
        if self.bent:
            a = np.random.randn(12)

            pose[1] = 0
            pose[2] = 0
            pose[3] = -1.0 + 0.1*a[0]
            pose[4] = 0 + 0.1*a[1]
            pose[5] = 0 + 0.1*a[2]
            pose[6] = -1.0 + 0.1*a[0]
            pose[7] = 0 + 0.1*a[3]
            pose[8] = 0 + 0.1*a[4]
            pose[9] = 0.9 + 0.1*a[6]
            pose[0] = - (-0.8 + 0.1*a[0] )
            pose[18] = 0.2 + 0.1*a[7]
            pose[43] = 1.5 + 0.1*a[8]
            pose[40] = -1.5 + 0.1*a[9]
            pose[44] = -0.15 
            pose[41] = 0.15
            pose[48:54] = 0
        
        return pose
        
    def _generate_database_surreal_(self, train_dict, val_dict, 
                                    gender='m', offset=0, bRotate = False, info=None):

        nb_generated_humans = train_dict['info']['nb_generated_humans']
        targetnpts = train_dict['info']['nb_resample']

        if gender == 'm':
            betas = self.database['maleshapes']
        else:
            betas = self.database['femaleshapes']

        poses = [i for i in self.database.keys() if "pose" in i]
        #print_pose_statics(betas, poses, self.database)

        ##################
        # TRAIN DATA
        stage = train_dict['info']['type']
        for i in range(nb_generated_humans):
            pose, beta, name = self._get_random_(poses, betas, info)
            pose = self._bent_pose(pose)

            verts = self._generate_surreal_(pose, beta, bRotate = bRotate, 
                                    anglex = 0.0, angley = 0.5 * np.pi, anglez = 0.0)
            #smpl.save_obj(verts, os.path.join(outpath_train, outname + '.obj'))

            faces = self.smpl.faces
            mesh = trimesh.Trimesh(verts, faces)
         
            if self.view_directions is not None:
                normals = mesh.face_normals
                mask = np.zeros(len(faces), dtype=bool)
                for j in range(len(faces)):
                    for k in range(np.shape(self.view_directions)[0]):
                        vd = self.view_directions[k,:]
                        if np.dot(vd, normals[j,:]) < 0.0:
                            mask[j] = True
                mesh.faces = mesh.faces[mask, ...]
     
            outname = '{}_{}_{}'.format(offset + i, name, gender)        
            
            if self.sample_count != 0: 
                resample(mesh, targetnpts, os.path.join(self.outpath_train, outname + '.pts'), 
                                    os.path.join(self.imgpath_train, outname + '.png'))
            
            train_dict['people'].append({  
                            'beta': beta.tolist(),
                            'pose': pose.tolist(),
                            'name': outname,
                        })  
            print('{} in total {} for male {} in {}'.format(offset + i, nb_generated_humans, gender, stage))

        #VAL DATA
        if val_dict is None: return

        stage = val_dict['info']['type']
        nb_generated_humans = val_dict['info']['nb_generated_humans']  

        for i in range(nb_generated_humans):
            pose, beta, name = self._get_random_(poses, betas, info)    
            pose = self._bent_pose(pose)
            verts = self._generate_surreal_(pose, beta, bRotate = bRotate, 
                                    anglex = 0.0, angley = 0.5 * np.pi, anglez = 0.0)

            faces = self.smpl.faces
            mesh = trimesh.Trimesh(verts, faces)
            if self.view_directions is not None:
                normals = mesh.face_normals
                mask = np.zeros(len(faces), dtype=bool)
                for j in range(len(faces)):
                    for k in range(np.shape(self.view_directions)[0]):
                        vd = self.view_directions[k,:]
                        if np.dot(vd, normals[j,:]) < 0.0:
                            mask[j] = True
                mesh.faces = mesh.faces[mask, ...]

            outname = '{}_{}_{}'.format(offset + i, name, gender)

            if self.sample_count != 0: 
                resample(mesh, targetnpts, os.path.join(self.outpath_val, outname + '.pts'), 
                                    os.path.join(self.imgpath_val, outname + '.png'))

            val_dict['people'].append({  
                            'beta': beta.tolist(),
                            'pose': pose.tolist(),
                            'name': outname,
                        }) 
            print('{} in total {} for gender {} in {}'.format(offset + i, nb_generated_humans, gender, stage))

        return

    def unify_database(self, dta='spe_dataset_train_noRotation1', dtb='spe_dataset_val_noRotation1', 
                dtout='spe_dataset_train_noRotation2'):
        import shutil
        def move_file(pathin, pathout, namein, nameout=None):
            if nameout is None:
                nameout = namein
            # try:
            of = os.path.join(pathin, namein + '.pts')
            nf = os.path.join(pathout, nameout + '.pts')
            shutil.copy(of, nf)

            of = os.path.join(pathin, 'image', namein + '.png')
            nf = os.path.join(pathout, 'image', nameout + '.png')
            shutil.copy(of, nf)    

            print('{} copied'.format(of)) 
            # except BaseException as E:
            #     print('no file: {}'.format(namein))

        patha = os.path.join(self.data_root, dta)
        pathb = os.path.join(self.data_root, dtb)
        pathout = os.path.join(self.data_root, dtout)
        
        imout = os.path.join(pathout, 'image')
        if not os.path.exists(imout):
            os.makedirs(imout)

        jfa = os.path.join(patha, 'data.json')
        with open(jfa) as ja: 
            data1 = json.load(ja) 
            for p in data1['people']:
                move_file(patha, pathout, p['name'])        

            offset = data1['info']['nb_generated_humans']
            jfb = os.path.join(pathb, 'data.json')

            with open(jfb) as jb:  
                data2 = json.load(jb)
                data1['info']['nb_generated_humans'] += data2['info']['nb_generated_humans']
                for p in data2['people']:
                    fname = p['name']
                    idx = fname.find('_')
                    tname = '{}{}'.format(offset + int(fname[:idx]), fname[idx:])
                    data1['people'].append({  
                        'beta': p['beta'],
                        'pose': p['pose'],
                        'name': tname,
                    }) 
                    move_file(pathb, pathout, fname, tname)

            with open(os.path.join(pathout, 'data.json'), 'w') as outfile:  
                json.dump(data1, outfile)

        print('database unification finished. ')

    def generate_database(self, beta_pose_info = None):
        ############### 
        # prepare dictionaries       
        self.imgpath_train =  os.path.join(self.outpath_train, 'image')
        self.imgpath_val =  os.path.join(self.outpath_val, 'image')   
        if not os.path.exists(self.imgpath_train):
            os.makedirs(self.imgpath_train)
        if not os.path.exists(self.imgpath_val):
            os.makedirs(self.imgpath_val)

        ###############
        # generate data
        train_dict, val_dict = self.setup_jason()  
        if self.gender == 'm':
            self._generate_database_surreal_(train_dict, val_dict, gender='m', offset = 0, bRotate = False, info=beta_pose_info)
        elif self.gender == 'f':
            self._generate_database_surreal_(train_dict, val_dict, gender='f', offset = 0, bRotate = False, info=beta_pose_info)
        else:
            self._generate_database_surreal_(train_dict, val_dict, gender='m', offset = 0, bRotate = False, info=beta_pose_info)            
            self._generate_database_surreal_(train_dict, val_dict, gender='f', offset = self.human_count, bRotate = False, info=beta_pose_info)              
        
        ##############
        # save json
        with open(os.path.join(self.outpath_train, 'data.json'), 'w') as outfile:  
            json.dump(train_dict, outfile)

        with open(os.path.join(self.outpath_val, 'data.json'), 'w') as outfile:  
            json.dump(val_dict, outfile)      

    def distill_database(self, db='spe_dataset_surreal_val_noRotation'):
        ''' 
            remove data if its image has been deleted manually. 
        '''

        patha = os.path.join(self.data_root, db)

        jfa = os.path.join(patha, 'data.json')
        ja = open(jfa)
        data = json.load(ja) 
        ja.close()

        removed_count = 0
        people = []

        for p in data['people']:
            im = os.path.join(patha, 'image', p['name'] + '.png')
            if os.path.exists(im):
                people.append(p)
            else:
                os.remove(os.path.join(patha, p['name'] + '.pts'))
                removed_count += 1 #
                print(removed_count)
        
        data['people'] = people
        data['info']['nb_generated_humans'] = data['info']['nb_generated_humans'] - removed_count
    
        with open(jfa, 'w') as outfile:  
            json.dump(data, outfile)

        print('{} people removed'.format(removed_count))

    def beta_pose_info_from(self, filename = './pose_select.npy'):
        '''
            there are 104 poses in pose_select.npy
            todo wait longke wang for correct npy file
        '''
        bpinfo = []
        pose_selection = np.load(filename) 

        poses = [i for i in self.database.keys() if "pose" in i]
        for j in range(np.shape(pose_selection)[0]):
            k = 0
            for nm_p in poses:
                if nm_p == 'pose_' + pose_selection[j][0]:
                    pose_id = k
                    break
                k = k + 1
            
            frame_id = int(pose_selection[j][1])
            bpinfo.append({'poseid': pose_id, 'frameid': frame_id, 'betaid': -1})
        
        return bpinfo

if __name__ == '__main__':
    if not os.path.exists(args.smpl_model):
        args.smpl_model = '../data/neutral_smpl_with_cocoplus_reg.txt' 
    
    DATA_ROOT = args.data_root    

    outpath_train = os.path.join(DATA_ROOT, '{}_{}_{}'.format(args.database_train, args.data_type, args.gender)) 
    outpath_val = os.path.join(DATA_ROOT, '{}_{}_{}'.format(args.database_val, args.data_type, args.gender)) 

    dg = SpeDataGenerator(DATA_ROOT, outpath_train, outpath_val, args.op)
    if args.op=='unify':
        dg.unify_database(dta='spe_dataset_val_w_m_1', dtb='spe_dataset_val_w_m_2', 
                dtout='spe_dataset_val_w_m_3')
    elif args.op=='distill':
        dg.distill_database(db = 'spe_dataset_train_w_m5')
    else: # 'generate'
        bpinfo = None
        #bpinfo = dg.beta_pose_info_from('./pose_select.npy')
        #bpinfo = [{'poseid': 168, 'frameid': 175, 'betaid': -1}]

        if args.data_type == 'f':
            dg.view_directions = np.array( [[0.0, 0.0, -1.0]] )
        elif args.data_type == 'fb':
            dg.view_directions = np.array( [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] )
        
        # else # dg.view_directions == None for whole view        
        dg.bent = True #for bent human
        dg.generate_database(beta_pose_info = bpinfo)


