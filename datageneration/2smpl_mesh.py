'''
Generate meshes (in .OBJ format) with varied shapes and poses based on SMPL model and pose parameters provided by 
surreal (Learning from Synthetic Humans, cvpr 17) [smpl_data]

Historybased:
3. 2018 Junjie Cao

'''
import random
import os
import json
import numpy as np
import torch
import torch.nn as nn
from config import args
from SMPL import SMPL
def smpl2mesh(smpl, gender, shapefile, posepath, outpath, jasondata, nsample_pose=2, nsample_shape=2):
    '''
        generate human mesh from smpl parameters.
    '''
    human_shape = np.load(shapefile)
    
    outcount = 0
    for root, dirs, files in os.walk(posepath):
        for name in files:
            filename = os.path.join(posepath, name)
            #print(name)

            if not name.startswith('pose_'):
                continue

            if name.startswith('pose_ung_'):
                continue

            human_pose = np.load(filename)
            tmpname = name[0:len(name)-4] + '-pose_'
            outpredix = tmpname

            poseidx = [int(human_pose.shape[0]*random.random()) for i in range(nsample_pose)]
            shapeidx = [int(human_shape.shape[0]*random.random()) for i in range(nsample_shape)]

            for poseid in poseidx:
                vpose = human_pose[poseid,:]
                if torch.cuda.is_available():                    
                    vpose = torch.tensor(np.array([vpose])).float().to(device)
                else:
                    vpose = torch.tensor(np.array([vpose])).float()

                for shapeid in shapeidx:
                    vbeta = human_shape[shapeid, :]
                    if torch.cuda.is_available():
                        vbeta = torch.tensor(np.array([vbeta])).float().to(device)
                    else:
                        vbeta = torch.tensor(np.array([vbeta])).float()

                    outname = str(poseid) + '_shape_' + str(shapeid) + '_' + gender
                    #outfile = outpredix + outname  + '.obj'
                    outfile = os.path.join(outpath, outpredix + outname  + '.obj')
                    #print(outfile)  

                    jasondata['people'].append({  
                                            'shapepara': human_shape[shapeid, :].tolist(),
                                            'posepara': human_pose[poseid,:].tolist(),
                                            'name': tmpname + outname
                                        })  

                    verts, j, r = smpl(vbeta, vpose, get_skin = True)
                    smpl.save_obj(verts[0].cpu().numpy(), outfile)

                    print(outcount)
                    if outcount > 10000: 
                        return outcount
                    outcount +=1



    return outcount

def mesh_of_posefile(posefiles, posepath, shapepara, outpath, posestep):
    '''
        posefiles = ['01_01', '05_10']
    '''
    for posefile in posefiles:
        #pose_01_01.npy
        predix = 'pose_'
        filename = os.path.join(posepath, predix + posefile + '.npy')
        #print(filename)
        human_pose = np.load(filename)    

        #trans_01_01.npy
        # filename = os.path.join(posepath, 'trans_' + posefile + '.npy')
        # trans = np.load(filename)

        outcount = 0     
        for poseid in range(0, human_pose.shape[0], posestep):
            vpose = human_pose[poseid,:]
            vpose[0:3]=0.0 # cancel global transformation
            vpose = torch.tensor(np.array([vpose])).float()
            if torch.cuda.is_available():
                vpose = vpose.cuda()     
            verts, j, r = smpl(shapepara, vpose, get_skin = True)
            outfile = os.path.join(outpath, predix + posefile + '_p_' + str(poseid) + '.obj')
            smpl.save_obj(verts[0].cpu().numpy(), outfile)
            print(outcount)
            outcount += 1
            
    return

def pose_browser():    
    #posefiles = ['01_01', '05_10']
    posefiles = ['01_02']

    outpath = 'spe_tmp/mesh'
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ########################       
    shapeid = 0
    shapepara = human_shape[shapeid, :]
    shapepara = torch.tensor(np.array([shapepara])).float()
    if torch.cuda.is_available():
        shapepara = shapepara.cuda()        

    mesh_of_posefile(posefiles, posepath, shapepara, outpath, posestep=100)


def pose_with_varied_shapes(outpath):    
    posefiles = ['01_02'] #'01_01','01_02' 
    # posestep = 100 # 27 + 48
    # shapestep = 10 # 200
    posestep = 100 # 27 + 48
    shapestep = 10 # 200

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    outcount = 0 
    for posefile in posefiles:
        #pose_01_01.npy
        predix = 'pose_'
        filename = os.path.join(posepath, predix + posefile + '.npy')
        #print(filename)
        try:
            human_pose = np.load(filename)    
        except OSError as e:
            print(e.errno)  
            continue

        for poseid in range(0, human_pose.shape[0], posestep):
            vpose = human_pose[poseid,:]
            vpose[0:3]=0.0 # cancel global transformation
            vpose = torch.tensor(np.array([vpose])).float()
            if torch.cuda.is_available():
                vpose = vpose.cuda()     

            for shapeid in range(0, human_shape.shape[0], shapestep):            
                shapepara = human_shape[shapeid, :]
                shapepara = torch.tensor(np.array([shapepara])).float()
                if torch.cuda.is_available():
                    shapepara = shapepara.cuda() 

                verts, j, r = smpl(shapepara, vpose, get_skin = True)

                outname = predix + posefile + '_p' + str(poseid) + '_s' + str(shapeid)
                outfile = os.path.join(outpath, outname + '.obj')
                smpl.save_obj(verts[0].cpu().numpy(), outfile)

                data['people'].append({  
                        'shapepara': human_shape[shapeid, :].tolist(),
                        'posepara': vpose.tolist(),
                        'name': outname
                    })  

                print("num_{}_{}".format(outcount, outname))
                outcount += 1

    data['info'][0]['count'] = outcount

    ##############
    with open(os.path.join(DATA_ROOT, 'spe_data/data.json'), 'w') as outfile:  
        json.dump(data, outfile)

    return


def old_generator():
    ###############
    nsample_pose, nsample_shape = 1000, 10
    info = data['info']
    for val in info:
        outcount = smpl2mesh(smpl, val['gender'], val['shape'], posepath, outpath, data, nsample_pose, nsample_shape) 
        val['count'] = outcount

    ##############
    with open(os.path.join(DATA_ROOT, 'spe_data/data.json'), 'w') as outfile:  
        json.dump(data, outfile)
  
    
if __name__ == '__main__':
    ########
    DATA_ROOT = '../../data' #../../data' # /Users/jjcao/data/    
    # 2,667 (1261 pose_* + 1406 pose_ung_*) pose sequence files, each contains a series of continue poses 
    posepath = os.path.join('../../data', 'smpl_data') #../../data, /data
    #posepath = os.path.join(DATA_ROOT, 'smpl_data_test')
    outpath = os.path.join(DATA_ROOT, 'spe_data/mesh') 

    ##################
    human_shape = np.load(os.path.join(posepath, 'femaleshapes.npy'))
    smpl = SMPL(args.smpl_model, obj_saveable = True) #'./model/neutral_smpl_with_cocoplus_reg.txt'
    if torch.cuda.is_available():
        smpl = smpl.cuda()  

    ##################
    #pose_browser()

    ##################
    data = {}  
    data['count_resample']=1500
    data['info'] = []
    data['people'] = []  
 
    data['info'].append({'gender': 'female',
                        'smpl':  args.smpl_model,
                        'shape': os.path.join(posepath, 'femaleshapes.npy'),
                    })
    # data['info'].append({'gender': 'male',
    #                     'smpl':  args.smpl_model,
    #                 'shape': os.path.join(posepath, 'maleshapes.npy'),
    #                 })    
                       
    pose_with_varied_shapes(outpath)

    ################## 
    #old_generator()
