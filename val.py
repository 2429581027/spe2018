'''
    file:   val.py

    train for shape & pose parameters regression

    date:   2018_11
    author: Junjie Cao
    based on pointnet.pytorch, pointnet++.pytorch & pytorch_hmr
'''

import os
import random
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import visdom
import numpy as np
from torch.utils.data import DataLoader

from config import args
import trimesh
from model import SPENet
from datasets import SurrealDataset

from util.util import copy_state_dict, batch_rodrigues

from train import SPETrainer

#import pdb; pdb.set_trace()

class SPEValidator(SPETrainer):
    def __init__(self, bTrain = False, bVal = True):
        super().__init__(bTrain, bVal)
        vert_count = 6890
        self.ave_pervert = np.zeros(vert_count)     
        #self._init_weight_(bUniform = True)

    def batch_3d_l2_loss(self, verts, predict_verts):
        B = verts.shape[0]
        N = verts.shape[1]
        k = B * N * 3    

        dif = self.loss_fun(verts, predict_verts)
        dif = dif.sum(2) * self.pervertex_weight  # N x 6890 x 3 => N x 6890 x 1
        
        self.ave_pervert += dif.sum(0).detach().cpu().numpy()/B

        return  dif.sum() / k

    def batch_save_obj_png(self, pts, verts, outputs, name, valmsg, shapepara, posepara): 
        '''
        pts: input points
        verts: vertices GT mesh, i.e. generated mesh from shape and pose parameters of pts
        '''
        (predict_shape, predict_pose, predict_verts, predict_j3d, decoded_verts) = outputs      
        for i in range(predict_verts.shape[0]):              
            fname = '{}_s{}_p{}_3d{}_3dd{}'.format(name[i], 
                                            valmsg['shape_loss'], valmsg['pose_loss'],
                                            valmsg['3d_loss'], valmsg['3d_decoded_loss'])

            self.spenet.module.smpl.save_obj(verts[i].detach().cpu().numpy(), 
                                      os.path.join(self.val_folder, name[i] + '.obj'))


            with open(os.path.join(self.val_folder, name[i] + '.txt'), 'w') as fp:                
                fp.write(fname)
                fp.write("\n beta GT: \n")
                fp.write(str(predict_shape[i].detach().cpu().numpy()))
                fp.write("\n beta predicated: \n")
                fp.write(str(shapepara[i].detach().cpu().numpy()))


            self.spenet.module.smpl.save_obj(predict_verts[i].detach().cpu().numpy(), 
                                      os.path.join(self.val_folder, fname + '.obj'))

            if args.decoder != 'None':
                self.spenet.module.smpl.save_obj(decoded_verts[i].detach().cpu().numpy(), 
                                      os.path.join(self.val_folder, fname + '-decoded.obj'))                
                self.spenet.module.smpl.save_png(pts[i].detach().cpu().numpy(), 
                                            predict_verts[i].detach().cpu().numpy(),
                                            decoded_verts[i].detach().cpu().numpy(), 
                                            os.path.join(self.val_folder, fname + '.png'))                                        
            else:
                self.spenet.module.smpl.save_png(pts[i].detach().cpu().numpy(), 
                            verts[i].detach().cpu().numpy(), #predict_verts[i].detach().cpu().numpy(),
                            predict_verts[i].detach().cpu().numpy(), 
                            os.path.join(self.val_folder, fname + '.png'))

    #@torch.no_grad() # this works for pytorh 0.4.1
    def val(self):
        with torch.no_grad():  # this works for pytorh 0.4.0
            torch.backends.cudnn.benchmark = True
            #self.spenet.train()
            self.spenet.eval()

            self.val_folder = os.path.join(args.outf, 'val')
            if not os.path.exists(self.val_folder):
                os.makedirs(self.val_folder)

            ave_loss, ave_shape, ave_pose = 0.0, 0.0, 0.0
            ave_3d, ave_3d_decoded, ave_j3d = 0.0, 0.0, 0.0
            
            val_i = 0
            for data in self.valloader:
                pts, shapepara, posepara = data['pts'], data['shape'], data['pose']          
                name = data['name']
                if torch.cuda.is_available():
                    pts, shapepara, posepara = pts.cuda(), shapepara.cuda(), posepara.cuda()
                                            
                outputs = self.spenet(pts, inBeta = shapepara, inTheta = posepara)

                loss_dict = self._calc_loss(outputs, pts, shapepara, posepara, data)
                loss_shape, loss_pose = loss_dict["shape"], loss_dict["pose"] 
                loss_3d, loss_3d_decoded = loss_dict["3d"], loss_dict["3d_decoded"] 
                loss_j3d = loss_dict["j3d"] 
                #loss_chamfer = loss_dict["chamfer"] 

                loss = loss_shape + loss_pose + loss_3d + loss_3d_decoded + loss_j3d # + loss_chamfer 
                
                ave_loss += loss                
                ave_shape += loss_shape
                ave_pose += loss_pose
                ave_3d += loss_3d
                ave_3d_decoded += loss_3d_decoded                    
                ave_j3d += loss_j3d                

                val_msg = OrderedDict(
                    [
                        ('time',datetime.datetime.now().strftime('%m-%d-%H:%M')),
                        ('loss', "{0:.3f}".format(loss.item())),                 
                        ('shape_loss',"{0:.3f}".format(loss_shape.item())),
                        ('pose_loss', "{0:.4f}".format(loss_pose.item())),
                        ('3d_loss', "{0:.7f}".format(loss_3d.item())),
                        ('3d_decoded_loss', "{0:.7f}".format(loss_3d_decoded.item())),
                        ('j3d_loss', "{0:.4f}".format(loss_j3d.item())),
                        #('chamfer_loss', "{0:.4f}".format(loss_chamfer.item())),   
                    ]
                )
                print(val_msg)
                self.batch_save_obj_png(pts, data['verts'], outputs, name, val_msg, shapepara, posepara)
                val_i += 1
                # End of for data in self.valloader:
            
            v = self.ave_pervert
            v = v / val_i            
            v= v/v.max() * 0.9 + 0.1
            print( 'min error-{}, max error-{}, after normalization'.format(v.min(), v.max()) )
            np.savez_compressed(os.path.join(self.val_folder, 'pervertex_weight'), pervertex_weight=v)
            mesh = trimesh.load('mesh.obj', process=False)
            for i in range(len(mesh.vertices)):
                mesh.visual.vertex_colors[i] = np.array([v[i]*255,0,0,255], dtype = np.uint8)
            mesh.scene().show()
            png = mesh.scene().save_image(resolution=[800, 800],visible=True)            
            with open( os.path.join(self.val_folder, 'pervertex_weight.png'), 'wb') as f:
                f.write(png)
                f.close()


            ave_loss = ave_loss / val_i
            ave_shape = ave_shape / val_i
            ave_pose = ave_pose / val_i
            ave_3d = ave_3d / val_i
            ave_3d_decoded = ave_3d_decoded / val_i
            ave_j3d = ave_j3d / val_i
            #ave_chamfer = ave_chamfer / val_i

            val_msg = OrderedDict(
                [
                    ('time',val_msg['time']),
                    ('ave_loss', "{0:.4f}".format(ave_loss.item())),                 
                    ('ave_shape_loss',"{0:.3f}".format(ave_shape.item())),
                    ('ave_pose_loss', "{0:.4f}".format(ave_pose.item())),
                    ('ave_3d_loss', "{0:.7f}".format(ave_3d.item())),
                    ('ave_3d_decoded_loss', "{0:.7f}".format(ave_3d_decoded.item())),
                    ('ave_j3d_loss', "{0:.4f}".format(ave_j3d.item())),
                    #('ave_chamfer_loss', "{0:.4f}".format(ave_chamfer.item())),                
                ]
            )


            val_msg['title'] = 'spe_{}_s{}_p{}_3d{}_3dd{}'.format(
                                val_msg['ave_loss'], 
                                val_msg['ave_shape_loss'], val_msg['ave_pose_loss'], 
                                val_msg['ave_3d_loss'], val_msg['ave_3d_decoded_loss'])
            print(val_msg)
            with open(os.path.join(self.val_folder, val_msg['title'] + '.txt'), 'w') as fp:
                fp.write(str(val_msg))

        print('val finished.')

def test_batch_3d_l2_loss():
    from torch.autograd import Variable
    batch = 2
    nsamples = 2400
    verts = Variable(torch.ones(batch,nsamples,3)).cuda()
    predict_verts = Variable(torch.zeros(batch,nsamples,3)).cuda()

    idx = np.random.choice(6890, nsamples, replace=False)
    idx = torch.from_numpy(idx)
    idx = idx.unsqueeze(0).expand(batch, nsamples).contiguous()

    validator = SPEValidator(bTrain = False, bVal = True)

    nruns = 3
    for i in range(nruns):
        validator.batch_3d_l2_loss(verts, predict_verts, idx)
    
    v = validator.ave_pervert
    v = v / nruns
    v=v/v.max()*255
    print(v.max())

    np.savez_compressed('pervertex_weight', pervertex_weight=v)
    loaded = np.load('pervertex_weight.npz')
    vv = loaded['pervertex_weight']
    print(vv.max())

    mesh = trimesh.load('mesh.obj', process=False)

    for i in range(len(mesh.vertices)):
        mesh.visual.vertex_colors[i] = np.array([vv[i],0,0,255], dtype = np.uint8)
    mesh.show()
    

if __name__ == '__main__':    
    args.batch_size = 1
    args.pervertex_weight = 'None'
    args.ls = 'L2'
    #args.with_stn = 'None'
    args.model = './model/SPENet_pointnetmini_PointGenCon_108_0.043558_s0.039577_p0.002866_3d0.000499_decoded0.000263_j0.00035_r0.000010.pkl'
    
    print (args)
    args.outf = '{}_{}_{}_{}_{}'.format(args.outf, args.network, args.encoder, args.decoder, datetime.datetime.now().strftime('%m-%d-%H:%M'))
    
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    with open(os.path.join(args.outf, 'config.txt'), 'w') as fp:
        fp.write(str(args))

    #test_batch_3d_l2_loss()

    args.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    args.decoded_ratio = args.threeD_ratio
    validator = SPEValidator(bTrain = False, bVal = True)
    validator.val()
